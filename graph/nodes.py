"""LangGraph node functions — the brain of each state transition.

Each node is an async function that:
  1. Receives the current AgentState
  2. Performs its specific logic (plan, execute, reflect, replan, etc.)
  3. Returns a partial state dict with the fields it updated

Node factories (create_*) accept dependencies like LLM and ToolRegistry,
returning the actual node function that LangGraph will call.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel

from agent.state import AgentState
from runtime.shell_platform import get_shell_environment
from prompts.system import SYSTEM_PROMPT
from prompts.templates import (
    PLAN_SYSTEM_PROMPT,
    PLAN_USER_TEMPLATE,
    STEP_CONTEXT_TEMPLATE,
    RETRY_CONTEXT_TEMPLATE,
    REACT_CONTEXT_TEMPLATE,
    ASK_CONTEXT_TEMPLATE,
    REFLECT_SYSTEM_PROMPT,
    REFLECT_USER_TEMPLATE,
    REPLAN_SYSTEM_PROMPT,
    REPLAN_USER_TEMPLATE,
    FINISH_SYSTEM_PROMPT,
    FINISH_USER_TEMPLATE,
)


# ═══════════════════════════════════════════════════════════════════
# ── Init Node ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

async def init_node(
    state: AgentState,
    memory_manager=None,
) -> Dict[str, Any]:
    """Initialize agent state — run once at the very beginning.

    V2.1 enhancements:
      - Respects mode: "plan" → phase="planning", "ask"/"agent" → phase="executing"
      - Injects session_context from MemoryManager (cross-turn recall).

    V3: Removed intent_class regex pre-classification. The model decides
        whether to use tools — no automatic routing shortcuts.
    """
    mode = state.get("mode", "agent")
    task = state["task"]

    # ── Cross-turn memory recall (V2.1) ───────────────────
    session_context = ""
    if memory_manager is not None:
        try:
            session_context = memory_manager.build_context_for_task(task)
        except Exception:
            session_context = ""  # Memory errors should not block execution

    # ── Build system prompt with session context ──────────
    system_content = SYSTEM_PROMPT
    if session_context:
        system_content += "\n\n" + session_context

    # ── Phase: plan mode starts planning, others start executing ──
    phase = "planning" if mode == "plan" else "executing"

    return {
        "phase": phase,
        "iteration": 0,
        "error_message": "",
        "final_answer": "",
        "plan": [],
        "current_step_index": 0,
        "step_retry_count": 0,
        "tool_history": [],
        "session_context": session_context,
        "messages": [
            SystemMessage(content=system_content),
            HumanMessage(content=task),
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# ──  Plan Node  ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# Steps that look like they require a tool but don't name one specifically
_FORBIDDEN_STEP_PATTERNS = [
    r"^look at", r"^check out", r"^examine", r"^inspect",
    r"^understand", r"^analyze", r"^consider",
]

_REQUIRED_TOOL_KEYWORDS = [
    "read_file", "write_file", "edit_file",
    "grep_search", "glob_search", "shell_execute",
    "search for", "read the", "find all", "list all",
    "execute", "run the", "create a", "write a",
    "edit the", "modify", "install", "test",
]


def _validate_step(step: dict) -> list[str]:
    """Validate a single plan step. Returns list of warnings (empty = valid)."""
    warnings = []
    desc = step.get("description", "").lower().strip()

    if len(desc) < 5:
        warnings.append(f"Step {step.get('id', '?')} description is too short")

    for pattern in _FORBIDDEN_STEP_PATTERNS:
        if re.match(pattern, desc):
            warnings.append(
                f"Step {step.get('id', '?')}: '{desc[:40]}...' is too vague. "
                f"Name a specific tool or file."
            )
            break

    has_tool_hint = any(kw in desc for kw in _REQUIRED_TOOL_KEYWORDS)
    if not has_tool_hint:
        warnings.append(
            f"Step {step.get('id', '?')}: no tool keyword found. "
            f"Consider mentioning read_file, write_file, edit_file, "
            f"grep_search, glob_search, or shell_execute."
        )

    return warnings


async def plan_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Generate a structured task plan using the LLM.

    Enhancements over V1:
      - Validates each step for concreteness
      - Injects workspace context
      - Better JSON extraction (handles more LLM output quirks)
      - Fallback with validation warnings
    """
    workspace_path = state.get("workspace_path", ".")
    validation_warnings: list[str] = []

    try:
        plan_response = await llm.ainvoke([
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=PLAN_USER_TEMPLATE.format(
                task=state["task"],
                workspace_path=workspace_path,
                shell_environment=get_shell_environment().agent_prompt_section.strip(),
            )),
        ])

        plan_text = plan_response.content.strip()
        plan = _extract_json_array(plan_text)

        # Validate
        if not isinstance(plan, list):
            raise ValueError("Plan is not a list")

        # Allow empty plans — the LLM may return [] for non-coding queries
        # (e.g. conversational questions that reached plan_node)
        if len(plan) == 0:
            return {
                "phase": "executing",
                "plan": [],
                "current_step_index": 0,
                "step_retry_count": 0,
                "error_message": "",
            }

        if len(plan) > 10:
            # Cap at 10 steps to keep plans manageable
            plan = plan[:10]
            validation_warnings.append("Plan truncated to 10 steps maximum")

        # Normalize + validate each step
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} is not a dict: {step}")
            if "id" not in step:
                step["id"] = str(i + 1)
            step.setdefault("status", "pending")
            step.setdefault("retry_count", 0)
            step.setdefault("max_retries", 2)
            # Validate
            warnings = _validate_step(step)
            validation_warnings.extend(warnings)

        return {
            "phase": "executing",
            "plan": plan,
            "current_step_index": 0,
            "step_retry_count": 0,
            "error_message": (
                "; ".join(validation_warnings) if validation_warnings else ""
            ),
        }

    except (json.JSONDecodeError, ValueError) as e:
        fallback_plan = [
            {
                "id": "1",
                "description": state["task"],
                "status": "pending",
                "retry_count": 0,
                "max_retries": 2,
            }
        ]
        return {
            "phase": "executing",
            "plan": fallback_plan,
            "current_step_index": 0,
            "step_retry_count": 0,
            "error_message": f"Plan parsing fell back to single-step: {e}",
        }


# ═══════════════════════════════════════════════════════════════════
# ── Audit Plan Node (Phase 5–8: Intent Auditor integration) ──────
# ═══════════════════════════════════════════════════════════════════

async def audit_plan_node(
    state: AgentState,
    llm: BaseChatModel,
    settings=None,
) -> Dict[str, Any]:
    """Check each plan step for Goal-Plan alignment using Two-Layer Auditor.

    Runs AFTER plan_node and BEFORE execute_node in plan mode.
    Controlled by settings.intent_auditor_enabled (default: True).

    V4: Uses TwoLayerAuditor (Embedding Filter → NLI Judge).
        - Batch embeds plan steps for fast similarity check
        - Only invokes LLM for gray-zone steps
        - If all steps rejected: routes to finish (direct answer fallback)

    Returns updated state with audit results in plan step metadata.
    """
    from intent_auditor.two_layer import create_two_layer_auditor
    from intent_auditor.intent_auditor import is_predicted_error

    # ── Feature flag guard ──────────────────────────────────
    if settings is None:
        try:
            from config.settings import settings as _settings
            settings = _settings
        except Exception:
            pass

    if settings is not None and not getattr(settings, "intent_auditor_enabled", False):
        # Pass-through: no auditing
        return {
            "phase": "executing",
            "error_message": state.get("error_message", ""),
        }

    plan = state.get("plan", [])
    task = state.get("task", "")
    threshold = getattr(settings, "intent_auditor_threshold", 0.6) if settings else 0.6

    if not plan:
        return {"phase": "executing"}

    # ── Gather pending steps ────────────────────────────────
    pending_steps = [
        s for s in plan if s.get("status") != "done"
    ]
    if not pending_steps:
        return {"phase": "executing"}

    # ── V4: Two-Layer batch audit ───────────────────────────
    try:
        use_two_layer = (
            getattr(settings, "auditor_two_layer", True) if settings else True
        )
        # Runtime override via env var (pytest conftest)
        env_tl = os.environ.get("AUDITOR_TWO_LAYER", "").lower()
        if env_tl in ("false", "0", "no", "off"):
            use_two_layer = False
    except Exception:
        use_two_layer = True

    rejected_count = 0

    if use_two_layer:
        # ── Two-layer: Embedding filter + NLI batch ─────────
        try:
            auditor = create_two_layer_auditor(llm=llm)
            pairs = [(task, s.get("description", "")) for s in pending_steps]
            results = await auditor.audit_batch(pairs)

            for step, result in zip(pending_steps, results):
                is_error = (
                    result.label == "contradiction"
                    or result.score < threshold
                )
                step["audit_result"] = {
                    "label": result.label,
                    "score": result.score,
                    "reason": result.reason,
                    "rejected": is_error,
                    "path": result.path,          # "embed" or "nli"
                    "cosine_sim": result.cosine_sim,
                }
                if is_error:
                    rejected_count += 1

            # Log bypass rate for observability
            bypass = auditor.nli_bypass_rate
            if bypass > 0:
                import logging
                _log = logging.getLogger(__name__)
                _log.debug(
                    "TwoLayerAuditor bypass rate: %.0f%% (%d/%d steps via embed only)",
                    bypass * 100,
                    auditor.stats["embed_hits"],
                    auditor.stats["total"],
                )

        except Exception as e:
            # Embedding layer failed → fall back to single-layer NLI
            import logging
            _log = logging.getLogger(__name__)
            _log.warning("TwoLayerAuditor failed, falling back to single-layer NLI: %s", e)
            from intent_auditor.intent_auditor import audit_intent as _audit_single

            for step in pending_steps:
                try:
                    result = await _audit_single(
                        goal=task,
                        plan_step=step.get("description", ""),
                        llm=llm,
                    )
                    is_error = is_predicted_error(result, threshold=threshold)
                    step["audit_result"] = {
                        "label": result.label,
                        "score": result.score,
                        "reason": result.reason,
                        "rejected": is_error,
                        "path": "nli",
                        "cosine_sim": 0.0,
                    }
                    if is_error:
                        rejected_count += 1
                except Exception as e2:
                    step["audit_result"] = {
                        "label": "neutral", "score": 0.5,
                        "reason": f"Auditor error (allowed): {e2}",
                        "rejected": False, "path": "error", "cosine_sim": 0.0,
                    }
    else:
        # ── Single-layer: pure NLI (legacy path) ────────────
        from intent_auditor.intent_auditor import audit_intent

        for step in pending_steps:
            step_desc = step.get("description", "")
            try:
                result = await audit_intent(
                    goal=task,
                    plan_step=step_desc,
                    llm=llm,
                )
                is_error = is_predicted_error(result, threshold=threshold)
                step["audit_result"] = {
                    "label": result.label,
                    "score": result.score,
                    "reason": result.reason,
                    "rejected": is_error,
                    "path": "nli",
                    "cosine_sim": 0.0,
                }
                if is_error:
                    rejected_count += 1
            except Exception as e:
                step["audit_result"] = {
                    "label": "neutral",
                    "score": 0.5,
                    "reason": f"Auditor error (allowed): {e}",
                    "rejected": False,
                    "path": "error",
                    "cosine_sim": 0.0,
                }

    # ── Safety net: if ALL rejected purely by embedding, verify with NLI
    if rejected_count == len(pending_steps):
        all_embed = all(
            s.get("audit_result", {}).get("path") == "embed"
            for s in pending_steps
        )
        if all_embed:
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                "All %d steps rejected by embedding — double-checking with NLI",
                rejected_count,
            )
            from intent_auditor.intent_auditor import audit_intent as _nli_verify

            rejected_count = 0
            for step in pending_steps:
                try:
                    result = await _nli_verify(
                        goal=task,
                        plan_step=step.get("description", ""),
                        llm=llm,
                    )
                    is_error = is_predicted_error(result, threshold=threshold)
                    old_csim = step.get("audit_result", {}).get("cosine_sim", 0.0)
                    step["audit_result"] = {
                        "label": result.label,
                        "score": result.score,
                        "reason": f"[Embed+ NLI verify] {result.reason}",
                        "rejected": is_error,
                        "path": "nli",
                        "cosine_sim": old_csim,
                    }
                    if is_error:
                        rejected_count += 1
                except Exception:
                    step["audit_result"] = {
                        "label": "neutral", "score": 0.5,
                        "reason": "NLI fallback — allowing step.",
                        "rejected": False, "path": "nli",
                        "cosine_sim": step.get("audit_result", {}).get("cosine_sim", 0.0),
                    }

    # ── If ALL steps rejected, route directly to finish ─────
    if rejected_count == len(pending_steps):
        return {
            "phase": "done",
            "plan": plan,
            "current_step_index": len(plan),
            "error_message": (
                f"[Intent Auditor] All {rejected_count} pending steps rejected "
                f"as misaligned with goal. Routing to finish for direct answer."
            ),
        }

    return {
        "phase": "executing",
        "plan": plan,
        "error_message": state.get("error_message", ""),
    }


# ── Thought extraction for Intent Auditor (agent/ask mode) ──────────

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _extract_thought(content: str) -> str | None:
    """Extract the agent's Thought: from response content.

    The REACT_CONTEXT_TEMPLATE and ASK_CONTEXT_TEMPLATE now require
    the model to write "Thought: <reasoning>" before any tool call.
    This function extracts that reasoning for Intent Auditor checking.

    Returns None if no Thought: line is found.
    """
    if not content:
        return None
    m = _THOUGHT_RE.search(content)
    if m:
        thought = m.group(1).strip()
        if len(thought) >= 10:  # Minimum meaningful thought
            return thought
    return None


# ═══════════════════════════════════════════════════════════════════
# ── Execute Node (factory) ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# Number of times to retry the LLM call itself on transient errors
_LLM_CALL_MAX_RETRIES = 3

def create_execute_node(
    llm_with_tools: BaseChatModel,
    context_manager=None,
):
    """Create the execute_node.

    Enhancements over V2:
      - Retry-aware: when phase="retry", injects error feedback into context
      - Dual-mode: react mode uses REACT_CONTEXT_TEMPLATE (free ReAct)
      - V2: delegates message truncation to ContextManager when provided
      - LLM call retry on transient API errors (RateLimitError, etc.)
      - Tracks step_start_tool_count for heuristic error detection in reflect_node
    """

    async def execute_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        step_idx = state["current_step_index"]
        iteration = state.get("iteration", 0)
        phase = state.get("phase", "")
        error_message = state.get("error_message", "")
        tool_history_len = len(state.get("tool_history", []))
        mode = state.get("mode", "agent")
        session_context = state.get("session_context", "")

        # ── Track step boundary for heuristic reflect ──────
        is_new_step = phase not in ("retry",)
        step_start_tool_count = (
            tool_history_len
            if is_new_step
            else state.get("step_start_tool_count", 0)
        )

        # ── Build step context (mode-dependent) ─────────────
        if mode == "ask":
            # Ask mode: read-only exploration, answer directly when possible
            step_context = ASK_CONTEXT_TEMPLATE.format(
                task=state["task"],
            )
        elif mode == "agent" or mode == "react":
            # Agent mode (V3 default): free-form context, model decides tool usage
            # React is a deprecated alias, treated identically
            task_desc = state["task"]
            if session_context:
                task_desc = f"{session_context}\n\n## Current Task\n{task_desc}"
            step_context = REACT_CONTEXT_TEMPLATE.format(
                task=task_desc,
            )
        elif plan and step_idx < len(plan):
            current_step = plan[step_idx]
            plan_lines = _build_plan_summary_lines(plan, current_step)
            plan_summary = "\n".join(plan_lines)

            if phase == "retry":
                retry_num = state.get("step_retry_count", 1)
                step_tool_errors = _extract_step_tool_errors(state, step_start_tool_count)
                error_context = error_message + "\n\n" + step_tool_errors

                step_context = RETRY_CONTEXT_TEMPLATE.format(
                    task=state["task"],
                    plan_summary=plan_summary,
                    current_step_id=current_step.get("id", "?"),
                    total_steps=len(plan),
                    step_description=current_step.get("description", ""),
                    retry_number=retry_num,
                    error_context=error_context,
                )
            else:
                step_context = STEP_CONTEXT_TEMPLATE.format(
                    task=state["task"],
                    plan_summary=plan_summary,
                    current_step_id=current_step.get("id", "?"),
                    total_steps=len(plan),
                    step_description=current_step.get("description", ""),
                )
        else:
            # Empty plan fallback — model decides whether tools are needed (V3)
            task_desc = state["task"]
            if session_context:
                task_desc = f"{session_context}\n\n## Current Task\n{task_desc}"
            step_context = f"Complete this task: {task_desc}"

        # ── Inject session context for plan mode ─────────────
        if mode == "plan" and session_context and "Recent Session History" in session_context:
            step_context = session_context + "\n\n" + step_context

        context_message = HumanMessage(content=step_context)

        # ── Prepare messages (V2: use ContextManager if available) ──
        messages = state["messages"]

        if context_manager is not None:
            # V2: delegate to ContextManager for intelligent compression
            try:
                invoke_messages = await context_manager.prepare_messages(
                    messages,
                    extra_context=step_context,
                )
            except Exception:
                # Fallback to V1 behavior on error
                invoke_messages = _v1_truncate(messages, context_message)
        else:
            # V1: simple truncation at MAX_MSG_COUNT
            invoke_messages = _v1_truncate(messages, context_message)

        # ── Call LLM with retry on transient errors ──────────
        last_error = None
        for llm_attempt in range(_LLM_CALL_MAX_RETRIES):
            try:
                response = await llm_with_tools.ainvoke(invoke_messages)
                break
            except Exception as e:
                last_error = e
                if llm_attempt < _LLM_CALL_MAX_RETRIES - 1:
                    await asyncio.sleep(1.0 * (llm_attempt + 1))
        else:
            error_text = (
                f"LLM API call failed after {_LLM_CALL_MAX_RETRIES} attempts: {last_error}. "
                f"The agent cannot continue this step."
            )
            return {
                "messages": [context_message, AIMessage(content=error_text)],
                "iteration": iteration + 1,
                "step_start_tool_count": step_start_tool_count,
                "error_message": str(last_error),
            }

        # ── V2: estimate tokens for budget tracking ──────────
        token_estimate = _estimate_messages_tokens(invoke_messages)

        # ═══════════════════════════════════════════════════════
        # V4: Two-Layer Intent Auditor check for agent/ask mode
        # ═══════════════════════════════════════════════════════
        audit_messages = []
        if mode in ("agent", "ask"):
            has_tool_calls = (
                hasattr(response, "tool_calls")
                and response.tool_calls
            )
            if has_tool_calls:
                # ── Check if auditor is enabled ──────────
                try:
                    from config.settings import settings as app_settings
                    auditor_enabled = getattr(
                        app_settings, "intent_auditor_enabled", True
                    )
                    use_two_layer = getattr(
                        app_settings, "auditor_two_layer", True
                    )
                except Exception:
                    auditor_enabled = True
                    use_two_layer = True

                if auditor_enabled:
                    thought = _extract_thought(response.content or "")
                    if thought:
                        try:
                            threshold = getattr(
                                app_settings, "intent_auditor_threshold", 0.6
                            )
                            # ── V4: Two-Layer audit ───────────
                            if use_two_layer:
                                from intent_auditor.two_layer import (
                                    create_two_layer_auditor,
                                )
                                auditor = create_two_layer_auditor(
                                    llm=llm_with_tools,
                                )
                                result = await auditor.audit(
                                    goal=state["task"],
                                    plan_step=thought,
                                )
                                is_error = (
                                    result.label == "contradiction"
                                    or result.score < threshold
                                )
                                audit_path = result.path  # "embed" or "nli"
                            else:
                                # Legacy: pure NLI
                                from intent_auditor.intent_auditor import (
                                    audit_intent, is_predicted_error,
                                )
                                nli_result = await audit_intent(
                                    goal=state["task"],
                                    plan_step=thought,
                                )
                                is_error = is_predicted_error(
                                    nli_result, threshold=threshold
                                )
                                audit_path = "nli"
                                # Package into same shape for feedback
                                result = nli_result

                            if is_error:
                                # Block tool execution:
                                # Replace response with no-tool-calls version
                                # and inject auditor feedback for the LLM
                                blocked_content = (
                                    f"{response.content}\n\n"
                                    f"---INTENT_AUDITOR---\n"
                                    f"⚠️ Your last action was BLOCKED.\n"
                                    f"Path: {audit_path}\n"
                                    f"Label: {result.label}\n"
                                    f"Score: {result.score:.2f}\n"
                                    f"Reason: {result.reason}\n"
                                    f"---END_INTENT_AUDITOR---"
                                )
                                response = AIMessage(
                                    content=blocked_content,
                                    # No tool_calls → routing won't go to tools
                                )

                                # Feedback message for the LLM to reconsider
                                task_preview = state["task"][:300]
                                audit_feedback = HumanMessage(content=(
                                    f"## ⚠️ Intent Auditor — Action Blocked\n\n"
                                    f"Your last Thought was flagged as "
                                    f"**{result.label}** "
                                    f"(alignment score: {result.score:.2f}).\n\n"
                                    f"**Decision path**: {audit_path}\n"
                                    f"**Reason**: {result.reason}\n\n"
                                    f"**Your goal was**: {task_preview}\n\n"
                                    f"Please reconsider. Does this action **directly** "
                                    f"serve the user's goal? If the task is a simple "
                                    f"question, answer directly without tools. "
                                    f"If you need a different approach, write a new "
                                    f"Thought: explaining how your next action aligns "
                                    f"with the goal."
                                ))
                                audit_messages = [audit_feedback]

                        except Exception:
                            # Auditor error → allow (conservative, don't block)
                            pass

        return {
            "messages": [context_message, response] + audit_messages,
            "iteration": iteration + 1,
            "step_start_tool_count": step_start_tool_count,
            "messages_token_estimate": token_estimate,
        }

    return execute_node


# ═══════════════════════════════════════════════════════════════════
# ── Tool Node (factory) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def create_tool_node(tool_registry):
    """Create the tool_node backed by our ToolRegistry.

    Enhancements over V1:
      - Tool results include success/failure markers for reflection
      - Timed execution
    """

    async def tool_node(state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {
                "error_message": "tool_node called but last message has no tool_calls",
            }

        tool_messages: List[ToolMessage] = []
        tool_records: List[dict] = []
        now = datetime.now().isoformat()

        for tc in last_message.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            try:
                tool = tool_registry.get(tool_name)
                result = await tool.execute(**tool_args)
                content = result.to_langchain_message()
                success = result.success
            except KeyError:
                content = f"Error: Unknown tool '{tool_name}'. Available: {tool_registry.tool_names}"
                success = False
            except Exception as e:
                content = f"Error: Tool '{tool_name}' failed: {e}"
                success = False

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id)
            )

            tool_records.append({
                "tool": tool_name,
                "args": dict(tool_args),
                "result": content[:500],
                "success": success,
                "timestamp": now,
            })

        existing_history = state.get("tool_history", [])
        new_history = existing_history + tool_records

        return {
            "messages": tool_messages,
            "tool_history": new_history,
        }

    return tool_node


# ═══════════════════════════════════════════════════════════════════
# ── Reflect Node (factory — LLM-powered with heuristic pre-check) ─
# ═══════════════════════════════════════════════════════════════════

def create_reflect_node(llm: BaseChatModel):
    """Create the reflection node. Uses heuristics + LLM to evaluate step completion.

    Enhancements over V2:
      - Heuristic pre-check: detects tool failures BEFORE asking LLM
      - Step-boundary-aware: uses step_start_tool_count to isolate current-step tools
      - Safe fallback: if reflection LLM fails, checks tool results instead of assuming success
      - Three-way outcome: success / retry / replan
      - Error classification: recoverable vs fatal vs wrong_approach
      - Per-step retry counter with max retries guard
    """

    async def reflect_node(state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        step_idx = state["current_step_index"]
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 30)
        step_retry_count = state.get("step_retry_count", 0)
        max_retries = state.get("max_retries_per_step", 2)
        tool_history = state.get("tool_history", [])
        step_start_tool_count = state.get("step_start_tool_count", 0)

        # ── Guard: max iterations ────────────────────────────
        if iteration >= max_iterations:
            return {
                "phase": "done",
                "error_message": (
                    f"Reached max iterations ({max_iterations}). "
                    f"Task may be incomplete."
                ),
            }

        # ── Guard: no plan ───────────────────────────────────
        if not plan or step_idx >= len(plan):
            return {"phase": "done"}

        current_step = plan[step_idx]
        step_desc = current_step.get("description", "Unknown step")

        # ── Guard: step-level max retries ────────────────────
        if step_retry_count > max_retries:
            current_step["status"] = "failed"
            warning = (
                f"Step {current_step.get('id', '?')} failed after "
                f"{max_retries + 1} attempts. Moving on."
            )
            next_idx = step_idx + 1
            if next_idx >= len(plan):
                return {
                    "phase": "done",
                    "plan": plan,
                    "error_message": warning,
                }
            return {
                "phase": "executing",
                "plan": plan,
                "current_step_index": next_idx,
                "step_retry_count": 0,
                "step_start_tool_count": len(tool_history),
                "error_message": warning,
            }

        # ── Extract agent response ────────────────────────────
        messages = state["messages"]
        agent_response = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                agent_response = m.content[:1500]
                break

        # ═══════════════════════════════════════════════════════
        # HEURISTIC PRE-CHECK — before asking the LLM
        # ═══════════════════════════════════════════════════════

        # Isolate tools called during THIS step only
        step_tools = tool_history[step_start_tool_count:]
        step_tool_count = len(step_tools)

        # Count failures within this step
        step_failures = [t for t in step_tools if not t.get("success", True)]
        step_failure_count = len(step_failures)
        all_step_tools_failed = step_tool_count > 0 and step_failure_count == step_tool_count
        some_step_tools_failed = step_failure_count > 0

        # Check if the agent's response looks like an error report
        agent_indicates_error = _text_indicates_error(agent_response)

        # ── Heuristic: ALL tools in this step failed ─────────
        if all_step_tools_failed:
            retry_suggestion = _build_failure_suggestion(step_failures)
            if step_retry_count < max_retries:
                current_step["status"] = "in_progress"
                current_step["retry_count"] = step_retry_count + 1
                return {
                    "phase": "retry",
                    "plan": plan,
                    "step_retry_count": step_retry_count + 1,
                    "error_message": (
                        f"[HEURISTIC] All {step_tool_count} tool(s) in step "
                        f"{current_step.get('id', '?')} failed. "
                        f"Retrying (attempt {step_retry_count + 2}). "
                        f"Suggestion: {retry_suggestion}"
                    ),
                }
            # Max retries exceeded → already handled by guard above
            # (this branch reached when step_retry_count == max_retries, guard catches >)

        # ── Heuristic: some tools failed AND agent reports error ──
        if some_step_tools_failed and agent_indicates_error:
            if step_retry_count < max_retries:
                current_step["status"] = "in_progress"
                current_step["retry_count"] = step_retry_count + 1
                return {
                    "phase": "retry",
                    "plan": plan,
                    "step_retry_count": step_retry_count + 1,
                    "error_message": (
                        f"[HEURISTIC] Tool failures detected + agent reports error "
                        f"in step {current_step.get('id', '?')}. "
                        f"Retrying (attempt {step_retry_count + 2})."
                    ),
                }

        # ── Build tool summaries for LLM reflection ──────────
        recent_tools = step_tools if step_tools else tool_history[-8:]
        tool_lines = []
        tool_errors_lines = []
        for t in recent_tools:
            result_preview = str(t.get("result", ""))[:120]
            success = t.get("success", True)
            tag = "OK" if success else "FAIL"
            tool_lines.append(
                f"  [{t['tool']}] {str(t.get('args', {}))[:80]}"
                f" → [{tag}] {result_preview}"
            )
            if not success:
                tool_errors_lines.append(
                    f"  [{t['tool']}] {result_preview}"
                )

        tool_summary = "\n".join(tool_lines) if tool_lines else "(no tools called)"
        tool_errors = "\n".join(tool_errors_lines) if tool_errors_lines else "(no errors)"

        # ═══════════════════════════════════════════════════════
        # LLM-BASED REFLECTION — for nuanced evaluation
        # ═══════════════════════════════════════════════════════

        try:
            reflect_response = await llm.ainvoke([
                SystemMessage(content=REFLECT_SYSTEM_PROMPT),
                HumanMessage(content=REFLECT_USER_TEMPLATE.format(
                    step_description=step_desc,
                    agent_response=agent_response,
                    tool_summary=tool_summary,
                    tool_errors=tool_errors,
                )),
            ])

            reflection_text = reflect_response.content.strip()
            reflection = _extract_json_object(reflection_text)

        except Exception:
            # FALLBACK: don't blindly assume success. Check tool results.
            if some_step_tools_failed and step_retry_count < max_retries:
                reflection = {
                    "step_done": False,
                    "success": False,
                    "error_type": "recoverable",
                    "reasoning": "Reflection LLM call failed; tool errors detected, retrying step.",
                    "should_retry": True,
                    "should_replan": False,
                    "retry_suggestion": "Fix the tool errors and try again.",
                }
            elif some_step_tools_failed:
                # No retries left but tools failed → mark failed
                reflection = {
                    "step_done": True,
                    "success": False,
                    "error_type": "fatal",
                    "reasoning": "Reflection LLM call failed; tools failed with no retries left.",
                    "should_retry": False,
                    "should_replan": False,
                    "retry_suggestion": "",
                }
            else:
                # No tools called or all succeeded → assume step done
                reflection = {
                    "step_done": True,
                    "success": True,
                    "error_type": "none",
                    "reasoning": "Reflection LLM call failed; no tool errors detected, assuming done.",
                    "should_retry": False,
                    "should_replan": False,
                    "retry_suggestion": "",
                }

        # ═══════════════════════════════════════════════════════
        # ACT on reflection result
        # ═══════════════════════════════════════════════════════

        should_retry = reflection.get("should_retry", False)
        should_replan = reflection.get("should_replan", False)
        error_type = reflection.get("error_type", "none")

        # ── Safety override: if heuristic detected failures but LLM says success, force retry
        if all_step_tools_failed and not should_retry and error_type == "none":
            if step_retry_count < max_retries:
                reflection["should_retry"] = True
                reflection["error_type"] = "recoverable"
                reflection["retry_suggestion"] = _build_failure_suggestion(step_failures)
                should_retry = True
                error_type = "recoverable"

        if should_retry and error_type == "recoverable":
            current_step["status"] = "in_progress"
            current_step["retry_count"] = step_retry_count + 1
            retry_hint = reflection.get("retry_suggestion", "")

            return {
                "phase": "retry",
                "plan": plan,
                "step_retry_count": step_retry_count + 1,
                "error_message": (
                    f"Retrying step {current_step.get('id', '?')} "
                    f"(attempt {step_retry_count + 2}): {retry_hint}"
                ),
            }

        if should_replan:
            current_step["status"] = "failed"
            return {
                "phase": "replan",
                "plan": plan,
                "error_message": reflection.get("reasoning", "Plan needs revision"),
            }

        if error_type == "fatal":
            current_step["status"] = "failed"
            next_idx = step_idx + 1
            if next_idx >= len(plan):
                return {
                    "phase": "done",
                    "plan": plan,
                    "step_retry_count": 0,
                    "error_message": reflection.get("reasoning", ""),
                }
            return {
                "phase": "executing",
                "plan": plan,
                "current_step_index": next_idx,
                "step_retry_count": 0,
                "step_start_tool_count": len(tool_history),
                "error_message": reflection.get("reasoning", ""),
            }

        # ── Default: step done successfully ──────────────────
        current_step["status"] = "done"
        next_idx = step_idx + 1

        if next_idx >= len(plan):
            return {
                "phase": "done",
                "plan": plan,
                "current_step_index": next_idx,
                "step_retry_count": 0,
            }
        else:
            return {
                "phase": "executing",
                "plan": plan,
                "current_step_index": next_idx,
                "step_retry_count": 0,
                "step_start_tool_count": len(tool_history),
            }

    return reflect_node


# ═══════════════════════════════════════════════════════════════════
# ── Replan Node ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

async def replan_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Regenerate the remaining (unfinished) steps of a plan.

    Called when reflection determines the original plan is wrong.
    Keeps completed steps, rewrites the rest.
    """
    plan = state.get("plan", [])
    step_idx = state["current_step_index"]
    error_message = state.get("error_message", "")

    # Separate completed and remaining
    completed = [s for s in plan[:step_idx] if s.get("status") == "done"]
    remaining = plan[step_idx:]

    completed_lines = []
    for s in completed:
        icon = _status_icon(s.get("status", "done"))
        completed_lines.append(f"  {icon} Step {s['id']}: {s['description']}")

    remaining_lines = []
    for s in remaining:
        remaining_lines.append(f"  ⏳ Step {s['id']}: {s['description']}")

    # Gather recent discoveries (from tool history)
    tool_history = state.get("tool_history", [])
    discoveries_lines = []
    for t in tool_history[-5:]:
        if t.get("success"):
            discoveries_lines.append(
                f"  [{t['tool']}] {str(t.get('args', {}))[:80]} → {str(t.get('result', ''))[:100]}"
            )
    if not discoveries_lines:
        discoveries_lines.append("  (no tools called yet)")

    try:
        replan_response = await llm.ainvoke([
            SystemMessage(content=REPLAN_SYSTEM_PROMPT),
            HumanMessage(content=REPLAN_USER_TEMPLATE.format(
                task=state["task"],
                completed_summary="\n".join(completed_lines) or "(none yet)",
                remaining_summary="\n".join(remaining_lines),
                error_context=error_message,
                discoveries="\n".join(discoveries_lines),
            )),
        ])

        replan_text = replan_response.content.strip()
        new_remaining = _extract_json_array(replan_text)

        if not isinstance(new_remaining, list) or len(new_remaining) == 0:
            raise ValueError("Replan produced empty or invalid result")

    except Exception:
        # Fallback: keep remaining steps but mark them for retry
        new_remaining = []
        for i, s in enumerate(remaining):
            new_remaining.append({
                "id": str(len(completed) + i + 1),
                "description": s.get("description", "Retry step"),
                "status": "pending",
                "retry_count": 0,
                "max_retries": 2,
            })

    # Reassign IDs: completed keep theirs, new remaining get sequential
    for i, step in enumerate(new_remaining):
        step["id"] = str(len(completed) + i + 1)
        step.setdefault("status", "pending")
        step.setdefault("retry_count", 0)
        step.setdefault("max_retries", 2)

    new_plan = completed + new_remaining

    return {
        "phase": "executing",
        "plan": new_plan,
        "current_step_index": len(completed),
        "step_retry_count": 0,
        "error_message": "",
    }


# ═══════════════════════════════════════════════════════════════════
# ── Finish Node ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def _should_pass_through_finish(state: AgentState) -> bool:
    """True when the task produced no tool calls — use the agent's text as-is.

    V3: Simple rule — if no tools were called, the answer is already complete.
    No intent_class or regex dependency. Works for all modes:
      - ask mode: read-only exploration answered without tools → pass through
      - agent mode: simple Q&A answered without tools → pass through
      - plan mode: unlikely to have 0 tools, but handled consistently
    """
    return len(state.get("tool_history", [])) == 0


async def finish_node(
    state: AgentState,
    llm: BaseChatModel,
    memory_manager=None,
) -> Dict[str, Any]:
    """Generate the final answer for the user.

    V3 pass-through rule: if no tools were called, the agent's response is
    already complete — use it directly without LLM re-summarization.
    This handles simple Q&A ("中国首都是哪里"), identity ("你是谁"), and
    zero-tool explorations naturally across all modes.

    When tools were used, generates a structured coding-task summary.

    V2.1: Persists turn record via MemoryManager after final_answer is generated.
    """
    plan = state.get("plan", [])
    tool_history = state.get("tool_history", [])

    # ── Last AI response ───────────────────────────────────
    messages = state["messages"]
    agent_response = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            agent_response = m.content[:2000]
            break

    # ── No-tool pass-through (V3) ───────────────────────────
    # If no tools were called, the answer is already complete.
    # Simple Q&A, identity, greetings — use the agent's text as-is.
    if _should_pass_through_finish(state):
        final_answer = _strip_status_block(agent_response) or agent_response or ""
        if not final_answer.strip():
            final_answer = f"Task completed: {state['task']}"

        # Still write memory
        if memory_manager is not None and final_answer:
            try:
                await memory_manager.record_completed_turn(
                    state, final_answer, success=True
                )
            except Exception:
                pass

        return {
            "phase": "done",
            "final_answer": final_answer,
        }

    # ── Coding task: full structured summary ────────────────
    # Plan summary
    plan_lines = _build_plan_summary_lines(plan)
    plan_summary = "\n".join(plan_lines) if plan_lines else "No plan recorded"

    # Tool summary (last 20)
    tool_lines = []
    for t in tool_history[-20:]:
        args_short = str(t.get("args", {}))[:80]
        result_short = str(t.get("result", ""))[:120]
        success = "OK" if t.get("success", True) else "FAIL"
        tool_lines.append(
            f"  [{t['tool']}] [{success}] {args_short} → {result_short}"
        )
    tool_summary = "\n".join(tool_lines) if tool_lines else "No tools called"

    # ── Step 1: Generate final_answer FIRST (B3 fix) ──────
    try:
        summary_response = await llm.ainvoke([
            SystemMessage(content=FINISH_SYSTEM_PROMPT),
            HumanMessage(content=FINISH_USER_TEMPLATE.format(
                task=state["task"],
                plan_summary=plan_summary,
                tool_summary=tool_summary,
                agent_response=agent_response,
            )),
        ])
        final_answer = summary_response.content.strip()
    except Exception:
        final_answer = (
            f"Task completed: {state['task']}\n\n"
            f"Steps:\n{plan_summary}\n\n"
            f"Actions:\n{tool_summary}"
        )

    # ── Step 2: Write memory AFTER final_answer is ready (B3 fix) ──
    if memory_manager is not None and final_answer:
        try:
            success = state.get("phase", "done") == "done" or True
            await memory_manager.record_completed_turn(
                state, final_answer, success=success
            )
        except Exception:
            pass  # Memory errors should not block agent completion

    return {
        "phase": "done",
        "final_answer": final_answer,
    }


# ═══════════════════════════════════════════════════════════════════
# ── V2 Helpers ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

_V1_MAX_MSG_COUNT = 40


def _v1_truncate(
    messages: list,
    context_message: HumanMessage,
    max_count: int = _V1_MAX_MSG_COUNT,
) -> list:
    """V1-style message truncation — keep system + recent messages.

    This is the fallback when ContextManager is unavailable.
    """
    if len(messages) <= max_count:
        return list(messages) + [context_message]

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    recent_msgs = messages[-(max_count - len(system_msgs)):]
    return system_msgs + recent_msgs + [context_message]


def _estimate_messages_tokens(messages: list) -> int:
    """Quick token estimate: char_count / 4 heuristic (works across providers)."""
    total_chars = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total_chars += len(str(content))
    return max(1, total_chars // 4)


# ═══════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def _status_icon(status: str) -> str:
    icons = {
        "pending": "⏳",
        "in_progress": "🔄",
        "done": "✅",
        "failed": "❌",
    }
    return icons.get(status, "❓")


def _build_plan_summary_lines(
    plan: list, current_step: dict | None = None
) -> list[str]:
    """Build a list of plan progress lines."""
    lines = []
    for s in plan:
        marker = "← NOW" if (current_step and s["id"] == current_step.get("id")) else ""
        icon = _status_icon(s.get("status", "pending"))
        retry_note = (
            f" (retry {s.get('retry_count', 0)})"
            if s.get("retry_count", 0) > 0
            else ""
        )
        lines.append(
            f"  {icon} Step {s['id']}: {s['description']}{retry_note} {marker}"
        )
    return lines


# Regex to strip AGENT_STATUS blocks from LLM output (kept in sync with routing.py)
_STATUS_STRIP_PATTERN = re.compile(
    r"---AGENT_STATUS---\s*\{.*?\}\s*---END_STATUS---",
    re.DOTALL,
)


def _strip_status_block(text: str) -> str:
    """Remove any AGENT_STATUS block from LLM output for clean display."""
    if not text:
        return text
    return _STATUS_STRIP_PATTERN.sub("", text).strip()


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM output that may have markdown fences or extra text."""
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence line
        lines = lines[1:]
        # Remove closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find [...] in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON array from: {text[:200]}")


def _extract_json_object(text: str) -> dict:
    """Extract a JSON object from LLM output."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON object from: {text[:200]}")


def _extract_recent_errors(state: AgentState) -> str:
    """Extract error messages from recent tool results."""
    tool_history = state.get("tool_history", [])
    error_lines = []
    for t in tool_history[-5:]:
        if not t.get("success", True):
            result = str(t.get("result", ""))[:200]
            error_lines.append(f"  [{t['tool']}] Error: {result}")
    return "\n".join(error_lines) if error_lines else "(no tool errors detected)"


def _extract_step_tool_errors(state: AgentState, step_start_count: int) -> str:
    """Extract detailed errors from tools called during the CURRENT step only.

    Args:
        state: The current AgentState.
        step_start_count: The tool_history length when this step began.

    Returns:
        Formatted string of all tool errors within this step.
    """
    tool_history = state.get("tool_history", [])
    step_tools = tool_history[step_start_count:]
    error_lines = []
    for t in step_tools:
        if not t.get("success", True):
            tool_name = t.get("tool", "?")
            args = t.get("args", {})
            result = str(t.get("result", ""))[:300]
            error_lines.append(
                f"  [{tool_name}]({_format_args_short(args)})\n"
                f"    → {result}"
            )
    if not error_lines:
        return "(no tool errors)"
    return "Tool failures in this step:\n" + "\n".join(error_lines)


def _format_args_short(args: dict) -> str:
    """Format tool arguments compactly for error display."""
    parts = []
    for k, v in args.items():
        v_str = str(v)[:60].replace("\n", "\\n")
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


# ── Error detection patterns ─────────────────────────────────────

# Phrases that indicate the agent encountered an error/unexpected result
_ERROR_INDICATOR_PATTERNS = [
    r"\b(error|fail|failed|failing|FAILED)\b",
    r"\b(cannot|can't|can\s*not)\b",
    r"\b(doesn't|does\s*not)\s+(exist|work|run|find|match)\b",
    r"\b(ModuleNotFoundError|ImportError|SyntaxError|TypeError|AttributeError|KeyError|NameError)\b",
    r"\b(traceback|stack\s*trace)\b",
    r"\b(unable|impossible)\b",
    r"\b(didn't|did\s*not)\s+(work|succeed|complete)\b",
    r"\bno\s+such\s+file\b",
    r"\bpermission\s+denied\b",
    r"\bcommand\s+not\s+found\b",
    r"\b(timeout|timed?\s*out)\b",
    r"\bI\s+(couldn't|could\s*not|was\s*unable)\b",
]


def _text_indicates_error(text: str) -> bool:
    """Check if the agent's text response indicates it encountered an error.

    Looks for error keywords, exception names, and failure descriptions.
    """
    if not text:
        return False
    text_lower = text.lower()
    for pattern in _ERROR_INDICATOR_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def _build_failure_suggestion(failed_tools: list) -> str:
    """Build a retry suggestion based on which tools failed and how.

    Args:
        failed_tools: List of tool_history entries that failed.

    Returns:
        A suggestion string for the retry context.
    """
    if not failed_tools:
        return "Investigate and fix the issue before retrying."

    tool_names = set(t.get("tool", "?") for t in failed_tools)

    suggestions = []
    if "shell_execute" in tool_names:
        suggestions.append("Check the command output for the specific error and fix it")
    if "read_file" in tool_names:
        suggestions.append("Verify the file path exists and is correct")
    if "edit_file" in tool_names:
        suggestions.append("Read the file again to get the exact text for old_string")
    if "grep_search" in tool_names:
        suggestions.append("Try a simpler or corrected regex pattern")
    if "write_file" in tool_names:
        suggestions.append("Check that the target directory exists and is writable")

    if suggestions:
        return ". ".join(suggestions) + "."
    return "Analyze the tool errors above and try a different approach."
