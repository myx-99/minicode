"""V3 routing functions — signal parsing and three-mode conditional edge routing.

Extracted from builder.py for testability and separation of concerns.
Handles the AGENT_STATUS protocol and ask/agent/plan routing decisions.

AGENT_STATUS Protocol (V2, preserved in V3):
  LLM appends this block to non-tool text responses:
  ---AGENT_STATUS---
  {"action": "continue" | "step_done" | "task_complete" | "replan", "reason": "..."}
  ---END_STATUS---
"""

import json
import re
from typing import Literal, Optional
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from agent.state import AgentState


# ═══════════════════════════════════════════════════════════════════
# ── Signal Types ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentStatus:
    """Parsed AGENT_STATUS signal from LLM response."""
    action: Literal["continue", "step_done", "task_complete", "replan"]
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════
# ── Signal Parsing ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

# Regex to find the AGENT_STATUS block (structuring-based)
_STATUS_PATTERN = re.compile(
    r"---AGENT_STATUS---\s*"
    r"(\{.*?\})"
    r"\s*---END_STATUS---",
    re.DOTALL,
)


def parse_agent_status(content: str) -> Optional[AgentStatus]:
    """Parse AGENT_STATUS block from LLM text response.

    Returns None if no status block is found — callers fall back to
    mode-specific default behavior.

    Args:
        content: The AIMessage content string.

    Returns:
        AgentStatus if parsed successfully, None otherwise.
    """
    if not content:
        return None

    match = _STATUS_PATTERN.search(content)
    if not match:
        return None

    json_str = match.group(1)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    action = data.get("action")
    if action not in ("continue", "step_done", "task_complete", "replan"):
        return None

    return AgentStatus(
        action=action,
        reason=data.get("reason", ""),
    )


def _extract_text_without_status(content: str) -> str:
    """Strip the AGENT_STATUS block from LLM content for clean display."""
    if not content:
        return content
    return _STATUS_PATTERN.sub("", content).strip()


# ═══════════════════════════════════════════════════════════════════
# ── V3 Routing Functions ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def route_after_execute(
    state: AgentState,
) -> Literal["tools", "reflect", "replan", "finish", "execute"]:
    """Decide where to go after the execute node — V3 three-mode routing.

    Priority (highest first):
      1. LLM has tool_calls → "tools" (execute tool calls in ReAct loop)
      2. task_complete signal → "finish" (LLM declares task done)
      3. replan signal → "replan" (LLM asks for plan revision)
      4. plan mode, no signal → "reflect" (step evaluation)
      5. ask/agent mode, no signal → "execute" (free loop, model decides)
      6. max_iterations reached → "finish" (safety guard)

    V3: No intent_class routing. The model decides tool usage via the mode
    and tool registry — no regex pre-classification.

    Returns:
        One of "tools", "reflect", "replan", "finish", "execute".
    """
    messages = state.get("messages", [])
    if not messages:
        return "reflect"

    last_message = messages[-1]
    mode = state.get("mode", "agent")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 30)

    # ── 1. LLM wants to call tools ──────────────────────────
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # ── 2-3. Parse AGENT_STATUS signal ──────────────────────
    content = ""
    if isinstance(last_message, AIMessage):
        content = last_message.content or ""

    status = parse_agent_status(content)

    if status is not None:
        if status.action == "task_complete":
            return "finish"
        if status.action == "replan":
            return "replan"
        # step_done / continue: fall through to mode-specific routing

    # ── 4-6. Mode-specific default routing ──────────────────
    if mode == "plan":
        # Plan mode: go through reflect for step evaluation
        return "reflect"

    # ask / agent mode: keep looping (free ReAct — model decides when done)
    if iteration >= max_iterations:
        return "finish"
    return "execute"


def route_after_reflect(
    state: AgentState,
) -> Literal["execute", "replan", "finish"]:
    """Decide where to go after the reflect node (same as V1).

    - "executing" → next step (normal advance)
    - "retry" → back to execute for the SAME step (error recovery)
    - "replan" → replan_node to regenerate remaining steps
    - "done" / "error" → finish
    """
    phase = state.get("phase", "done")

    if phase == "executing":
        return "execute"
    elif phase == "retry":
        return "execute"
    elif phase == "replan":
        return "replan"
    else:
        return "finish"


def route_after_audit(
    state: AgentState,
) -> Literal["execute", "finish"]:
    """Decide where to go after the audit_plan node.

    - "done" → finish (all plan steps rejected by Auditor — direct answer)
    - anything else → execute (normal plan execution)
    """
    phase = state.get("phase", "executing")
    if phase == "done":
        return "finish"
    return "execute"


# ═══════════════════════════════════════════════════════════════════
# ── V3 Mode Normalization ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

def normalize_mode(mode: str) -> str:
    """Normalize mode strings for V3.

    - "ask" / "agent" / "plan" → pass through
    - "react" → "agent" (deprecated alias, caller should warn)
    - unknown → "agent" (safe default)

    Args:
        mode: Raw mode string from CLI or settings.

    Returns:
        Normalized mode string ("ask", "agent", or "plan").
    """
    if mode in ("ask", "agent", "plan"):
        return mode
    if mode == "react":
        return "agent"  # deprecated alias
    return "agent"  # safe default
