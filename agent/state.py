"""AgentState — the global state that flows through all LangGraph nodes.

V2.1 extensions: mode, session_context, context_summary, messages_token_estimate.
"""

from typing import TypedDict, List, Annotated, Literal, Optional, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """The complete state of the Coding Agent.

    This TypedDict flows through every node in the LangGraph StateGraph.
    Each node returns a partial dict with the fields it updates.
    LangGraph's add_messages reducer handles message accumulation.

    V2 additions (all optional for backward compatibility with V1 tests):
      - mode: "plan" or "react" execution mode
      - session_context: cross-turn session/project memory from MemoryManager
      - context_summary: rolling summary of compressed older messages
      - messages_token_estimate: approximate token count
    """

    # ── Task ─────────────────────────────────────────────
    task: str
    """The user's original task description."""

    # ── Conversation History ─────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    """Full conversation: user messages + AI responses + tool calls + tool results.
    Managed automatically by LangGraph's add_messages reducer."""

    # ── Planning ─────────────────────────────────────────
    plan: List[dict]
    """Structured task plan: [{"id": "1", "description": "...", "status": "pending"}, ...]"""

    current_step_index: int
    """Index into plan[] — which step the agent is currently executing."""

    # ── Execution Tracking ───────────────────────────────
    tool_history: List[dict]
    """Record of all tool calls: [{"tool": "read_file", "args": {...}, "result": "..."}, ...]"""

    step_start_tool_count: int
    """Length of tool_history when the current step began execution.
    Allows reflect_node to isolate tools called during THIS step.
    Set by execute_node when starting a fresh (non-retry) step."""

    # ── Control ──────────────────────────────────────────
    phase: str
    """Current phase: "init" | "planning" | "executing" | "retry" | "replan" | "done" | "error" """

    iteration: int
    """Total number of execute-node invocations (safety counter against infinite loops)."""

    max_iterations: int
    """Hard cap on iterations. Default: 30."""

    step_retry_count: int
    """Current retry count for the active step. Reset to 0 on new step."""

    max_retries_per_step: int
    """Max retries per single step before giving up. Default: 2."""

    # ── V2: Execution Mode ───────────────────────────────
    mode: str
    """Execution mode: "ask" (read-only), "agent" (full tools, default), or "plan" (Plan-and-Execute)."""

    # ── V2.2: Intent Classification (BUG-001, DEPRECATED in V3) ──
    intent_class: str
    """DEPRECATED in V3. Kept for test compatibility only. No longer drives graph routing.
    In V3, the model decides tool usage via mode + tool registry, not regex pre-classification."""

    # ── V3: Plan mode two-phase tracking ──────────────────
    plan_phase: str
    """For plan mode only: "planning" (single-step, read-only context) or "executing" (multi-step, full tools).
    Used to toggle tool registry between plan generation and plan execution phases."""

    # ── V2.1: Cross-Turn Memory ──────────────────────────
    session_context: str
    """Cross-turn memory context injected at init (from MemoryManager). Replaces V2 memory_context."""

    session_id: str
    """Current REPL session UUID (same across all turns in one python main.py)."""

    turn_index: int
    """0-based index of the current turn within the session."""

    context_summary: str
    """Rolling summary of compressed older messages (populated by ContextManager)."""

    messages_token_estimate: int
    """Approximate token count of current messages (for budget tracking)."""

    # ── Output ───────────────────────────────────────────
    error_message: str
    """The most recent error message, if any. Injected as feedback on retry."""

    final_answer: str
    """The final summary returned to the user when the task is complete."""
