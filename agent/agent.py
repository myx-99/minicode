"""ClaudeCodeMini — the main Agent class.

This is the top-level entry point. It wires together:
  - LLM (via config/llm.py)
  - ToolRegistry (mode-aware: 3 tools for ask, 6 for agent/plan)
  - Workspace (path safety)
  - LangGraph StateGraph

V3: Three execution modes — ask (read-only), agent (full, default), plan (opt-in).
    "react" is accepted as a deprecated alias for "agent".

V2.1: Replaces V2 long_term_memory with MemoryManager for cross-turn memory.
      Preserves ContextManager for single-task context window management.

Usage:
    agent = ClaudeCodeMini(workspace_path=".", mode="agent")
    result = await agent.run("Fix the import error in main.py")
    print(result)
"""

from pathlib import Path
from typing import Optional, Dict, Any

from langchain_core.language_models import BaseChatModel

from config.llm import create_llm
from config.settings import settings
from runtime.workspace import Workspace
from tools.registry import ToolRegistry
from graph.builder import build_graph
from graph.routing import normalize_mode
from agent.state import AgentState


class ClaudeCodeMini:
    """The Claude Code Mini Coding Agent.

    This class encapsulates the entire agent lifecycle:
      1. Initialize workspace + tools + LLM + graph
      2. Accept user tasks
      3. Run the agent loop to completion
      4. Return structured results

    V3 features:
      - Three modes: "ask" (read-only), "agent" (full, default), "plan" (opt-in)
      - Model-driven tool usage — no regex pre-classification
      - Mode-aware tool registry (ask=3 tools, agent/plan=6 tools)
      - Context management via ContextManager
      - Cross-turn memory via MemoryManager (Session + Project)

    Example:
        agent = ClaudeCodeMini(workspace_path="/path/to/project", mode="agent")
        result = await agent.run("Add a docstring to every function")
        print(result["final_answer"])
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
        mode: str = "agent",
        max_iterations: int = 30,
        max_retries_per_step: int = 2,
        verbose: bool = False,
        memory_enabled: bool = True,
        memory_manager=None,  # MemoryManager | None — inject shared instance from CLI
        context_max_tokens: int = 120_000,
    ):
        """Initialize the agent.

        Args:
            workspace_path: Project root directory. Defaults to settings.WORKSPACE_PATH.
            llm: Optional pre-configured LLM. If None, creates from settings.
            mode: "ask" (read-only), "agent" (full tools, default), or "plan" (Plan-and-Execute).
                  "react" is accepted as a deprecated alias for "agent".
            max_iterations: Maximum ReAct loop iterations per task.
            max_retries_per_step: Max retries per step before giving up.
            verbose: If True, enable debug logging.
            memory_enabled: Enable cross-turn memory (V2.1).
            memory_manager: Shared MemoryManager instance (CLI injects to share across turns).
            context_max_tokens: Max token budget for context window (V2).
        """
        # ── Mode normalization (react → agent deprecation) ──
        self._mode: str = normalize_mode(mode)

        # ── Workspace ───────────────────────────────────────
        ws_path = workspace_path or settings.workspace_path
        self._workspace = Workspace(ws_path)

        # ── LLM ─────────────────────────────────────────────
        self._llm = llm if llm is not None else create_llm()

        # ── Tools (V3: mode-aware) ──────────────────────────
        self._tool_registry = ToolRegistry.create_for_mode(
            self._workspace, self._mode
        )

        # ── V2.1: Memory & Context ──────────────────────────
        self._memory_enabled = memory_enabled
        self._context_max_tokens = context_max_tokens

        # V2: ContextManager for single-task context window (unchanged)
        from memory.context_manager import ContextManager
        self._context_manager = ContextManager(
            llm=self._llm,
            max_tokens=context_max_tokens,
            reserve_tokens=8_000,
            keep_recent_messages=20,
        )

        # V2.1: MemoryManager for cross-turn memory
        if memory_enabled and memory_manager is not None:
            # CLI injects shared instance — use it directly
            self._memory_manager = memory_manager
        elif memory_enabled:
            # Standalone mode (single task via python main.py "task")
            # Create a fresh MemoryManager for this run
            from memory.manager import MemoryManager
            self._memory_manager = MemoryManager(
                workspace_root=self._workspace.root,
                enabled=True,
                session_turns=getattr(settings, "memory_session_turns", 10),
                project_recent=getattr(settings, "memory_project_recent", 5),
                project_search_k=getattr(settings, "memory_search_top_k", 3),
                max_turns=getattr(settings, "memory_max_turns", 200),
            )
        else:
            self._memory_manager = None

        # ── Graph (V3: three-mode) ──────────────────────────
        self._graph = build_graph(
            self._llm,
            self._tool_registry,
            mode=self._mode,
            context_manager=self._context_manager,
            memory_manager=self._memory_manager,
        )

        # ── Config ──────────────────────────────────────────
        self._max_iterations = max_iterations
        self._max_retries_per_step = max_retries_per_step
        self._verbose = verbose

    # ── Public API ────────────────────────────────────────────────

    async def run(self, task: str) -> Dict[str, Any]:
        """Execute a coding task end-to-end.

        This is the main entry point. It:
          1. Creates the initial AgentState with the task
          2. Runs the LangGraph until completion
          3. Extracts and returns the final result

        Args:
            task: Natural-language task description.

        Returns:
            Dict with keys:
              - success: bool — whether the task completed
              - final_answer: str — the agent's summary
              - plan: list — the executed plan steps
              - tool_history: list — all tool calls made
              - phase: str — final phase ("done" or "error")
              - error_message: str — error details if any
              - iteration: int — total iterations used

        Raises:
            ValueError: If task is empty.
        """
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        initial_state = self._build_initial_state(task)

        # ── Run the graph ────────────────────────────────────
        if self._verbose:
            print(f"\n{'='*60}")
            print(f"Agent starting task: {task}")
            print(f"Mode: {self._mode}")
            print(f"Workspace: {self._workspace.root}")
            print(f"Tools: {self._tool_registry.tool_names}")
            print(f"Max iterations: {self._max_iterations}")
            print(f"{'='*60}\n")

        try:
            final_state = await self._graph.ainvoke(
                initial_state,
                config={"recursion_limit": self._max_iterations + 20},
            )
        except Exception as e:
            return {
                "success": False,
                "final_answer": f"Agent failed with error: {e}",
                "plan": initial_state.get("plan", []),
                "tool_history": initial_state.get("tool_history", []),
                "phase": "error",
                "error_message": str(e),
                "iteration": initial_state.get("iteration", 0),
            }

        return self._result_from_state(final_state)

    @staticmethod
    def _result_from_state(final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build the public result dict from a graph state snapshot."""
        phase = final_state.get("phase", "unknown")
        return {
            "success": phase == "done",
            "final_answer": final_state.get("final_answer", ""),
            "plan": final_state.get("plan", []),
            "tool_history": final_state.get("tool_history", []),
            "phase": phase,
            "error_message": final_state.get("error_message", ""),
            "iteration": final_state.get("iteration", 0),
        }

    async def stream(self, task: str):
        """Run a task with streaming output.

        Yields state updates as the graph progresses through nodes.
        Useful for building interactive UIs.

        Args:
            task: Natural-language task description.

        Yields:
            Dict[str, Any]: State snapshot at each graph step.
        """
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        initial_state = self._build_initial_state(task)

        async for event in self._graph.astream(
            initial_state,
            config={"recursion_limit": self._max_iterations + 20},
            stream_mode="values",
        ):
            yield event

    # ── Initial state builder ──────────────────────────────────────

    def _build_initial_state(self, task: str) -> AgentState:
        """Build the initial AgentState for a new task.

        Injects session_context from MemoryManager for cross-turn recall.
        """
        # Build cross-turn memory context
        session_context = ""
        session_id = ""
        turn_index = 0

        if self._memory_manager is not None:
            session_context = self._memory_manager.build_context_for_task(task)
            session_id = self._memory_manager.session_id
            turn_index = len(self._memory_manager.get_session_turns())

        initial_state: AgentState = {
            "task": task.strip(),
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": self._max_iterations,
            "step_retry_count": 0,
            "max_retries_per_step": self._max_retries_per_step,
            "step_start_tool_count": 0,
            "error_message": "",
            "final_answer": "",
            # V2 fields
            "mode": self._mode,
            "context_summary": "",
            "messages_token_estimate": 0,
            # V2.1 cross-turn memory fields
            "session_context": session_context,
            "session_id": session_id,
            "turn_index": turn_index,
        }
        return initial_state

    # ── Properties ────────────────────────────────────────────────

    @property
    def workspace(self) -> Workspace:
        """Return the workspace instance."""
        return self._workspace

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the tool registry (for inspection)."""
        return self._tool_registry

    @property
    def graph(self):
        """Return the compiled LangGraph (for inspection/debugging)."""
        return self._graph

    @property
    def mode(self) -> str:
        """Return the current execution mode."""
        return self._mode

    @property
    def memory_manager(self):
        """Return the MemoryManager (for CLI inspection)."""
        return self._memory_manager

    def __repr__(self) -> str:
        return (
            f"ClaudeCodeMini(\n"
            f"  workspace={self._workspace.root},\n"
            f"  mode={self._mode},\n"
            f"  tools={self._tool_registry.tool_names},\n"
            f"  max_iterations={self._max_iterations}\n"
            f")"
        )
