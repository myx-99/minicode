"""MemoryManager — unified entry point for cross-turn memory (V2.1).

Orchestrates SessionMemory (in-process) and ProjectMemory (persistent).
Called by agent/graph nodes for context injection and turn recording,
and by CLI for shared lifecycle across REPL turns.

Replaces V2's disparate LongTermMemoryKeyword + init_node search patterns.
V2.2: Upgraded to vector-based semantic search (embeddings + time decay).
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel

from memory.types import TurnRecord
from memory.session import SessionMemory
from memory.project import ProjectMemory, is_meta_query


class MemoryManager:
    """Unified memory manager for cross-turn context.

    Layers:
      1. SessionMemory — in-RAM REPL turn history (this process only)
      2. ProjectMemory — disk-persisted turns (survives restarts)

    Usage:
        mm = MemoryManager(workspace_root=Path("."), enabled=True)
        ctx = mm.build_context_for_task("Fix the bug")
        turn = await mm.record_completed_turn(state, final_answer, success=True)
    """

    def __init__(
        self,
        workspace_root: Path,
        enabled: bool = True,
        session: Optional[SessionMemory] = None,
        project: Optional[ProjectMemory] = None,
        session_turns: int = 10,
        project_recent: int = 5,
        project_search_k: int = 3,
        max_turns: int = 200,
        llm: Optional[BaseChatModel] = None,  # Reserved for future LLM-based compression
    ):
        """Initialize the memory manager.

        Args:
            workspace_root: Project root directory.
            enabled: If False, all operations are no-ops.
            session: Pre-existing SessionMemory (for mode-switch preservation).
            project: Pre-existing ProjectMemory.
            session_turns: Number of recent session turns to inject.
            project_recent: Number of recent project turns to inject.
            project_search_k: Number of vector-search results to inject.
            max_turns: Max turns persisted in project store.
            llm: Optional LLM for future compression support.
        """
        self._enabled = enabled

        if not enabled:
            self._session: Optional[SessionMemory] = None
            self._project: Optional[ProjectMemory] = None
            self._session_id = ""
            self._session_turns = session_turns
            self._project_recent = project_recent
            self._project_search_k = project_search_k
            return

        self._session = session or SessionMemory()
        self._project = project or ProjectMemory(workspace_root, max_turns=max_turns)
        self._session_id = self._session.session_id
        self._session_turns = session_turns
        self._project_recent = project_recent
        self._project_search_k = project_search_k
        self._llm = llm

    # ── Properties ────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session(self) -> Optional[SessionMemory]:
        return self._session

    @property
    def project(self) -> Optional[ProjectMemory]:
        return self._project

    # ── Context building ──────────────────────────────────────

    def build_context_for_task(self, task: str) -> str:
        """Build a combined memory context string for injection into the system prompt.

        Merges:
          1. Recent session turns (in-process REPL history)
          2. Recent project turns (from disk, across sessions)
          3. Vector-search results (if not a meta query)

        For meta queries ("what did I just do"), includes more session history
        and skips vector search (which wouldn't help).

        Args:
            task: The current user task.

        Returns:
            Markdown string to inject, or "" if nothing is available.
        """
        if not self._enabled or self._session is None or self._project is None:
            return ""

        meta = is_meta_query(task)

        # ── Session block ─────────────────────────────────
        session_n = self._session_turns * 2 if meta else self._session_turns
        session_turns = self._session.get_recent_turns(session_n)
        session_block = self._session.format_for_prompt(session_turns)

        # ── Project block ─────────────────────────────────
        project_recent = self._project.load_recent(self._project_recent)

        # Vector search (skip for meta queries — they need recency, not relevance)
        if meta:
            project_search = []
        else:
            project_search = self._project.search(task, k=self._project_search_k)

        # Merge + deduplicate by turn.id (session wins)
        seen_ids = {t.id for t in session_turns}
        project_turns = []
        for t in project_recent + project_search:
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                project_turns.append(t)

        project_block = ""
        if project_turns:
            lines = ["## Previous Session History (from disk)", ""]
            for i, t in enumerate(project_turns, 1):
                status_icon = "✅" if t.success else "❌"
                lines.append(
                    f"### Previous Turn {i} ({t.created_at[:16]}, {t.mode}, {status_icon})"
                )
                lines.append(f"**Task:** {t.user_task}")
                lines.append(f"**Result:** {t.final_answer[:200]}")
                if t.files_changed:
                    lines.append(f"**Files:** {', '.join(t.files_changed[:5])}")
                lines.append("")
            project_block = "\n".join(lines)

        # ── Combine ────────────────────────────────────────
        parts = []
        if session_block:
            parts.append(session_block)
        if project_block:
            parts.append(project_block)

        return "\n".join(parts)

    # ── Turn recording ─────────────────────────────────────────

    async def record_completed_turn(
        self,
        state: dict,
        final_answer: str,
        *,
        success: bool,
    ) -> Optional[TurnRecord]:
        """Record a completed task turn in both session and project memory.

        MUST be called AFTER final_answer is generated (B3 fix).

        Args:
            state: The AgentState at completion time.
            final_answer: The finish_node generated summary.
            success: Whether the task completed successfully.

        Returns:
            The created TurnRecord, or None if memory is disabled.
        """
        if not self._enabled or self._session is None or self._project is None:
            return None

        # ── Extract metadata from state ─────────────────
        task = state.get("task", "")
        mode = state.get("mode", "plan")
        tool_history = state.get("tool_history", [])

        # Files changed: write_file + edit_file successful paths
        files_changed = []
        tools_used = set()
        for t in tool_history:
            tool_name = t.get("tool", "")
            tools_used.add(tool_name)
            if tool_name in ("write_file", "edit_file") and t.get("success", True):
                args = t.get("args", {})
                path = args.get("file_path", "")
                if path and path not in files_changed:
                    files_changed.append(path)

        # ── Build TurnRecord ────────────────────────────
        turn_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        turn = TurnRecord(
            id=turn_id,
            user_task=task,
            final_answer=final_answer,
            success=success,
            mode=mode,
            files_changed=files_changed,
            tools_used=sorted(tools_used),
            created_at=datetime.now().isoformat(),
            session_id=self._session_id,
        )

        # ── Persist ──────────────────────────────────────
        self._session.add_turn(turn)
        self._project.add_turn(turn)

        return turn

    # ── Session management ──────────────────────────────────────

    def get_session_turns(self) -> List[TurnRecord]:
        """Return all turns from the current session."""
        if self._session is None:
            return []
        return self._session.get_all_turns()

    def get_project_turn_count(self) -> int:
        """Return number of turns persisted on disk."""
        if self._project is None:
            return 0
        return self._project.count

    async def clear_all(self) -> None:
        """Clear both session and project memory."""
        if self._session is not None:
            self._session.clear()
        if self._project is not None:
            self._project.clear()
