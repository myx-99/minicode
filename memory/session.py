"""SessionMemory — in-process turn history for the current REPL session (V2.1).

Manages the ordered list of TurnRecords accumulated during one python main.py
invocation.  This is the "short-term" cross-turn layer — it persists only in
memory and is NOT written to disk (ProjectMemory handles that).
"""

from typing import List, Optional
from datetime import datetime

from memory.types import TurnRecord


class SessionMemory:
    """Tracks all task turns within the current REPL process.

    Lives entirely in RAM.  Each turn is also persisted to ProjectMemory
    (turns.jsonl) by MemoryManager.record_completed_turn().

    Usage:
        session = SessionMemory()
        session.add_turn(turn)
        recent = session.get_recent_turns(5)
        formatted = session.format_for_prompt(recent)
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize an empty session.

        Args:
            session_id: Unique ID for this REPL session. Auto-generated if None.
        """
        import uuid
        self._session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._turns: List[TurnRecord] = []
        self._started_at = datetime.now().isoformat()

    # ── Properties ────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    # ── Turn management ───────────────────────────────────────

    def add_turn(self, turn: TurnRecord) -> None:
        """Append a completed turn to session history."""
        turn.session_id = self._session_id
        self._turns.append(turn)

    def get_recent_turns(self, n: int = 10) -> List[TurnRecord]:
        """Return the most recent N turns (newest last)."""
        return self._turns[-n:] if n > 0 else []

    def get_all_turns(self) -> List[TurnRecord]:
        """Return all turns in this session."""
        return list(self._turns)

    def clear(self) -> None:
        """Clear all in-memory turns (does NOT touch project storage)."""
        self._turns.clear()

    # ── Prompt formatting ─────────────────────────────────────

    def format_for_prompt(self, turns: Optional[List[TurnRecord]] = None) -> str:
        """Format a list of turns as a markdown block for LLM injection.

        Args:
            turns: Specific turns to format. Defaults to all session turns.

        Returns:
            Markdown string like:
            ## Recent Session History (newest last)

            ### Turn 1 (2026-06-03 15:23, react, ✅)
            **User asked:** ...
            **Result:** ...
            **Files changed:** ...
        """
        items = turns if turns is not None else self._turns
        if not items:
            return ""

        lines = ["## Recent Session History (newest last)", ""]
        for t in items:
            status_icon = "✅" if t.success else "❌"
            lines.append(
                f"### Turn {items.index(t) + 1} ({t.created_at[:16]}, {t.mode}, {status_icon})"
            )
            lines.append(f"**User asked:** {t.user_task}")
            lines.append(f"**Result:** {t.final_answer[:300]}")
            if t.files_changed:
                lines.append(f"**Files changed:** {', '.join(t.files_changed[:10])}")
            lines.append("")
        return "\n".join(lines)
