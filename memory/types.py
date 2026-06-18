"""Memory types — data models for the cross-turn memory subsystem (V2.1).

Replaces V2 MemoryEntry + category fragmentation with TurnRecord as the
unit of memory — a complete record of one task execution.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TurnRecord:
    """A complete record of one task execution — the unit of memory.

    Created by MemoryManager.record_completed_turn() after finish_node
    generates the final_answer.
    """
    id: str                          # "turn_20260603_abc123"
    user_task: str                   # User's original input
    final_answer: str                # finish_node generated summary (required, non-empty)
    success: bool                    # phase == "done"
    mode: str                        # "plan" | "react"
    files_changed: List[str] = field(default_factory=list)  # write/edit paths from tool_history
    tools_used: List[str] = field(default_factory=list)      # Deduplicated tool names
    created_at: str = ""             # ISO timestamp
    session_id: str = ""             # REPL session UUID (same across all turns in one python main.py)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSONL storage."""
        return {
            "id": self.id,
            "user_task": self.user_task,
            "final_answer": self.final_answer,
            "success": self.success,
            "mode": self.mode,
            "files_changed": self.files_changed,
            "tools_used": self.tools_used,
            "created_at": self.created_at,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TurnRecord":
        """Deserialize from a plain dict."""
        return cls(
            id=data.get("id", ""),
            user_task=data.get("user_task", ""),
            final_answer=data.get("final_answer", ""),
            success=data.get("success", False),
            mode=data.get("mode", "plan"),
            files_changed=data.get("files_changed", []),
            tools_used=data.get("tools_used", []),
            created_at=data.get("created_at", ""),
            session_id=data.get("session_id", ""),
        )


@dataclass
class SessionState:
    """Current REPL session state (in-memory, optionally persisted)."""
    session_id: str
    turns: List[TurnRecord] = field(default_factory=list)  # Ordered, newest last
    started_at: str = ""
