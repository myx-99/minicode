"""TurnStore — JSONL persistence backend for turn-based memory (V2.1).

Replaces V2 MemoryStore (entries.jsonl) with turn-oriented persistence.
Writes to <workspace>/.agent/memory/turns.jsonl.

Old entries.jsonl is silently ignored — no migration.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Iterator

from memory.types import TurnRecord


class TurnStore:
    """JSONL-backed persistent store for TurnRecord objects.

    Writes to <workspace>/.agent/memory/turns.jsonl.
    Supports append, load-all, load-recent, delete, clear.

    Usage:
        store = TurnStore(workspace_root)
        store.append(turn)
        turns = store.load_all()
        store.clear()
    """

    def __init__(self, workspace_root: Path, max_turns: int = 200):
        """Initialize the turn store.

        Args:
            workspace_root: Project root directory.
            max_turns: Soft cap — loaded entries beyond this are truncated (LRU-style).
        """
        self._workspace = workspace_root
        self._max_turns = max_turns
        self._mem_dir = workspace_root / ".agent" / "memory"
        self._mem_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._mem_dir / "turns.jsonl"

    # ── Core operations ──────────────────────────────────────

    def append(self, turn: TurnRecord) -> None:
        """Append a single TurnRecord to the JSONL file."""
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            json.dump(turn.to_dict(), f, ensure_ascii=False)
            f.write("\n")

    def load_all(self) -> List[TurnRecord]:
        """Load all TurnRecords from disk."""
        entries = list(self._iter_lines())
        # Enforce max_turns (LRU: keep most recent)
        if len(entries) > self._max_turns:
            entries = entries[-self._max_turns:]
            self._rewrite(entries)
        return [TurnRecord.from_dict(d) for d in entries]

    def load_recent(self, n: int = 20) -> List[TurnRecord]:
        """Load the N most recent turns."""
        all_turns = self.load_all()
        return all_turns[-n:]

    def get(self, turn_id: str) -> Optional[TurnRecord]:
        """Get a single turn by ID."""
        for data in self._iter_lines():
            if data.get("id") == turn_id:
                return TurnRecord.from_dict(data)
        return None

    def delete(self, turn_id: str) -> bool:
        """Delete a single turn by ID. Returns True if found and deleted."""
        entries = list(self._iter_lines())
        filtered = [e for e in entries if e.get("id") != turn_id]
        if len(filtered) == len(entries):
            return False
        self._rewrite(filtered)
        return True

    def clear(self) -> None:
        """Remove all turns from disk."""
        if self._jsonl_path.exists():
            self._jsonl_path.unlink()

    def count(self) -> int:
        """Return the number of turns on disk."""
        return sum(1 for _ in self._iter_lines())

    # ── Helpers ──────────────────────────────────────────────

    def _iter_lines(self) -> Iterator[dict]:
        """Yield each JSONL line as a dict, skipping malformed lines."""
        if not self._jsonl_path.exists():
            return
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupted lines
                    continue

    def _rewrite(self, entries: List[dict]) -> None:
        """Rewrite the entire JSONL file atomically."""
        tmp_path = self._jsonl_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        os.replace(tmp_path, self._jsonl_path)
