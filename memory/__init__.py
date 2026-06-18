"""Memory subsystem — cross-turn context and persistence (V2.2).

Layers:
  - SessionMemory: in-process REPL turn history
  - ProjectMemory: disk-persisted turns (turns.jsonl)
  - MemoryManager: unified entry point for both
  - Vector search: powered by Mini Vector DB's EmbeddingClient
    + cosine similarity + exponential time decay

Preserves context_manager.py (single-task context window management) unchanged.
"""

from memory.types import TurnRecord, SessionState
from memory.store import TurnStore
from memory.session import SessionMemory
from memory.project import ProjectMemory, is_meta_query, time_decay_score
from memory.manager import MemoryManager

__all__ = [
    "TurnRecord",
    "SessionState",
    "TurnStore",
    "SessionMemory",
    "ProjectMemory",
    "is_meta_query",
    "time_decay_score",
    "MemoryManager",
]
