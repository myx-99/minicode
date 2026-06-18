"""Unit tests for V2.1 cross-turn memory subsystem.

Tests verify:
  - TurnRecord serialization/deserialization
  - TurnStore CRUD + JSONL persistence (turns.jsonl)
  - SessionMemory add_turn, format_for_prompt
  - ProjectMemory search, meta query detection
  - MemoryManager build_context_for_task, record_completed_turn
  - memory_enabled=False prevents read/write

Run with:  pytest tests/test_memory.py -v
"""

import json
import pytest
from pathlib import Path

from memory.types import TurnRecord
from memory.project import is_meta_query


# ═══════════════════════════════════════════════════════════════════
# Test: TurnRecord
# ═══════════════════════════════════════════════════════════════════

class TestTurnRecord:
    """Verify TurnRecord model."""

    def test_to_dict_roundtrip(self):
        turn = TurnRecord(
            id="turn_001",
            user_task="Fix import error in main.py",
            final_answer="Fixed the import by adding requests to requirements.txt.",
            success=True,
            mode="react",
            files_changed=["requirements.txt"],
            tools_used=["read_file", "edit_file"],
            created_at="2026-06-03T10:00:00",
            session_id="session_abc123",
        )
        d = turn.to_dict()
        restored = TurnRecord.from_dict(d)
        assert restored.id == turn.id
        assert restored.user_task == turn.user_task
        assert restored.final_answer == turn.final_answer
        assert restored.success is True
        assert restored.mode == "react"
        assert restored.files_changed == ["requirements.txt"]
        assert restored.tools_used == ["read_file", "edit_file"]

    def test_defaults(self):
        turn = TurnRecord(id="t", user_task="u", final_answer="f", success=False, mode="plan")
        assert turn.files_changed == []
        assert turn.tools_used == []
        assert turn.created_at == ""
        assert turn.session_id == ""


# ═══════════════════════════════════════════════════════════════════
# Test: TurnStore
# ═══════════════════════════════════════════════════════════════════

class TestTurnStore:
    """Verify TurnStore persistence (turns.jsonl)."""

    @pytest.fixture
    def store(self, tmp_path):
        from memory.store import TurnStore
        return TurnStore(workspace_root=tmp_path, max_turns=100)

    def _make_turn(self, id, user_task="Test task", final_answer="Done"):
        return TurnRecord(
            id=id, user_task=user_task, final_answer=final_answer,
            success=True, mode="plan",
        )

    def test_append_and_load(self, store):
        turn = self._make_turn("turn_1", "Task A")
        store.append(turn)
        assert store.count() == 1
        all_turns = store.load_all()
        assert len(all_turns) == 1
        assert all_turns[0].user_task == "Task A"

    def test_jsonl_persistence(self, store):
        """Turns survive store reload (disk persistence)."""
        turn = self._make_turn("turn_2", "Persisted task")
        store.append(turn)

        from memory.store import TurnStore
        store2 = TurnStore(workspace_root=store._workspace, max_turns=100)
        assert store2.count() == 1
        assert store2.load_all()[0].user_task == "Persisted task"

    def test_delete_turn(self, store):
        turn = self._make_turn("turn_3", "To delete")
        store.append(turn)
        assert store.count() == 1
        assert store.delete("turn_3") is True
        assert store.count() == 0
        assert store.delete("nonexistent") is False

    def test_clear(self, store):
        store.append(self._make_turn("t1", "a"))
        store.append(self._make_turn("t2", "b"))
        store.clear()
        assert store.count() == 0

    def test_load_recent(self, store):
        for i in range(5):
            store.append(self._make_turn(f"turn_{i}", f"Task {i}"))
        recent = store.load_recent(3)
        assert len(recent) == 3
        assert recent[0].user_task == "Task 2"
        assert recent[2].user_task == "Task 4"

    def test_max_turns_enforcement(self, tmp_path):
        """Oldest turns evicted when exceeding max_turns."""
        from memory.store import TurnStore
        store = TurnStore(workspace_root=tmp_path, max_turns=3)

        for i in range(5):
            store.append(TurnRecord(
                id=f"turn_{i}", user_task=f"Task {i}",
                final_answer=f"Result {i}", success=True, mode="plan",
                created_at=f"2026-06-03T10:00:{i:02d}",
            ))

        entries = store.load_all()
        assert len(entries) <= 3
        contents = {e.user_task for e in entries}
        assert "Task 0" not in contents
        assert "Task 1" not in contents

    def test_get_turn(self, store):
        turn = self._make_turn("turn_g", "Get me")
        store.append(turn)
        found = store.get("turn_g")
        assert found is not None
        assert found.user_task == "Get me"
        assert store.get("nonexistent") is None

    def test_empty_store(self, store):
        assert store.count() == 0
        assert store.load_all() == []


# ═══════════════════════════════════════════════════════════════════
# Test: SessionMemory
# ═══════════════════════════════════════════════════════════════════

class TestSessionMemory:
    """Verify SessionMemory in-process turn tracking."""

    @pytest.fixture
    def session(self):
        from memory.session import SessionMemory
        return SessionMemory(session_id="test_session")

    def _make_turn(self, id, user_task="task", final_answer="result"):
        return TurnRecord(
            id=id, user_task=user_task, final_answer=final_answer,
            success=True, mode="react",
        )

    def test_add_and_get_turns(self, session):
        session.add_turn(self._make_turn("t1", "Task 1", "Result 1"))
        session.add_turn(self._make_turn("t2", "Task 2", "Result 2"))
        assert session.turn_count == 2
        turns = session.get_all_turns()
        assert len(turns) == 2
        assert turns[0].user_task == "Task 1"
        assert turns[1].user_task == "Task 2"

    def test_get_recent_turns(self, session):
        for i in range(10):
            session.add_turn(self._make_turn(f"t{i}", f"Task {i}"))
        recent = session.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[0].user_task == "Task 7"
        assert recent[-1].user_task == "Task 9"

    def test_clear(self, session):
        session.add_turn(self._make_turn("t1", "a"))
        session.clear()
        assert session.turn_count == 0

    def test_format_for_prompt(self, session):
        turn = TurnRecord(
            id="turn_x", user_task="新建文件夹 matrix_demo",
            final_answer="Created matrix_multiplication/matrix_mul.py",
            success=True, mode="react",
            files_changed=["matrix_multiplication/matrix_mul.py"],
            created_at="2026-06-03T15:23:00",
            session_id="test_session",
        )
        session.add_turn(turn)

        formatted = session.format_for_prompt()
        assert "Recent Session History" in formatted
        assert "matrix_demo" in formatted
        assert "matrix_multiplication/matrix_mul.py" in formatted
        assert "✅" in formatted

    def test_format_empty(self, session):
        assert session.format_for_prompt() == ""


# ═══════════════════════════════════════════════════════════════════
# Test: ProjectMemory
# ═══════════════════════════════════════════════════════════════════

class TestProjectMemory:
    """Verify ProjectMemory persistence + search."""

    @pytest.fixture
    def pm(self, tmp_path):
        from memory.project import ProjectMemory
        return ProjectMemory(workspace_root=tmp_path, max_turns=100)

    def _make_turn(self, id, user_task="task", final_answer="result", **kwargs):
        return TurnRecord(
            id=id, user_task=user_task, final_answer=final_answer,
            success=True, mode="plan",
            **kwargs,
        )

    def test_add_and_load_recent(self, pm):
        pm.add_turn(self._make_turn("t1", "First task", "Done"))
        pm.add_turn(self._make_turn("t2", "Second task", "Also done"))
        recent = pm.load_recent(5)
        assert len(recent) == 2
        assert recent[0].user_task == "First task"

    def test_search_returns_relevant(self, pm):
        pm.add_turn(self._make_turn(
            "t1", "Implement matrix multiplication",
            "Created matrix_mul.py", files_changed=["matrix_mul.py"],
        ))
        pm.add_turn(self._make_turn(
            "t2", "Add logging to all modules",
            "Added logging", files_changed=["main.py", "utils.py"],
        ))

        results = pm.search("matrix", k=3)
        assert len(results) >= 1
        assert "matrix" in results[0].user_task.lower()

    def test_search_empty(self, pm):
        results = pm.search("anything")
        assert results == []

    def test_clear(self, pm):
        pm.add_turn(self._make_turn("t1", "a", "b"))
        pm.clear()
        assert pm.count == 0

    def test_count(self, pm):
        assert pm.count == 0
        pm.add_turn(self._make_turn("t1", "a", "b"))
        assert pm.count == 1


# ═══════════════════════════════════════════════════════════════════
# Test: Meta Query Detection
# ═══════════════════════════════════════════════════════════════════

class TestMetaQuery:
    """Verify meta-query detection for session-aware recall."""

    def test_chinese_meta(self):
        assert is_meta_query("刚才完成了什么") is True
        assert is_meta_query("删除刚刚完成的任务") is True
        assert is_meta_query("上一个任务是什么") is True
        assert is_meta_query("之前修改了什么") is True

    def test_english_meta(self):
        assert is_meta_query("what did I just do") is True
        assert is_meta_query("what was the previous task") is True
        assert is_meta_query("last task result") is True

    def test_non_meta(self):
        assert is_meta_query("Fix all import errors") is False
        assert is_meta_query("Create a new module") is False
        assert is_meta_query("matrix multiplication") is False


# ═══════════════════════════════════════════════════════════════════
# Test: MemoryManager
# ═══════════════════════════════════════════════════════════════════

class TestMemoryManager:
    """Verify MemoryManager orchestration."""

    @pytest.fixture
    def mm(self, tmp_path):
        from memory.manager import MemoryManager
        return MemoryManager(workspace_root=tmp_path, enabled=True)

    def _make_turn(self, id, user_task="task", final_answer="result", **kwargs):
        return TurnRecord(
            id=id, user_task=user_task, final_answer=final_answer,
            success=True, mode="react",
            created_at="2026-06-03T15:00:00",
            **kwargs,
        )

    def test_build_context_empty(self, mm):
        ctx = mm.build_context_for_task("Fix bug")
        assert ctx == ""  # No turns yet

    def test_build_context_with_session_turns(self, mm):
        """Session turn appears in context."""
        # Simulate recording a turn
        turn = self._make_turn(
            "turn_a", "新建文件夹 matrix_demo",
            "Created matrix_multiplication/matrix_mul.py",
            files_changed=["matrix_mul.py"],
        )
        mm._session.add_turn(turn)

        ctx = mm.build_context_for_task("删除刚刚完成的任务")
        assert "Recent Session History" in ctx
        assert "matrix_demo" in ctx  # Should recall the previous turn

    def test_meta_query_includes_session(self, mm):
        """Meta queries include session history even with empty project."""
        turn = self._make_turn(
            "turn_prev", "Implement auth module",
            "Created auth.py with login/logout",
        )
        mm._session.add_turn(turn)

        ctx = mm.build_context_for_task("刚才完成了什么")
        assert "Recent Session History" in ctx
        assert "Implement auth module" in ctx

    @pytest.mark.asyncio
    async def test_record_completed_turn(self, mm):
        """record_completed_turn creates and persists a TurnRecord."""
        state = {
            "task": "Fix import error",
            "mode": "react",
            "tool_history": [
                {"tool": "read_file", "args": {"file_path": "main.py"}, "result": "ok", "success": True},
                {"tool": "edit_file", "args": {"file_path": "main.py", "old_string": "x", "new_string": "y"}, "result": "changed", "success": True},
            ],
        }

        turn = await mm.record_completed_turn(state, "Fixed the import", success=True)
        assert turn is not None
        assert turn.user_task == "Fix import error"
        assert turn.final_answer == "Fixed the import"
        assert turn.success is True
        assert "main.py" in turn.files_changed
        assert "read_file" in turn.tools_used
        assert "edit_file" in turn.tools_used

        # Should be in session
        assert mm._session.turn_count == 1
        # Should be persisted
        assert mm._project.count == 1

    @pytest.mark.asyncio
    async def test_record_completed_turn_with_final_answer(self, mm):
        """final_answer is always non-empty (B3 fix — called after finish_node)."""
        state = {"task": "Test", "mode": "plan", "tool_history": []}

        turn = await mm.record_completed_turn(
            state, "Task completed successfully.", success=True
        )
        assert turn is not None
        assert turn.final_answer == "Task completed successfully."
        assert turn.final_answer != ""  # B3: must NOT be empty

    def test_memory_disabled(self, tmp_path):
        """memory_enabled=False makes all operations no-ops."""
        from memory.manager import MemoryManager
        mm = MemoryManager(workspace_root=tmp_path, enabled=False)

        assert mm.build_context_for_task("test") == ""
        assert mm.enabled is False
        assert mm.session is None
        assert mm.project is None

    @pytest.mark.asyncio
    async def test_record_when_disabled(self, tmp_path):
        """record_completed_turn with disabled memory returns None."""
        from memory.manager import MemoryManager
        mm = MemoryManager(workspace_root=tmp_path, enabled=False)

        turn = await mm.record_completed_turn(
            {"task": "test"}, "done", success=True
        )
        assert turn is None

    def test_get_session_turns(self, mm):
        assert mm.get_session_turns() == []
        t = self._make_turn("tx", "Task X", "Result X")
        mm._session.add_turn(t)
        assert len(mm.get_session_turns()) == 1

    def test_get_project_turn_count(self, mm):
        assert mm.get_project_turn_count() == 0

    @pytest.mark.asyncio
    async def test_clear_all(self, mm):
        t = self._make_turn("tc", "Task C", "Result C")
        mm._session.add_turn(t)
        mm._project.add_turn(t)
        assert mm.get_session_turns()
        assert mm.get_project_turn_count() == 1

        await mm.clear_all()
        assert mm.get_session_turns() == []
        assert mm.get_project_turn_count() == 0


# ═══════════════════════════════════════════════════════════════════
# Test: init_node with MemoryManager (integration)
# ═══════════════════════════════════════════════════════════════════

class TestInitNodeWithMemoryManager:
    """Verify init_node injects session_context from MemoryManager."""

    @pytest.mark.asyncio
    async def test_init_node_injects_session_context(self, tmp_path):
        from graph.nodes import init_node
        from memory.manager import MemoryManager

        mm = MemoryManager(workspace_root=tmp_path, enabled=True)
        # Add a session turn
        from memory.types import TurnRecord
        turn = TurnRecord(
            id="turn_test", user_task="Previous task",
            final_answer="Did something.",
            success=True, mode="react",
            created_at="2026-06-03T15:00:00",
        )
        mm._session.add_turn(turn)

        state = {"task": "Delete what I just did", "mode": "react"}
        result = await init_node(state, memory_manager=mm)

        assert "session_context" in result
        assert "Previous task" in result["session_context"]

    @pytest.mark.asyncio
    async def test_init_node_without_memory(self):
        from graph.nodes import init_node

        state = {"task": "test", "mode": "plan"}
        result = await init_node(state, memory_manager=None)

        assert result.get("session_context", "") == ""

    @pytest.mark.asyncio
    async def test_init_node_memory_disabled(self, tmp_path):
        from graph.nodes import init_node
        from memory.manager import MemoryManager

        mm = MemoryManager(workspace_root=tmp_path, enabled=False)
        state = {"task": "test", "mode": "react"}
        result = await init_node(state, memory_manager=mm)

        assert result.get("session_context", "") == ""


# ═══════════════════════════════════════════════════════════════════
# Test: memory_enabled flag in agent
# ═══════════════════════════════════════════════════════════════════

class TestMemoryDisabled:
    """Verify memory_enabled=False prevents memory operations."""

    def test_agent_with_memory_disabled(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        class DummyLLM:
            def bind_tools(self, schemas):
                return self

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=DummyLLM(),
            memory_enabled=False,
        )
        assert agent._memory_manager is None
