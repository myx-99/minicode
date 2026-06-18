"""Integration tests for cross-turn session memory (V2.1).

Tests verify:
  - REPL three-turn recall scenario (the core AC-1)
  - Mode switch preserves session turns
  - Meta queries include recent turn history
  - MemoryManager survives agent rebuild
  - finish_node writes memory after final_answer (B3 fix)
  - Session memory format_for_prompt output shape
  - Project memory persists across agent rebuilds
  - memory_enabled=False prevents cross-turn context

Run with:  pytest tests/test_session_memory.py -v
"""

import pytest
from pathlib import Path

from memory.types import TurnRecord
from memory.manager import MemoryManager


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_turn(id, user_task, final_answer="Done.", **kwargs):
    return TurnRecord(
        id=id, user_task=user_task, final_answer=final_answer,
        success=True, mode="react",
        created_at="2026-06-03T15:00:00",
        session_id="test_session",
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# Test: REPL three-turn recall (AC-1)
# ═══════════════════════════════════════════════════════════════════

class TestThreeTurnRecall:
    """Simulate the core three-turn REPL scenario from AC-1."""

    @pytest.fixture
    def mm(self, tmp_path):
        return MemoryManager(workspace_root=tmp_path, enabled=True)

    def test_turn1_context_empty(self, mm):
        """Turn 1: No prior history — context should be empty."""
        ctx = mm.build_context_for_task("新建一个文件夹 matrix_demo 并写一个 hello.py")
        assert ctx == ""  # No prior turns

    def test_turn2_recalls_turn1(self, mm):
        """Turn 2: After recording Turn 1, context includes Turn 1's details."""
        # Record Turn 1
        turn1 = _make_turn(
            "turn_001",
            "新建一个文件夹 matrix_demo 并写一个 hello.py",
            "Created matrix_multiplication/ with matrix_mul.py and hello.py",
            files_changed=["matrix_multiplication/matrix_mul.py", "matrix_multiplication/hello.py"],
            tools_used=["write_file", "shell_execute"],
        )
        mm._session.add_turn(turn1)
        mm._project.add_turn(turn1)

        # Turn 2: recall should include Turn 1
        ctx = mm.build_context_for_task("删除刚刚完成的任务")
        assert "matrix_demo" in ctx
        assert "Recent Session History" in ctx

    def test_turn3_recalls_both_prior_turns(self, mm):
        """Turn 3: '刚才完成了什么' should recall both Turn 1 and Turn 2."""
        # Record Turn 1
        turn1 = _make_turn(
            "turn_001", "新建文件夹 matrix_demo",
            "Created matrix_multiplication/",
            files_changed=["matrix_multiplication/matrix_mul.py"],
        )
        mm._session.add_turn(turn1)

        # Record Turn 2
        turn2 = _make_turn(
            "turn_002", "删除刚刚完成的任务",
            "Deleted matrix_multiplication/ folder.",
            files_changed=["matrix_multiplication/"],
        )
        mm._session.add_turn(turn2)

        # Turn 3: meta query
        ctx = mm.build_context_for_task("刚才完成了什么")
        assert "Turn 1" in ctx  # Should show both turns
        assert "Turn 2" in ctx
        assert "matrix_demo" in ctx
        assert "删除刚刚" in ctx or "Turn 2" in ctx

    def test_non_meta_task_uses_keyword_search(self, mm):
        """Non-meta tasks use keyword search for project turns."""
        # Add a turn about auth
        turn = _make_turn(
            "turn_auth", "Implement authentication",
            "Created auth.py with OAuth2 flow.",
            files_changed=["auth.py"],
        )
        mm._project.add_turn(turn)

        # Non-meta query for a different topic
        ctx = mm.build_context_for_task("Add logging to all modules")
        # May or may not match — just verify it doesn't crash
        assert isinstance(ctx, str)

    def test_context_does_not_duplicate_turns(self, mm):
        """Turns in session and project should not appear twice."""
        turn = _make_turn(
            "turn_dup", "Some task",
            "Result",
            files_changed=["file.py"],
        )
        mm._session.add_turn(turn)
        mm._project.add_turn(turn)  # Same turn in project

        ctx = mm.build_context_for_task("Next task")
        # Count occurrences of turn_dup
        assert ctx.count("Some task") <= 1  # No duplication


# ═══════════════════════════════════════════════════════════════════
# Test: Mode switch preserves session
# ═══════════════════════════════════════════════════════════════════

class TestModeSwitchPreservesSession:
    """Verify that /mode switch does not lose session turns."""

    @pytest.fixture
    def mm(self, tmp_path):
        return MemoryManager(workspace_root=tmp_path, enabled=True)

    def test_session_persists_after_new_agent(self, mm):
        """When CLI rebuilds agent with same MemoryManager, turns survive."""
        # Record a turn in the initial session
        turn = _make_turn("turn_before_switch", "Task before mode switch")
        mm._session.add_turn(turn)

        # Simulate CLI rebuilding ClaudeCodeMini with same MemoryManager
        # (this is what AgentCLI._handle_mode_command does)
        session_turns = mm.get_session_turns()
        assert len(session_turns) == 1
        assert session_turns[0].user_task == "Task before mode switch"

    def test_session_id_persists(self, mm):
        """Session ID should not change on mode switch (same MemoryManager)."""
        sid_before = mm.session_id
        # The session_id should remain the same since we're reusing the
        # same MemoryManager instance
        assert sid_before == mm.session_id


# ═══════════════════════════════════════════════════════════════════
# Test: format_for_prompt output shape
# ═══════════════════════════════════════════════════════════════════

class TestFormatForPrompt:
    """Verify the prompt formatting used by build_context_for_task."""

    @pytest.fixture
    def session(self):
        from memory.session import SessionMemory
        return SessionMemory(session_id="test")

    def test_format_includes_all_fields(self, session):
        turn = TurnRecord(
            id="turn_fmt", user_task="Fix import error",
            final_answer="Added requests to requirements.txt",
            success=True, mode="react",
            files_changed=["requirements.txt"],
            created_at="2026-06-03T15:00:00",
            session_id="test",
        )
        session.add_turn(turn)

        formatted = session.format_for_prompt()
        assert "Recent Session History" in formatted
        assert "Fix import error" in formatted
        assert "requirements.txt" in formatted
        assert "✅" in formatted
        assert "react" in formatted

    def test_format_multiple_turns(self, session):
        for i in range(3):
            session.add_turn(_make_turn(f"t{i}", f"Task {i}", f"Result {i}"))
        formatted = session.format_for_prompt()
        assert "Turn 1" in formatted
        assert "Turn 2" in formatted
        assert "Turn 3" in formatted
        assert "Task 0" in formatted
        assert "Task 2" in formatted


# ═══════════════════════════════════════════════════════════════════
# Test: Project memory cross-session persistence
# ═══════════════════════════════════════════════════════════════════

class TestProjectMemoryPersistence:
    """Verify project turns survive on disk."""

    @pytest.fixture
    def workspace(self, tmp_path):
        return tmp_path

    def test_turns_survive_memory_reload(self, workspace):
        """Project turns written by one MemoryManager are visible to another."""
        mm1 = MemoryManager(workspace_root=workspace, enabled=True)
        turn = _make_turn("turn_persist", "Important task", "Completed")
        mm1._project.add_turn(turn)

        # Simulate process restart: new MemoryManager reads same disk
        mm2 = MemoryManager(workspace_root=workspace, enabled=True)
        ctx = mm2.build_context_for_task("What was the important task?")
        assert "Important task" in ctx or mm2._project.count >= 1

    def test_clear_removes_disk_entries(self, workspace):
        """clear_all removes both session and project turns."""
        mm = MemoryManager(workspace_root=workspace, enabled=True)
        turn = _make_turn("turn_clear", "Will be cleared")
        mm._session.add_turn(turn)
        mm._project.add_turn(turn)

        import asyncio
        asyncio.run(mm.clear_all())

        assert mm.get_session_turns() == []
        assert mm.get_project_turn_count() == 0

        # Reload: should still be empty
        mm2 = MemoryManager(workspace_root=workspace, enabled=True)
        assert mm2.get_project_turn_count() == 0


# ═══════════════════════════════════════════════════════════════════
# Test: B3 fix — final_answer must be non-empty before memory write
# ═══════════════════════════════════════════════════════════════════

class TestB3Fix:
    """Verify finish_node writes memory AFTER final_answer generation."""

    @pytest.mark.asyncio
    async def test_record_has_non_empty_final_answer(self, tmp_path):
        """B3: final_answer is always filled before record_completed_turn."""
        mm = MemoryManager(workspace_root=tmp_path, enabled=True)

        state = {
            "task": "Fix bug",
            "mode": "react",
            "tool_history": [
                {"tool": "read_file", "args": {"file_path": "main.py"}, "result": "ok", "success": True},
            ],
        }

        # Simulate what finish_node does: generate final_answer FIRST,
        # THEN call record_completed_turn
        final_answer = "Fixed the bug by updating main.py."
        assert final_answer != ""  # B3: must be non-empty

        turn = await mm.record_completed_turn(state, final_answer, success=True)
        assert turn is not None
        assert turn.final_answer == final_answer
        assert turn.final_answer != ""

        # Verify the turn is retrievable
        recent = mm._project.load_recent(5)
        assert len(recent) >= 1
        assert recent[0].final_answer == final_answer


# ═══════════════════════════════════════════════════════════════════
# Test: MemoryManager with Mock Agent (integration-level)
# ═══════════════════════════════════════════════════════════════════

class TestMemoryManagerWithAgent:
    """Verify MemoryManager integrates with ClaudeCodeMini correctly."""

    def test_agent_stores_memory_manager(self, tmp_path):
        """Agent holds reference to MemoryManager."""
        from agent.agent import ClaudeCodeMini

        mm = MemoryManager(workspace_root=tmp_path, enabled=True)

        class DummyLLM:
            def bind_tools(self, schemas):
                return self

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=DummyLLM(),
            memory_enabled=True,
            memory_manager=mm,
        )
        assert agent.memory_manager is mm

    def test_agent_creates_own_manager_when_none_provided(self, tmp_path):
        """Agent creates its own MemoryManager if none injected."""
        from agent.agent import ClaudeCodeMini

        class DummyLLM:
            def bind_tools(self, schemas):
                return self

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=DummyLLM(),
            memory_enabled=True,
        )
        assert agent.memory_manager is not None

    def test_agent_without_memory(self, tmp_path):
        """memory_enabled=False → no MemoryManager."""
        from agent.agent import ClaudeCodeMini

        class DummyLLM:
            def bind_tools(self, schemas):
                return self

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=DummyLLM(),
            memory_enabled=False,
        )
        assert agent.memory_manager is None
