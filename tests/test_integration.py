"""Integration tests for Phase 4 — CLI + Full End-to-End flows.

Tests cover:
  - ClaudeCodeMini agent with real tools on a tmp_path workspace
  - CLI print_result formatting
  - Agent error handling and edge cases
  - main.py argument parsing
  - Stream event tracking

Run with:  pytest tests/test_integration.py -v
"""

import pytest
import json
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class MockLLM:
    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0
        self._bound_tools = None

    async def ainvoke(self, messages, **kwargs):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, str):
                return AIMessage(content=resp)
            return resp
        return AIMessage(content="Default fallback.")

    def bind_tools(self, schemas):
        self._bound_tools = schemas
        return self


def make_tool_call_ai(tool_calls):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tc[0], "args": tc[1], "id": tc[2] if len(tc) > 2 else f"id_{tc[0]}", "type": "tool_call"}
            for tc in tool_calls
        ],
    )


def make_text_ai(text):
    return AIMessage(content=text)


_SUCCESS_REFLECTION = json.dumps({
    "step_done": True, "success": True, "error_type": "none",
    "reasoning": "Done.", "should_retry": False,
    "should_replan": False, "retry_suggestion": "",
})


# ═══════════════════════════════════════════════════════════════════
# Test: Agent with Real Filesystem
# ═══════════════════════════════════════════════════════════════════

class TestAgentRealFilesystem:
    """Agent operates on a real tmp_path workspace with mock LLM."""

    @pytest.mark.asyncio
    async def test_agent_creates_file(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan = json.dumps([{"id": "1", "description": "Create hello.py"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("write_file", {"file_path": "hello.py", "content": "print('hello')"}, "c1")]),
            make_text_ai("Created hello.py successfully."),
            _SUCCESS_REFLECTION,
            "Task complete: created hello.py",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Create a hello.py file")

        assert result["success"] is True
        assert (tmp_path / "hello.py").exists()
        assert (tmp_path / "hello.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_agent_reads_and_edits_file(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        (tmp_path / "config.py").write_text("DEBUG=False\nPORT=8080\n")

        plan = json.dumps([
            {"id": "1", "description": "Read config.py"},
            {"id": "2", "description": "Change DEBUG to True"},
        ])

        llm = MockLLM(responses=[
            plan,
            # Step 1: read
            make_tool_call_ai([("read_file", {"file_path": "config.py"}, "c1")]),
            make_text_ai("Read config.py: DEBUG=False, PORT=8080."),
            _SUCCESS_REFLECTION,
            # Step 2: edit
            make_tool_call_ai([("edit_file", {"file_path": "config.py", "old_string": "DEBUG=False", "new_string": "DEBUG=True"}, "c2")]),
            make_text_ai("Changed DEBUG to True."),
            _SUCCESS_REFLECTION,
            "Task complete: DEBUG is now True.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Change DEBUG to True in config.py")

        assert result["success"] is True
        content = (tmp_path / "config.py").read_text()
        assert "DEBUG=True" in content

    @pytest.mark.asyncio
    async def test_agent_searches_files(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("")

        plan = json.dumps([{"id": "1", "description": "Find all Python files"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("glob_search", {"pattern": "**/*.py"}, "c1")]),
            make_text_ai("Found Python files."),
            _SUCCESS_REFLECTION,
            "Found Python files in the project.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Find all Python files")

        assert result["success"] is True
        assert len(result["tool_history"]) >= 1

    @pytest.mark.asyncio
    async def test_agent_runs_shell(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        (tmp_path / "hello.txt").write_text("Hello World")

        plan = json.dumps([{"id": "1", "description": "Check directory contents"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("shell_execute", {"command": "echo hello"}, "c1")]),
            make_text_ai("Command executed successfully."),
            _SUCCESS_REFLECTION,
            "Task complete.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("List files in project")

        assert result["success"] is True


# ═══════════════════════════════════════════════════════════════════
# Test: Agent Error Handling
# ═══════════════════════════════════════════════════════════════════

class TestAgentErrors:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_agent_handles_tool_error(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan = json.dumps([{"id": "1", "description": "Read nonexistent file"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("read_file", {"file_path": "does_not_exist.txt"}, "c1")]),
            make_text_ai("File doesn't exist. Cannot complete."),
            json.dumps({
                "step_done": True, "success": False, "error_type": "fatal",
                "reasoning": "File not found.", "should_retry": False,
                "should_replan": False, "retry_suggestion": "",
            }),
            "Task attempted but file was missing.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Read nonexistent file")

        assert result["phase"] in ("done", "error")
        assert "final_answer" in result

    @pytest.mark.asyncio
    async def test_agent_graph_exception_caught(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        with patch("agent.agent.build_graph") as mock_build:
            mock_build.return_value.ainvoke.side_effect = RuntimeError("Graph exploded")

            llm = MockLLM()
            agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
            result = await agent.run("test task")

            assert result["success"] is False
            assert "Graph exploded" in result["final_answer"]
            assert result["phase"] == "error"

    @pytest.mark.asyncio
    async def test_agent_rejects_empty_task(self, tmp_path):
        from agent.agent import ClaudeCodeMini
        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=MockLLM())
        with pytest.raises(ValueError, match="empty"):
            await agent.run("")

    @pytest.mark.asyncio
    async def test_agent_rejects_whitespace_task(self, tmp_path):
        from agent.agent import ClaudeCodeMini
        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=MockLLM())
        with pytest.raises(ValueError, match="empty"):
            await agent.run("   \n  ")

    @pytest.mark.asyncio
    async def test_agent_max_iterations_does_not_crash(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan = json.dumps([{"id": "1", "description": "Step"}])
        responses = [plan]
        for _ in range(10):
            responses.append(make_tool_call_ai([("glob_search", {"pattern": "*.py"}, "c")]))

        llm = MockLLM(responses=responses)
        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm, max_iterations=2)
        result = await agent.run("test")

        assert result["phase"] in ("done", "error")


# ═══════════════════════════════════════════════════════════════════
# Test: Stream Events
# ═══════════════════════════════════════════════════════════════════

class TestStreamEvents:
    """Tests for the streaming API."""

    @pytest.mark.asyncio
    async def test_stream_yields_multiple_events(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan = json.dumps([{"id": "1", "description": "Search files"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("glob_search", {"pattern": "*.py"}, "c1")]),
            make_text_ai("Found files."),
            _SUCCESS_REFLECTION,
            "Done.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)

        events = []
        async for event in agent.stream("Find files"):
            events.append(event)

        phases = [e.get("phase") for e in events if e.get("phase")]
        assert "planning" in phases or "executing" in phases
        assert events[-1]["phase"] == "done"

    @pytest.mark.asyncio
    async def test_stream_final_state_has_plan(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan = json.dumps([{"id": "1", "description": "Search files"}])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([("glob_search", {"pattern": "*"}, "c1")]),
            make_text_ai("Done."),
            _SUCCESS_REFLECTION,
            "Summary.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm, mode="plan")

        final = None
        async for event in agent.stream("Search"):
            final = event

        assert final is not None
        assert final["phase"] == "done"
        assert len(final["plan"]) == 1


# ═══════════════════════════════════════════════════════════════════
# Test: CLI Module
# ═══════════════════════════════════════════════════════════════════

class TestCLIModule:
    """Tests for CLI app functions (no real LLM calls)."""

    def test_print_result_success(self):
        from cli.app import print_result

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)

        result = {
            "success": True,
            "final_answer": "All done! Created hello.py.",
            "plan": [{"id": "1", "description": "Create file", "status": "done"}],
            "tool_history": [
                {"tool": "write_file", "args": {"file_path": "hello.py"}, "result": "Created", "success": True},
            ],
            "phase": "done", "error_message": "", "iteration": 5,
        }
        print_result(result, console=console)
        output = buf.getvalue()
        assert "Task Complete" in output

    def test_print_result_failure(self):
        from cli.app import print_result

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)

        result = {
            "success": False,
            "final_answer": "Task failed.",
            "plan": [{"id": "1", "description": "Read file", "status": "failed"}],
            "tool_history": [],
            "phase": "error", "error_message": "File not found", "iteration": 2,
        }
        print_result(result, console=console)
        output = buf.getvalue()
        assert "Task Failed" in output

    def test_print_result_empty(self):
        from cli.app import print_result

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)

        result = {
            "success": True, "final_answer": "", "plan": [], "tool_history": [],
            "phase": "done", "error_message": "", "iteration": 0,
        }
        print_result(result, console=console)
        # Should not raise

    def test_error_panel_brackets_in_message(self):
        from cli.app import _error_panel

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        console.print(_error_panel("bad tag [/red] at position [58]"))
        assert "[/red]" in buf.getvalue()

    def test_print_plan_pending_step_no_markup_error(self, tmp_path):
        with patch("cli.app.ClaudeCodeMini") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.workspace.root_str = str(tmp_path)
            mock_agent.tool_registry.tool_names = ["glob_search"]
            mock_agent_cls.return_value = mock_agent

            from cli.app import AgentCLI

            buf = io.StringIO()
            cli = AgentCLI(workspace_path=str(tmp_path))
            cli._console = Console(file=buf, force_terminal=False, width=120)
            cli._print_plan(
                [
                    {"id": "1", "description": "Use glob_search with [*.py]", "status": "pending"},
                    {"id": "2", "description": "write_file sort.py", "status": "pending"},
                ],
                current_idx=0,
            )
        assert "glob_search" in buf.getvalue()

    def test_cli_instantiation(self, tmp_path):
        with patch("cli.app.ClaudeCodeMini") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.workspace.root_str = str(tmp_path)
            mock_agent.tool_registry.tool_names = ["read_file", "write_file"]
            mock_agent._max_iterations = 30
            mock_agent_cls.return_value = mock_agent

            from cli.app import AgentCLI
            cli = AgentCLI(workspace_path=str(tmp_path))
            assert cli._agent is not None


# ═══════════════════════════════════════════════════════════════════
# Test: main.py argument parsing
# ═══════════════════════════════════════════════════════════════════

class TestMainArgs:
    """Tests for main.py argument parsing."""

    def test_default_args(self):
        from main import parse_args
        with patch("sys.argv", ["main.py"]):
            args = parse_args()
            assert args.task is None
            assert args.max_iters == 30
            assert args.max_retries == 2
            assert args.raw is False

    def test_task_arg(self):
        from main import parse_args
        with patch("sys.argv", ["main.py", "Fix bugs"]):
            args = parse_args()
            assert args.task == "Fix bugs"

    def test_all_args(self):
        from main import parse_args
        with patch("sys.argv", [
            "main.py", "Add feature", "-w", "/my/project",
            "-m", "gpt-4o-mini", "--max-iters", "50",
            "--max-retries", "3", "--raw",
        ]):
            args = parse_args()
            assert args.task == "Add feature"
            assert args.workspace == "/my/project"
            assert args.model == "gpt-4o-mini"
            assert args.max_iters == 50
            assert args.max_retries == 3
            assert args.raw is True


# ═══════════════════════════════════════════════════════════════════
# Test: Complete Workflow Simulations
# ═══════════════════════════════════════════════════════════════════

class TestCompleteWorkflows:
    """Simulate realistic multi-step coding tasks."""

    @pytest.mark.asyncio
    async def test_fix_import_error_workflow(self, tmp_path):
        """Simulate: user asks to fix an import error."""
        from agent.agent import ClaudeCodeMini

        (tmp_path / "main.py").write_text("import nonexistent_module\nprint('hello')\n")

        plan = json.dumps([
            {"id": "1", "description": "Search for Python files"},
            {"id": "2", "description": "Read main.py"},
            {"id": "3", "description": "Fix the bad import"},
            {"id": "4", "description": "Verify the fix"},
        ])

        llm = MockLLM(responses=[
            plan,
            # Step 1
            make_tool_call_ai([("glob_search", {"pattern": "**/*.py"}, "c1")]),
            make_text_ai("Found main.py."),
            _SUCCESS_REFLECTION,
            # Step 2
            make_tool_call_ai([("read_file", {"file_path": "main.py"}, "c2")]),
            make_text_ai("Has bad import: nonexistent_module."),
            _SUCCESS_REFLECTION,
            # Step 3
            make_tool_call_ai([("edit_file", {
                "file_path": "main.py",
                "old_string": "import nonexistent_module\n",
                "new_string": "",
            }, "c3")]),
            make_text_ai("Removed the bad import."),
            _SUCCESS_REFLECTION,
            # Step 4
            make_tool_call_ai([("shell_execute", {"command": "python main.py"}, "c4")]),
            make_text_ai("Script runs successfully."),
            _SUCCESS_REFLECTION,
            "Fixed: removed the bad import from main.py.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Fix the import error in main.py")

        assert result["success"] is True
        assert "nonexistent_module" not in (tmp_path / "main.py").read_text()

    @pytest.mark.asyncio
    async def test_add_docstring_workflow(self, tmp_path):
        """Simulate: adding a docstring to a function."""
        from agent.agent import ClaudeCodeMini

        (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")

        plan = json.dumps([
            {"id": "1", "description": "Read utils.py and add docstring"},
        ])

        llm = MockLLM(responses=[
            plan,
            make_tool_call_ai([
                ("read_file", {"file_path": "utils.py"}, "c1"),
                ("edit_file", {
                    "file_path": "utils.py",
                    "old_string": "def add(a, b):\n    return a + b",
                    "new_string": 'def add(a, b):\n    """Add two numbers."""\n    return a + b',
                }, "c2"),
            ]),
            make_text_ai("Added docstring to add()."),
            _SUCCESS_REFLECTION,
            "Added docstring to add() in utils.py.",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        result = await agent.run("Add docstring to add() in utils.py")

        assert result["success"] is True
        content = (tmp_path / "utils.py").read_text()
        assert '"""Add two numbers."""' in content
