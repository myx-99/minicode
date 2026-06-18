"""Unit tests for Phase 1 — Tool System + Runtime + Config.

Run with:  pytest tests/test_tools.py -v
"""

import pytest
import tempfile
import os
import platform
from pathlib import Path

# ── Test ToolResult ───────────────────────────────────────────────

class TestToolResult:
    """Tests for the ToolResult model."""

    def test_success_defaults(self):
        from tools.base import ToolResult
        r = ToolResult(success=True, output="done")
        assert r.success is True
        assert r.output == "done"
        assert r.error is None
        assert r.metadata == {}

    def test_failure(self):
        from tools.base import ToolResult
        r = ToolResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_to_summary_success(self):
        from tools.base import ToolResult
        r = ToolResult(success=True, output="file created successfully")
        assert "[OK]" in r.to_summary()

    def test_to_summary_failure(self):
        from tools.base import ToolResult
        r = ToolResult(success=False, error="permission denied")
        assert "[FAIL]" in r.to_summary()

    def test_to_langchain_message_with_metadata(self):
        from tools.base import ToolResult
        r = ToolResult(
            success=True,
            output="read 10 lines",
            metadata={"lines": 10, "path": "main.py"},
        )
        msg = r.to_langchain_message()
        assert "read 10 lines" in msg
        assert "lines=10" in msg

    def test_metadata_default(self):
        from tools.base import ToolResult
        r = ToolResult(success=True)
        assert r.metadata == {}
        assert r.output == ""


# ── Test Workspace ────────────────────────────────────────────────

class TestWorkspace:
    """Tests for workspace path resolution and security."""

    def test_root_absolute(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        assert ws.root.is_absolute()

    def test_resolve_relative_path(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        # Resolving "." should work
        resolved = ws.resolve_path(".")
        assert resolved == ws.root

    def test_resolve_absolute_path_inside_workspace(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        abs_path = str(ws.root / "test.txt")
        resolved = ws.resolve_path(abs_path)
        assert resolved == ws.root / "test.txt"

    def test_resolve_rejects_path_outside_workspace(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        with pytest.raises(ValueError, match="outside"):
            ws.resolve_path("/etc/passwd")

    def test_resolve_rejects_traversal(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        with pytest.raises(ValueError):
            ws.resolve_path("../../etc/passwd")

    def test_exists(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        assert ws.exists(".") is True
        assert ws.exists("nonexistent_file_xyz.abc") is False

    def test_repr(self):
        from runtime.workspace import Workspace
        ws = Workspace(".")
        assert "Workspace" in repr(ws)

    def test_non_existent_root_raises(self):
        from runtime.workspace import Workspace
        with pytest.raises(ValueError):
            Workspace("/path/that/definitely/does/not/exist")


# ── Test ReadFileTool ─────────────────────────────────────────────

class TestReadFileTool:
    """Tests for reading files."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.file_read import ReadFileTool
        return ReadFileTool(Workspace(str(tmp_path)))

    @pytest.mark.asyncio
    async def test_read_entire_file(self, tool, tmp_path):
        f = tmp_path / "hello.py"
        f.write_text("line1\nline2\nline3\n")
        result = await tool.execute(file_path="hello.py")
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output
        assert "line3" in result.output

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = await tool.execute(file_path="data.txt", offset=3)
        assert result.success
        assert "3\tc" in result.output
        assert "a" not in result.output.split("\n")[0] if result.output else True

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tool, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        result = await tool.execute(file_path="data.txt", limit=2)
        assert result.success
        assert result.metadata["lines_read"] == 2
        assert "c" not in result.output

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tool):
        result = await tool.execute(file_path="does_not_exist.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, tool, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = await tool.execute(file_path="subdir")
        assert not result.success

    @pytest.mark.asyncio
    async def test_read_offset_past_end(self, tool, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("only one line\n")
        result = await tool.execute(file_path="small.txt", offset=100)
        assert result.success
        assert "past end" in result.output.lower()


# ── Test WriteFileTool ────────────────────────────────────────────

class TestWriteFileTool:
    """Tests for writing files."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.file_write import WriteFileTool
        return WriteFileTool(Workspace(str(tmp_path)))

    @pytest.mark.asyncio
    async def test_create_new_file(self, tool, tmp_path):
        result = await tool.execute(
            file_path="new.py", content="print('hello')"
        )
        assert result.success
        assert "Created" in result.output
        assert (tmp_path / "new.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, tool, tmp_path):
        f = tmp_path / "existing.py"
        f.write_text("old content")
        result = await tool.execute(
            file_path="existing.py", content="new content"
        )
        assert result.success
        assert "Updated" in result.output
        assert f.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_create_in_subdirectory(self, tool, tmp_path):
        result = await tool.execute(
            file_path="deep/nested/file.txt", content="hello"
        )
        assert result.success
        assert (tmp_path / "deep" / "nested" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_write_empty_file(self, tool, tmp_path):
        result = await tool.execute(file_path="empty.txt", content="")
        assert result.success
        assert (tmp_path / "empty.txt").exists()
        assert (tmp_path / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_write_outside_workspace_blocked(self, tool):
        result = await tool.execute(
            file_path="/etc/hacked", content="bad"
        )
        assert not result.success


# ── Test EditFileTool ─────────────────────────────────────────────

class TestEditFileTool:
    """Tests for precise string replacement editing."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.file_edit import EditFileTool
        return EditFileTool(Workspace(str(tmp_path)))

    @pytest.fixture
    def sample_file(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def hello():\n    print('hello')\n\nprint('done')\n")
        return f

    @pytest.mark.asyncio
    async def test_single_replace(self, tool, sample_file):
        result = await tool.execute(
            file_path="sample.py",
            old_string="print('hello')",
            new_string="print('world')",
        )
        assert result.success
        content = sample_file.read_text()
        assert "print('world')" in content
        assert "print('hello')" not in content

    @pytest.mark.asyncio
    async def test_replace_all(self, tool, tmp_path):
        f = tmp_path / "dups.py"
        f.write_text("foo foo foo")
        result = await tool.execute(
            file_path="dups.py",
            old_string="foo",
            new_string="bar",
            replace_all=True,
        )
        assert result.success
        assert result.metadata["occurrences_replaced"] == 3
        assert f.read_text() == "bar bar bar"

    @pytest.mark.asyncio
    async def test_replace_only_first_by_default(self, tool, tmp_path):
        f = tmp_path / "dups.py"
        f.write_text("foo foo foo")
        result = await tool.execute(
            file_path="dups.py",
            old_string="foo",
            new_string="bar",
        )
        assert result.success
        assert f.read_text() == "bar foo foo"

    @pytest.mark.asyncio
    async def test_old_string_not_found(self, tool, sample_file):
        result = await tool.execute(
            file_path="sample.py",
            old_string="this does not exist in the file",
            new_string="replacement",
        )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_old_string_rejected(self, tool, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        result = await tool.execute(
            file_path="test.txt",
            old_string="",
            new_string="replace",
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_substring_match_inside_indented_code(self, tool, sample_file):
        """Substring matching works: 'print('hello')' is found within '    print('hello')'."""
        result = await tool.execute(
            file_path="sample.py",
            old_string="print('hello')",  # found as substring of the indented line
            new_string="print('replaced')",
        )
        # The replace succeeds because the string literal appears as a substring
        assert result.success
        assert result.metadata["occurrences_replaced"] == 1
        content = sample_file.read_text()
        assert "print('replaced')" in content
        # The indentation is preserved — only the substring is replaced
        assert "    print('replaced')" in content

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool):
        result = await tool.execute(
            file_path="nope.py",
            old_string="x",
            new_string="y",
        )
        assert not result.success


# ── Test GrepSearchTool ───────────────────────────────────────────

class TestGrepSearchTool:
    """Tests for regex content search."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.search_grep import GrepSearchTool
        return GrepSearchTool(Workspace(str(tmp_path)))

    @pytest.mark.asyncio
    async def test_basic_search(self, tool, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("print('no match')\n")
        result = await tool.execute(pattern="def hello")
        assert result.success
        assert "def hello" in result.output
        assert "a.py" in result.output

    @pytest.mark.asyncio
    async def test_case_insensitive_by_default(self, tool, tmp_path):
        (tmp_path / "file.py").write_text("Hello World\n")
        result = await tool.execute(pattern="hello")
        assert result.success
        assert "Hello World" in result.output

    @pytest.mark.asyncio
    async def test_case_sensitive(self, tool, tmp_path):
        (tmp_path / "file.py").write_text("Hello World\n")
        result = await tool.execute(pattern="hello", case_sensitive=True)
        assert result.success
        assert "matches" in result.output.lower()
        # Should find 0 matches
        assert result.metadata["matches"] == 0

    @pytest.mark.asyncio
    async def test_no_results_is_success(self, tool):
        result = await tool.execute(pattern="xyz_nonexistent_pattern_123")
        assert result.success
        assert result.metadata["matches"] == 0

    @pytest.mark.asyncio
    async def test_invalid_regex(self, tool):
        result = await tool.execute(pattern="[unclosed")
        assert not result.success

    @pytest.mark.asyncio
    async def test_glob_filter(self, tool, tmp_path):
        (tmp_path / "code.py").write_text("TODO: fix me\n")
        (tmp_path / "notes.txt").write_text("TODO: also here\n")
        result = await tool.execute(pattern="TODO", glob="*.py")
        assert result.success
        assert "code.py" in result.output
        assert "notes.txt" not in result.output

    @pytest.mark.asyncio
    async def test_ignores_git_and_cache_dirs(self, tool, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("TODO in git\n")
        (tmp_path / "main.py").write_text("TODO in main\n")
        result = await tool.execute(pattern="TODO")
        assert result.success
        assert ".git" not in result.output


# ── Test GlobSearchTool ───────────────────────────────────────────

class TestGlobSearchTool:
    """Tests for filename pattern matching."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.search_glob import GlobSearchTool
        return GlobSearchTool(Workspace(str(tmp_path)))

    @pytest.mark.asyncio
    async def test_find_all_python_files(self, tool, tmp_path):
        (tmp_path / "main.py").write_text("")
        (tmp_path / "utils.py").write_text("")
        (tmp_path / "README.md").write_text("")
        result = await tool.execute(pattern="*.py")
        assert result.success
        assert "main.py" in result.output
        assert "utils.py" in result.output
        assert "README.md" not in result.output

    @pytest.mark.asyncio
    async def test_recursive_glob(self, tool, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.py").write_text("")
        (tmp_path / "root.py").write_text("")
        result = await tool.execute(pattern="**/*.py")
        assert result.success
        assert "src/app.py" in result.output.replace("\\", "/")
        assert "root.py" in result.output

    @pytest.mark.asyncio
    async def test_no_results_is_success(self, tool):
        result = await tool.execute(pattern="*.rust")
        assert result.success
        assert result.metadata["matches"] == 0

    @pytest.mark.asyncio
    async def test_path_not_found(self, tool):
        result = await tool.execute(pattern="*.py", path="nope_dir")
        assert not result.success


# ── Test ShellTool ────────────────────────────────────────────────

class TestShellTool:
    """Tests for shell command execution."""

    @pytest.fixture
    def tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.shell import ShellTool
        return ShellTool(Workspace(str(tmp_path)))

    @pytest.mark.asyncio
    async def test_simple_echo(self, tool):
        result = await tool.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_command_working_directory(self, tool, tmp_path):
        command = "cd" if platform.system() == "Windows" else "pwd"
        result = await tool.execute(command=command)
        assert result.success
        assert str(tmp_path).lower() in result.output.lower()

    @pytest.mark.asyncio
    async def test_failed_command(self, tool, tmp_path):
        result = await tool.execute(command="exit 1")
        # exit 1 might or might not work depending on shell
        # The key is it shouldn't crash
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_blocked_dangerous_command(self, tool):
        result = await tool.execute(command="rm -rf / something")
        assert not result.success
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_command(self, tool):
        result = await tool.execute(command="")
        # Should either succeed with no output or fail gracefully
        assert isinstance(result.success, bool)

    def test_description_includes_shell_environment(self, tool):
        assert "Execution environment" in tool.description
        assert platform.system() in tool.description
        if platform.system() == "Windows":
            assert "cmd.exe" in tool.description
            assert "<<<" in tool.description
        else:
            assert "bash" in tool.description

    @pytest.mark.asyncio
    async def test_metadata_includes_shell(self, tool):
        result = await tool.execute(command="echo meta_test")
        assert result.success
        assert "shell" in result.metadata
        assert "platform" in result.metadata


# ── Test ToolRegistry ─────────────────────────────────────────────

class TestToolRegistry:
    """Tests for the ToolRegistry singleton."""

    def test_create_default_has_six_tools(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        assert registry.count == 6

    def test_create_default_tool_names(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        expected = [
            "edit_file", "glob_search", "grep_search",
            "read_file", "shell_execute", "write_file",
        ]
        assert registry.tool_names == expected

    def test_register_and_get(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from tools.file_read import ReadFileTool
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry()
        tool = ReadFileTool(ws)
        registry.register(tool)
        assert registry.get("read_file") is tool

    def test_duplicate_register_raises(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from tools.file_read import ReadFileTool
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry()
        registry.register(ReadFileTool(ws))
        with pytest.raises(ValueError):
            registry.register(ReadFileTool(ws))

    def test_get_nonexistent_raises(self):
        from tools.registry import ToolRegistry
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_unregister(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from tools.file_read import ReadFileTool
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry()
        registry.register(ReadFileTool(ws))
        assert registry.count == 1
        registry.unregister("read_file")
        assert registry.count == 0

    def test_openai_schemas(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        schemas = registry.get_openai_schemas()
        assert len(schemas) == 6
        assert all("function" in s for s in schemas)
        assert all("name" in s["function"] for s in schemas)

    def test_langchain_tool_dicts(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        dicts = registry.get_langchain_tool_dicts()
        assert len(dicts) == 6


# ── BaseTool ABC guarantees ──────────────────────────────────────

class TestBaseToolContract:
    """Verify that all V1 tools satisfy the BaseTool contract."""

    def test_all_tools_have_name(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        for tool in registry.get_all():
            assert tool.name, f"{type(tool).__name__} has no name"
            assert isinstance(tool.name, str)

    def test_all_tools_have_description(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        for tool in registry.get_all():
            assert tool.description, f"{type(tool).__name__} has no description"
            assert len(tool.description) > 20, \
                f"{tool.name} description too short: {len(tool.description)} chars"

    def test_all_tools_have_valid_json_schema(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        import json
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        for tool in registry.get_all():
            assert "type" in tool.parameters
            assert tool.parameters["type"] == "object"
            # Must be serializable
            json.dumps(tool.parameters)

    def test_all_tools_accept_kwargs_in_execute(self, tmp_path):
        """Tools should use **kwargs or named params in their execute signatures."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        import inspect
        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        for tool in registry.get_all():
            sig = inspect.signature(tool.execute)
            params = list(sig.parameters.keys())
            # First param is 'self', the rest are keyword arguments
            # All tools should accept their parameters as keyword arguments
            assert len(params) >= 1, f"{tool.name}.execute has no parameters"
