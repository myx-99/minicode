"""Unit tests for V3 three-mode routing and signal parsing.

Tests verify:
  - parse_agent_status correctly parses AGENT_STATUS blocks
  - route_after_execute decisions for ask/agent/plan modes
  - Ask/Agent mode graphs exclude plan/reflect nodes
  - Plan mode graph includes plan/reflect nodes
  - task_complete signal → finish (all modes)
  - replan signal → replan_node (all modes)
  - No signal defaults: plan→reflect, ask/agent→execute
  - normalize_mode maps react→agent, unknown→agent
  - init_node respects mode: plan→planning, ask/agent→executing
  - ToolRegistry.create_for_mode: ask=3 tools, agent/plan=6 tools

Run with:  pytest tests/test_dual_mode.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage


# ═══════════════════════════════════════════════════════════════════
# Test: parse_agent_status
# ═══════════════════════════════════════════════════════════════════

class TestParseAgentStatus:
    """Verify AGENT_STATUS block parsing."""

    def test_parse_task_complete(self):
        from graph.routing import parse_agent_status
        content = (
            "I have fixed the bug.\n\n"
            "---AGENT_STATUS---\n"
            '{"action": "task_complete", "reason": "Fixed import error"}\n'
            "---END_STATUS---"
        )
        status = parse_agent_status(content)
        assert status is not None
        assert status.action == "task_complete"
        assert status.reason == "Fixed import error"

    def test_parse_replan(self):
        from graph.routing import parse_agent_status
        content = (
            "Need to change approach.\n"
            "---AGENT_STATUS---\n"
            '{"action": "replan", "reason": "Wrong file structure"}'
            "\n---END_STATUS---"
        )
        status = parse_agent_status(content)
        assert status is not None
        assert status.action == "replan"

    def test_parse_continue(self):
        from graph.routing import parse_agent_status
        content = (
            "---AGENT_STATUS---\n"
            '{"action": "continue", "reason": "Still investigating"}\n'
            "---END_STATUS---\n"
            "More text after..."
        )
        status = parse_agent_status(content)
        assert status is not None
        assert status.action == "continue"

    def test_parse_step_done(self):
        from graph.routing import parse_agent_status
        content = (
            "Step completed.\n"
            "---AGENT_STATUS---\n"
            '{"action": "step_done", "reason": "Files searched"}\n'
            "---END_STATUS---"
        )
        status = parse_agent_status(content)
        assert status is not None
        assert status.action == "step_done"

    def test_no_status_block_returns_none(self):
        from graph.routing import parse_agent_status
        content = "I have completed the task successfully."
        status = parse_agent_status(content)
        assert status is None

    def test_empty_content_returns_none(self):
        from graph.routing import parse_agent_status
        assert parse_agent_status("") is None
        assert parse_agent_status(None) is None

    def test_invalid_json_in_block_returns_none(self):
        from graph.routing import parse_agent_status
        content = (
            "---AGENT_STATUS---\n"
            "{not valid json}\n"
            "---END_STATUS---"
        )
        status = parse_agent_status(content)
        assert status is None

    def test_invalid_action_returns_none(self):
        from graph.routing import parse_agent_status
        content = (
            "---AGENT_STATUS---\n"
            '{"action": "unknown_action", "reason": "test"}\n'
            "---END_STATUS---"
        )
        status = parse_agent_status(content)
        assert status is None


# ═══════════════════════════════════════════════════════════════════
# Test: route_after_execute (V3)
# ═══════════════════════════════════════════════════════════════════

class TestRouteAfterExecute:
    """Verify V3 routing decisions after execute_node."""

    def build_state(self, messages, mode="agent", iteration=0, max_iterations=30):
        """Build a minimal AgentState for routing tests."""
        return {
            "messages": messages,
            "mode": mode,
            "iteration": iteration,
            "max_iterations": max_iterations,
        }

    def test_tool_calls_go_to_tools(self):
        """LLM with tool_calls → always "tools" regardless of mode."""
        from graph.routing import route_after_execute
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"file_path": "main.py"}, "id": "call_1", "type": "tool_call"}
            ],
        )
        state = self.build_state([msg])
        assert route_after_execute(state) == "tools"

    def test_task_complete_signal_goes_to_finish_agent_mode(self):
        """task_complete signal → finish in agent mode."""
        from graph.routing import route_after_execute
        content = (
            "All done.\n"
            "---AGENT_STATUS---\n"
            '{"action": "task_complete", "reason": "complete"}\n'
            "---END_STATUS---"
        )
        msg = AIMessage(content=content)
        state = self.build_state([msg], mode="agent")
        assert route_after_execute(state) == "finish"

    def test_task_complete_signal_goes_to_finish_plan_mode(self):
        """task_complete signal → finish in plan mode."""
        from graph.routing import route_after_execute
        content = (
            "All done.\n"
            "---AGENT_STATUS---\n"
            '{"action": "task_complete", "reason": "complete"}\n'
            "---END_STATUS---"
        )
        msg = AIMessage(content=content)
        state = self.build_state([msg], mode="plan")
        assert route_after_execute(state) == "finish"

    def test_replan_signal_goes_to_replan(self):
        """replan signal → replan in any mode."""
        from graph.routing import route_after_execute
        content = (
            "---AGENT_STATUS---\n"
            '{"action": "replan", "reason": "wrong approach"}\n'
            "---END_STATUS---"
        )
        msg = AIMessage(content=content)
        state = self.build_state([msg], mode="agent")
        assert route_after_execute(state) == "replan"

    def test_no_signal_plan_mode_goes_to_reflect(self):
        """No signal + plan mode → reflect (V1 behavior)."""
        from graph.routing import route_after_execute
        msg = AIMessage(content="Step completed successfully.")
        state = self.build_state([msg], mode="plan")
        assert route_after_execute(state) == "reflect"

    def test_no_signal_agent_mode_goes_to_execute(self):
        """No signal + agent mode → execute (free loop)."""
        from graph.routing import route_after_execute
        msg = AIMessage(content="Let me think about what to do next.")
        state = self.build_state([msg], mode="agent")
        assert route_after_execute(state) == "execute"

    def test_no_signal_ask_mode_goes_to_execute(self):
        """No signal + ask mode → execute (free loop)."""
        from graph.routing import route_after_execute
        msg = AIMessage(content="This is a direct answer to your question.")
        state = self.build_state([msg], mode="ask")
        assert route_after_execute(state) == "execute"

    def test_max_iterations_guard_plan_mode(self):
        """Plan mode: max iterations → reflect (guard lives in reflect_node)."""
        from graph.routing import route_after_execute
        msg = AIMessage(content="Working...")
        state = self.build_state([msg], mode="plan", iteration=30, max_iterations=30)
        assert route_after_execute(state) == "reflect"

    def test_max_iterations_guard_agent_mode(self):
        """Agent mode: max iterations reached → finish."""
        from graph.routing import route_after_execute
        msg = AIMessage(content="Still working...")
        state = self.build_state([msg], mode="agent", iteration=30, max_iterations=30)
        assert route_after_execute(state) == "finish"

    def test_empty_messages_plan_defaults_to_reflect(self):
        """Empty messages → reflect in plan mode."""
        from graph.routing import route_after_execute
        state = self.build_state([], mode="plan")
        assert route_after_execute(state) == "reflect"

    def test_step_done_signal_plan_mode_goes_to_reflect(self):
        """step_done signal in plan mode → reflect (for step evaluation)."""
        from graph.routing import route_after_execute
        content = (
            "---AGENT_STATUS---\n"
            '{"action": "step_done", "reason": "step finished"}\n'
            "---END_STATUS---"
        )
        msg = AIMessage(content=content)
        state = self.build_state([msg], mode="plan")
        assert route_after_execute(state) == "reflect"

    def test_continue_signal_agent_mode_goes_to_execute(self):
        """continue signal in agent mode → execute (keep looping)."""
        from graph.routing import route_after_execute
        content = (
            "---AGENT_STATUS---\n"
            '{"action": "continue", "reason": "still working"}\n'
            "---END_STATUS---"
        )
        msg = AIMessage(content=content)
        state = self.build_state([msg], mode="agent")
        assert route_after_execute(state) == "execute"


# ═══════════════════════════════════════════════════════════════════
# Test: route_after_reflect (V1 behavior, unchanged)
# ═══════════════════════════════════════════════════════════════════

class TestRouteAfterReflect:
    """Verify V1 reflect routing is preserved."""

    def test_executing_goes_to_execute(self):
        from graph.routing import route_after_reflect
        state = {"phase": "executing"}
        assert route_after_reflect(state) == "execute"

    def test_retry_goes_to_execute(self):
        from graph.routing import route_after_reflect
        state = {"phase": "retry"}
        assert route_after_reflect(state) == "execute"

    def test_replan_goes_to_replan(self):
        from graph.routing import route_after_reflect
        state = {"phase": "replan"}
        assert route_after_reflect(state) == "replan"

    def test_done_goes_to_finish(self):
        from graph.routing import route_after_reflect
        state = {"phase": "done"}
        assert route_after_reflect(state) == "finish"

    def test_error_goes_to_finish(self):
        from graph.routing import route_after_reflect
        state = {"phase": "error"}
        assert route_after_reflect(state) == "finish"


# ═══════════════════════════════════════════════════════════════════
# Test: normalize_mode (V3)
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeMode:
    """Verify mode normalization for V3."""

    def test_ask_passes_through(self):
        from graph.routing import normalize_mode
        assert normalize_mode("ask") == "ask"

    def test_agent_passes_through(self):
        from graph.routing import normalize_mode
        assert normalize_mode("agent") == "agent"

    def test_plan_passes_through(self):
        from graph.routing import normalize_mode
        assert normalize_mode("plan") == "plan"

    def test_react_maps_to_agent(self):
        from graph.routing import normalize_mode
        assert normalize_mode("react") == "agent"

    def test_unknown_defaults_to_agent(self):
        from graph.routing import normalize_mode
        assert normalize_mode("unknown") == "agent"


# ═══════════════════════════════════════════════════════════════════
# Test: Three-mode graph construction (V3)
# ═══════════════════════════════════════════════════════════════════

class TestThreeModeGraph:
    """Verify graph structure differs by mode."""

    def _build_graph(self, tmp_path, mode):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_for_mode(ws, mode)

        class MockLLM:
            _bound_tools = None
            async def ainvoke(self, messages, **kwargs):
                return AIMessage(content="ok")
            def bind_tools(self, schemas):
                self._bound_tools = schemas
                return self

        llm = MockLLM()
        return build_graph(llm, registry, mode=mode)

    def test_agent_mode_graph_no_plan_no_reflect(self, tmp_path):
        """Agent mode graph excludes plan and reflect nodes."""
        graph = self._build_graph(tmp_path, "agent")
        node_names = list(graph.get_graph().nodes.keys())
        assert "init" in node_names
        assert "execute" in node_names
        assert "tools" in node_names
        assert "replan" in node_names
        assert "finish" in node_names
        assert "plan" not in node_names
        assert "reflect" not in node_names

    def test_ask_mode_graph_no_plan_no_reflect(self, tmp_path):
        """Ask mode graph excludes plan and reflect nodes (same structure as agent)."""
        graph = self._build_graph(tmp_path, "ask")
        node_names = list(graph.get_graph().nodes.keys())
        assert "init" in node_names
        assert "execute" in node_names
        assert "tools" in node_names
        assert "replan" in node_names
        assert "finish" in node_names
        assert "plan" not in node_names
        assert "reflect" not in node_names

    def test_plan_mode_graph_has_plan_and_reflect(self, tmp_path):
        """Plan mode graph includes plan and reflect nodes."""
        graph = self._build_graph(tmp_path, "plan")
        node_names = list(graph.get_graph().nodes.keys())
        assert "init" in node_names
        assert "plan" in node_names
        assert "execute" in node_names
        assert "tools" in node_names
        assert "reflect" in node_names
        assert "replan" in node_names
        assert "finish" in node_names

    def test_react_mode_is_agent_alias(self, tmp_path):
        """React mode graph = agent mode graph (deprecated alias)."""
        graph = self._build_graph(tmp_path, "react")
        node_names = list(graph.get_graph().nodes.keys())
        assert "plan" not in node_names
        assert "reflect" not in node_names


# ═══════════════════════════════════════════════════════════════════
# Test: init_node mode-aware behavior (V3)
# ═══════════════════════════════════════════════════════════════════

class TestInitNodeMode:
    """Verify init_node respects mode field (V3 — no intent_class)."""

    @pytest.mark.asyncio
    async def test_init_node_plan_mode(self):
        from graph.nodes import init_node
        state = {"task": "test task", "mode": "plan"}
        result = await init_node(state)
        assert result["phase"] == "planning"
        assert result["plan"] == []
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_init_node_agent_mode(self):
        from graph.nodes import init_node
        state = {"task": "test task", "mode": "agent"}
        result = await init_node(state)
        assert result["phase"] == "executing"
        assert result["plan"] == []
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_init_node_ask_mode(self):
        from graph.nodes import init_node
        state = {"task": "test task", "mode": "ask"}
        result = await init_node(state)
        assert result["phase"] == "executing"

    @pytest.mark.asyncio
    async def test_init_node_default_mode(self):
        """Default mode (no mode field) = agent (V3 default)."""
        from graph.nodes import init_node
        state = {"task": "test task"}
        result = await init_node(state)
        assert result["phase"] == "executing"  # V3: default is agent


# ═══════════════════════════════════════════════════════════════════
# Test: Mode-aware Tool Registry (V3)
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistryMode:
    """Verify create_for_mode produces the right tool sets."""

    def test_ask_mode_has_three_read_only_tools(self):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(".")
        registry = ToolRegistry.create_for_mode(ws, "ask")
        names = registry.tool_names
        assert "read_file" in names
        assert "grep_search" in names
        assert "glob_search" in names
        assert "write_file" not in names
        assert "edit_file" not in names
        assert "shell_execute" not in names
        assert registry.count == 3

    def test_agent_mode_has_all_six_tools(self):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(".")
        registry = ToolRegistry.create_for_mode(ws, "agent")
        assert registry.count == 6
        assert "shell_execute" in registry.tool_names

    def test_plan_mode_has_all_six_tools(self):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(".")
        registry = ToolRegistry.create_for_mode(ws, "plan")
        assert registry.count == 6

    def test_react_alias_has_all_six_tools(self):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(".")
        registry = ToolRegistry.create_for_mode(ws, "react")
        assert registry.count == 6  # react → agent

    def test_create_default_is_agent_equivalent(self):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        ws = Workspace(".")
        default_reg = ToolRegistry.create_default(ws)
        agent_reg = ToolRegistry.create_for_mode(ws, "agent")
        assert default_reg.tool_names == agent_reg.tool_names


# ═══════════════════════════════════════════════════════════════════
# Test: is_conversational_query / is_direct_answer_query (retained, memory-only)
# ═══════════════════════════════════════════════════════════════════

class TestIsConversationalQuery:
    """Verify conversational query detection is retained for memory use (not routing)."""

    def test_identity_questions_are_conversational(self):
        from memory.project import is_conversational_query
        assert is_conversational_query("你是谁") is True
        assert is_conversational_query("who are you") is True
        assert is_conversational_query("谢谢") is True
        assert is_conversational_query("hello") is True

    def test_coding_tasks_are_not_conversational(self):
        from memory.project import is_conversational_query
        assert is_conversational_query("读取 main.py") is False
        assert is_conversational_query("修复 import 错误") is False
        assert is_conversational_query("fix the bug") is False

    def test_boundary_cases_not_conversational(self):
        """Queries mentioning code → NOT conversational."""
        from memory.project import is_conversational_query
        assert is_conversational_query("你是谁写的代码") is False
        assert is_conversational_query("who wrote this code") is False

    def test_direct_answer_detects_general_knowledge(self):
        from memory.project import is_direct_answer_query
        assert is_direct_answer_query("中国首都是哪里") is True
        assert is_direct_answer_query("你是谁") is True
        assert is_direct_answer_query("修复bug") is False
