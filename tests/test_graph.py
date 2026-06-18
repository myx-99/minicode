"""Unit tests for Phase 2 — LangGraph + Agent Loop.

Tests verify graph construction, node behavior, routing logic,
and end-to-end flow with a mock LLM.

Run with:  pytest tests/test_graph.py -v
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ═══════════════════════════════════════════════════════════════════
# Mock LLM — returns predetermined responses for testing
# ═══════════════════════════════════════════════════════════════════

class MockLLM:
    """A fake LLM that returns canned AIMessage responses.

    Supports:
      - .ainvoke(messages) → returns next response in the cycle
      - .bind_tools(schemas) → returns self (tracks what was bound)
    """

    def __init__(self, responses=None):
        """Args:
            responses: List of AIMessage or str to return on successive calls.
        """
        self._responses = responses or []
        self._call_count = 0
        self._bound_tools = None
        self._last_messages = None

    async def ainvoke(self, messages, **kwargs):
        self._last_messages = messages
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, str):
                return AIMessage(content=resp)
            return resp
        # Default: return empty text
        return AIMessage(content="Done.")

    def bind_tools(self, tool_schemas):
        self._bound_tools = tool_schemas
        return self

    def invoke(self, messages, **kwargs):
        # Sync fallback — not used in async tests
        return AIMessage(content="sync fallback")


def make_tool_call_ai(tool_calls):
    """Create an AIMessage that contains tool_calls (triggers routing to tool_node)."""
    # Replicate LangChain's AIMessage format with tool_calls
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tc[0],
                "args": tc[1],
                "id": tc[2] if len(tc) > 2 else f"call_test_{tc[0]}",
                "type": "tool_call",
            }
            for tc in tool_calls
        ],
    )


def make_text_ai(text):
    """Create a plain AIMessage (triggers routing to reflect_node)."""
    return AIMessage(content=text)


# Pre-canned Intent Auditor response: step is aligned with goal (entailment)
_AUDIT_ENTAILMENT = json.dumps({
    "label": "entailment",
    "score": 0.95,
    "reason": "The step directly serves the user's goal.",
})


# ═══════════════════════════════════════════════════════════════════
# Test: Graph Construction
# ═══════════════════════════════════════════════════════════════════

class TestGraphConstruction:
    """Verify the graph can be built and compiled."""

    def test_build_graph_compiles(self, tmp_path):
        """Graph builds without error and is compilable."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        llm = MockLLM()
        llm.bind_tools(registry.get_openai_schemas())

        from graph.builder import build_graph
        graph = build_graph(llm, registry)

        # Graph should be compiled and have standard methods
        assert graph is not None
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")

    def test_graph_has_all_nodes(self, tmp_path):
        """Agent mode (default) graph has 5 nodes: init, execute, tools, replan, finish."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        llm = MockLLM()

        graph = build_graph(llm, registry, mode="agent")

        # Check node names via the graph's internal structure
        node_names = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
        if node_names:
            expected = {"init", "execute", "tools", "replan", "finish"}
            assert set(node_names) & expected == expected
            # Agent mode: no plan, no reflect
            assert "plan" not in set(node_names)
            assert "reflect" not in set(node_names)


# ═══════════════════════════════════════════════════════════════════
# Test: Routing Functions
# ═══════════════════════════════════════════════════════════════════

class TestRouting:
    """Test the conditional routing logic."""

    def test_route_execute_with_tool_calls(self):
        """When last message has tool_calls → route to 'tools'."""
        from graph.routing import route_after_execute

        state = {
            "messages": [
                HumanMessage(content="fix the bug"),
                make_tool_call_ai([("read_file", {"file_path": "main.py"}, "call_1")]),
            ],
        }
        assert route_after_execute(state) == "tools"

    def test_route_execute_without_tool_calls(self):
        """When last message is plain text in plan mode → route to 'reflect'."""
        from graph.routing import route_after_execute

        state = {
            "messages": [
                HumanMessage(content="fix the bug"),
                AIMessage(content="I have completed the fix."),
            ],
            "mode": "plan",
        }
        assert route_after_execute(state) == "reflect"

    def test_route_execute_empty_messages(self):
        """Empty messages → reflect (defensive)."""
        from graph.routing import route_after_execute

        state = {"messages": [], "mode": "plan"}
        assert route_after_execute(state) == "reflect"

    def test_route_reflect_executing(self):
        """phase='executing' → back to execute."""
        from graph.routing import route_after_reflect

        state = {"phase": "executing"}
        assert route_after_reflect(state) == "execute"

    def test_route_reflect_done(self):
        """phase='done' → finish."""
        from graph.routing import route_after_reflect

        state = {"phase": "done"}
        assert route_after_reflect(state) == "finish"

    def test_route_reflect_error(self):
        """phase='error' → finish (graceful exit)."""
        from graph.routing import route_after_reflect

        state = {"phase": "error"}
        assert route_after_reflect(state) == "finish"


# ═══════════════════════════════════════════════════════════════════
# Test: Init Node
# ═══════════════════════════════════════════════════════════════════

class TestInitNode:
    """Tests for init_node."""

    @pytest.mark.asyncio
    async def test_init_sets_initial_state(self):
        from graph.nodes import init_node

        state = {
            "task": "test task",
            "messages": [],
            "plan": [],
            "mode": "plan",
        }
        result = await init_node(state)

        assert result["phase"] == "planning"
        assert result["iteration"] == 0
        assert result["current_step_index"] == 0
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], SystemMessage)
        assert isinstance(result["messages"][1], HumanMessage)
        assert result["messages"][1].content == "test task"
        assert result["plan"] == []
        assert result["tool_history"] == []


# ═══════════════════════════════════════════════════════════════════
# Test: Plan Node
# ═══════════════════════════════════════════════════════════════════

class TestPlanNode:
    """Tests for plan_node."""

    @pytest.mark.asyncio
    async def test_plan_node_parses_json(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"id": "1", "description": "Search for files"},
            {"id": "2", "description": "Read main.py"},
            {"id": "3", "description": "Verify"},
        ])
        llm = MockLLM(responses=[plan_json])

        state = {
            "task": "Add logging to the project",
            "workspace_path": ".",
            "messages": [],
        }
        result = await plan_node(state, llm)

        assert result["phase"] == "executing"
        assert len(result["plan"]) == 3
        assert result["plan"][0]["description"] == "Search for files"
        assert result["plan"][0]["status"] == "pending"
        assert result["current_step_index"] == 0

    @pytest.mark.asyncio
    async def test_plan_node_strips_markdown_fences(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"id": "1", "description": "Step one"},
        ])
        llm = MockLLM(responses=[f"```json\n{plan_json}\n```"])

        state = {"task": "test", "messages": []}
        result = await plan_node(state, llm)

        assert len(result["plan"]) == 1
        assert result["plan"][0]["description"] == "Step one"

    @pytest.mark.asyncio
    async def test_plan_node_fallback_on_invalid_json(self):
        from graph.nodes import plan_node

        llm = MockLLM(responses=["not valid json at all"])

        state = {"task": "fix the bug", "messages": []}
        result = await plan_node(state, llm)

        # Falls back to single-step plan
        assert len(result["plan"]) == 1
        assert result["plan"][0]["description"] == "fix the bug"
        assert result["phase"] == "executing"
        assert "Plan parsing fell back" in result.get("error_message", "")

    @pytest.mark.asyncio
    async def test_plan_node_adds_ids_if_missing(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"description": "Step A"},
            {"description": "Step B"},
        ])
        llm = MockLLM(responses=[plan_json])

        state = {"task": "test", "messages": []}
        result = await plan_node(state, llm)

        assert result["plan"][0]["id"] == "1"
        assert result["plan"][1]["id"] == "2"


# ═══════════════════════════════════════════════════════════════════
# Test: Execute Node
# ═══════════════════════════════════════════════════════════════════

class TestExecuteNode:
    """Tests for the execute node factory."""

    @pytest.mark.asyncio
    async def test_execute_increments_iteration(self):
        from graph.nodes import create_execute_node

        llm = MockLLM(responses=["I'll now read the file."])
        node = create_execute_node(llm)

        state = {
            "task": "read main.py",
            "messages": [
                SystemMessage(content="You are a coding agent."),
                HumanMessage(content="read main.py"),
            ],
            "plan": [{"id": "1", "description": "Read file", "status": "pending"}],
            "current_step_index": 0,
            "iteration": 0,
        }
        result = await node(state)

        assert result["iteration"] == 1
        assert len(result["messages"]) == 2  # context + AI response

    @pytest.mark.asyncio
    async def test_execute_injects_step_context(self):
        from graph.nodes import create_execute_node

        llm = MockLLM(responses=["Done."])
        node = create_execute_node(llm)

        state = {
            "task": "add tests",
            "mode": "plan",
            "messages": [
                SystemMessage(content="Be helpful."),
                HumanMessage(content="add tests"),
            ],
            "plan": [
                {"id": "1", "description": "Find files", "status": "pending"},
                {"id": "2", "description": "Add tests", "status": "pending"},
            ],
            "current_step_index": 1,
            "iteration": 3,
        }
        result = await node(state)

        # messages[0] should be the context message about step 2
        context_msg = result["messages"][0]
        assert isinstance(context_msg, HumanMessage)
        assert "Add tests" in context_msg.content
        assert "Step 2" in context_msg.content or "2/2" in context_msg.content

    @pytest.mark.asyncio
    async def test_execute_with_tool_calls(self):
        from graph.nodes import create_execute_node

        llm = MockLLM(responses=[
            make_tool_call_ai([("read_file", {"file_path": "app.py"}, "c1")])
        ])
        node = create_execute_node(llm)

        state = {
            "task": "check app.py",
            "messages": [HumanMessage(content="check app.py")],
            "plan": [],
            "current_step_index": 0,
            "iteration": 0,
        }
        result = await node(state)

        ai_msg = result["messages"][-1]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.tool_calls
        assert ai_msg.tool_calls[0]["name"] == "read_file"


# ═══════════════════════════════════════════════════════════════════
# Test: Tool Node
# ═══════════════════════════════════════════════════════════════════

class TestToolNode:
    """Tests for the tool execution node."""

    @pytest.mark.asyncio
    async def test_tool_node_executes_registered_tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.nodes import create_tool_node

        # Create a real workspace with a real file
        (tmp_path / "hello.txt").write_text("Hello World")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        node = create_tool_node(registry)

        state = {
            "messages": [
                HumanMessage(content="read hello.txt"),
                make_tool_call_ai([("read_file", {"file_path": "hello.txt"}, "c1")]),
            ],
            "tool_history": [],
        }
        result = await node(state)

        # Should have ToolMessage(s)
        assert len(result["messages"]) >= 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert "Hello World" in result["messages"][0].content

        # Should have tool history record
        assert len(result["tool_history"]) == 1
        assert result["tool_history"][0]["tool"] == "read_file"
        assert result["tool_history"][0]["args"] == {"file_path": "hello.txt"}

    @pytest.mark.asyncio
    async def test_tool_node_handles_unknown_tool(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.nodes import create_tool_node

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry()  # empty — no tools registered
        node = create_tool_node(registry)

        state = {
            "messages": [
                HumanMessage(content="do something"),
                make_tool_call_ai([("nonexistent_tool", {}, "c_bad")]),
            ],
            "tool_history": [],
        }
        result = await node(state)

        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "Unknown tool" in msg.content or "Error" in msg.content

    @pytest.mark.asyncio
    async def test_tool_node_appends_to_existing_history(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.nodes import create_tool_node

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        node = create_tool_node(registry)

        state = {
            "messages": [
                make_tool_call_ai([
                    ("glob_search", {"pattern": "*.py"}, "c2"),
                ]),
            ],
            "tool_history": [
                {"tool": "read_file", "args": {}, "result": "old", "timestamp": ""}
            ],
        }
        result = await node(state)

        assert len(result["tool_history"]) == 2  # old + new

    @pytest.mark.asyncio
    async def test_tool_node_no_tool_calls_is_noop(self, tmp_path):
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.nodes import create_tool_node

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        node = create_tool_node(registry)

        state = {
            "messages": [
                AIMessage(content="No tool calls needed."),
            ],
            "tool_history": [],
        }
        result = await node(state)

        # Should return error_message since last message is not AIMessage with tool_calls
        # (The AIMessage(content=...) without tool_calls still has tool_calls=[])
        assert "no tool_calls" in result.get("error_message", "")


# ═══════════════════════════════════════════════════════════════════
# Test: Reflect Node (Phase 3 — now LLM-powered via create_reflect_node)
# ═══════════════════════════════════════════════════════════════════

# Reusable "step done successfully" reflection JSON
_SUCCESS_REFLECTION = json.dumps({
    "step_done": True, "success": True, "error_type": "none",
    "reasoning": "Step completed.", "should_retry": False,
    "should_replan": False, "retry_suggestion": "",
})


def _make_reflect_state(**overrides):
    """Create a minimal state for reflect_node testing."""
    state = {
        "plan": [
            {"id": "1", "description": "Step 1", "status": "in_progress",
             "retry_count": 0, "max_retries": 2},
            {"id": "2", "description": "Step 2", "status": "pending",
             "retry_count": 0, "max_retries": 2},
        ],
        "current_step_index": 0,
        "iteration": 5,
        "max_iterations": 30,
        "step_retry_count": 0,
        "max_retries_per_step": 2,
        "messages": [
            HumanMessage(content="do task"),
            AIMessage(content="Step 1 is complete."),
        ],
        "tool_history": [],
        "phase": "executing",
    }
    state.update(overrides)
    return state


class TestReflectNode:
    """Tests for the LLM-powered reflection / step-advancing node."""

    @pytest.mark.asyncio
    async def test_reflect_advances_to_next_step(self):
        from graph.nodes import create_reflect_node

        llm = MockLLM(responses=[_SUCCESS_REFLECTION])
        reflect_node = create_reflect_node(llm)

        state = _make_reflect_state()
        result = await reflect_node(state)

        assert result["phase"] == "executing"
        assert result["current_step_index"] == 1
        assert result["plan"][0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_reflect_all_steps_done(self):
        from graph.nodes import create_reflect_node

        llm = MockLLM(responses=[_SUCCESS_REFLECTION])
        reflect_node = create_reflect_node(llm)

        state = _make_reflect_state(
            plan=[
                {"id": "1", "description": "Only step", "status": "in_progress",
                 "retry_count": 0, "max_retries": 2},
            ],
            current_step_index=0,
        )
        result = await reflect_node(state)

        assert result["phase"] == "done"
        assert result["plan"][0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_reflect_max_iterations_guard(self):
        from graph.nodes import create_reflect_node

        # LLM won't be called — guard fires first
        llm = MockLLM()
        reflect_node = create_reflect_node(llm)

        state = _make_reflect_state(iteration=30, max_iterations=30)
        result = await reflect_node(state)

        assert result["phase"] == "done"
        assert "max iterations" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_reflect_empty_plan(self):
        from graph.nodes import create_reflect_node

        llm = MockLLM()
        reflect_node = create_reflect_node(llm)

        state = _make_reflect_state(plan=[], current_step_index=0)
        result = await reflect_node(state)

        assert result["phase"] == "done"


# ═══════════════════════════════════════════════════════════════════
# Test: Finish Node
# ═══════════════════════════════════════════════════════════════════

class TestFinishNode:
    """Tests for the final summary node."""

    @pytest.mark.asyncio
    async def test_finish_node_produces_summary(self):
        from graph.nodes import finish_node

        llm = MockLLM(responses=["## Summary\n\nAll done! Files were modified."])

        state = {
            "task": "Add docstrings",
            "messages": [
                HumanMessage(content="Add docstrings"),
                AIMessage(content="I added docstrings to all functions."),
            ],
            "plan": [
                {"id": "1", "description": "Find functions", "status": "done"},
                {"id": "2", "description": "Add docstrings", "status": "done"},
            ],
            "tool_history": [
                {
                    "tool": "edit_file",
                    "args": {"file_path": "main.py"},
                    "result": "Added docstring",
                    "timestamp": "",
                },
            ],
            "phase": "done",
        }
        result = await finish_node(state, llm)

        assert result["phase"] == "done"
        assert len(result["final_answer"]) > 0

    @pytest.mark.asyncio
    async def test_finish_node_fallback_on_llm_error(self):
        from graph.nodes import finish_node

        # LLM that raises
        class BrokenLLM:
            async def ainvoke(self, messages, **kwargs):
                raise RuntimeError("API down")

        llm = BrokenLLM()

        state = {
            "task": "test",
            "messages": [HumanMessage(content="test")],
            "plan": [],
            "tool_history": [],
            "phase": "done",
        }
        result = await finish_node(state, llm)

        # Should fall back gracefully
        assert result["phase"] == "done"
        assert len(result["final_answer"]) > 0  # fallback message

    @pytest.mark.asyncio
    async def test_finish_node_conversational_pass_through(self):
        """Conversational answers pass through finish_node without LLM re-summarization."""
        from graph.nodes import finish_node

        # LLM should NOT be called for conversational intent
        class NoCallLLM:
            async def ainvoke(self, messages, **kwargs):
                raise RuntimeError("LLM should not be called for conversational pass-through")

        llm = NoCallLLM()

        state = {
            "task": "你是谁",
            "intent_class": "conversational",
            "messages": [
                HumanMessage(content="你是谁"),
                AIMessage(content="我是 Claude Code Mini，一个编码助手。"),
            ],
            "plan": [],
            "tool_history": [],
            "phase": "done",
        }
        result = await finish_node(state, llm)

        assert result["phase"] == "done"
        assert "Claude Code Mini" in result["final_answer"]
        # No structured summary header — it's the direct answer
        assert "## Summary" not in result["final_answer"]

    @pytest.mark.asyncio
    async def test_finish_node_zero_tool_empty_plan_pass_through(self):
        """General-knowledge Q&A (React): no tools, no plan → direct answer, no summary LLM."""
        from graph.nodes import finish_node

        class NoCallLLM:
            async def ainvoke(self, messages, **kwargs):
                raise RuntimeError("LLM should not be called for zero-tool pass-through")

        llm = NoCallLLM()
        state = {
            "task": "1+1等于多少",
            "intent_class": "coding",
            "messages": [
                HumanMessage(content="1+1等于多少"),
                AIMessage(content="1+1 等于 2。"),
            ],
            "plan": [],
            "tool_history": [],
            "phase": "done",
        }
        result = await finish_node(state, llm)

        assert result["phase"] == "done"
        assert "2" in result["final_answer"]
        assert "做了什么" not in result["final_answer"]


# ═══════════════════════════════════════════════════════════════════
# Test: is_direct_answer_query (general knowledge)
# ═══════════════════════════════════════════════════════════════════

class TestIsDirectAnswerQuery:
    def test_general_knowledge_questions(self):
        from memory.project import is_direct_answer_query
        assert is_direct_answer_query("中国首都是哪里") is True
        assert is_direct_answer_query("1+1等于多少") is True
        assert is_direct_answer_query("what is the capital of France?") is True

    def test_coding_tasks_not_direct_answer(self):
        from memory.project import is_direct_answer_query
        assert is_direct_answer_query("读取 main.py") is False
        assert is_direct_answer_query("Tell me what main.py does") is False


# ═══════════════════════════════════════════════════════════════════
# Test: End-to-End Graph Execution
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Integration tests running the full graph with mock LLM."""

    @pytest.mark.asyncio
    async def test_simple_task_flow(self, tmp_path):
        """Simulate a simple task: "read main.py and tell me what you see".

        Phase 3 response sequence includes LLM-powered reflection.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        (tmp_path / "main.py").write_text("print('hello')")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read main.py"},
        ])

        llm = MockLLM(responses=[
            plan_json,                                               # plan
            _AUDIT_ENTAILMENT,                                       # audit_plan (1 step)
            make_tool_call_ai([("read_file", {"file_path": "main.py"}, "c1")]),  # execute
            AIMessage(content="The file contains a print statement."),           # execute done
            _SUCCESS_REFLECTION,                                     # reflect
            "## Summary\n\nRead main.py. It prints hello.",          # finish
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "Tell me what main.py does",
            "mode": "plan",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        # Should have completed
        assert final["phase"] == "done"
        assert len(final["final_answer"]) > 0
        # Plan should have 1 step marked done
        assert len(final["plan"]) == 1
        assert final["plan"][0]["status"] == "done"
        # Tool should have been called
        assert len(final["tool_history"]) >= 1

    @pytest.mark.asyncio
    async def test_multistep_task_flow(self, tmp_path):
        """Run a 2-step task: find Python files, then read one.

        Phase 3: includes LLM reflection after each step.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Find all Python files"},
            {"id": "2", "description": "Read one of them"},
        ])

        llm = MockLLM(responses=[
            plan_json,                                                                    # plan
            _AUDIT_ENTAILMENT,                                                            # audit step 1
            _AUDIT_ENTAILMENT,                                                            # audit step 2
            make_tool_call_ai([("glob_search", {"pattern": "*.py"}, "c1")]),             # exec step1
            AIMessage(content="Found 2 Python files."),                                    # exec step1 done
            _SUCCESS_REFLECTION,                                                           # reflect step1
            make_tool_call_ai([("read_file", {"file_path": "a.py"}, "c2")]),             # exec step2
            AIMessage(content="a.py contains x=1."),                                       # exec step2 done
            _SUCCESS_REFLECTION,                                                           # reflect step2
            "## Summary\n\nFound and read Python files.",                                  # finish
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "Find and read Python files",
            "mode": "plan",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        assert final["phase"] == "done"
        assert len(final["plan"]) == 2
        assert final["plan"][0]["status"] == "done"
        assert final["plan"][1]["status"] == "done"
        assert len(final["tool_history"]) >= 2

    @pytest.mark.asyncio
    async def test_plan_fallback_still_completes(self, tmp_path):
        """Even when JSON parsing fails, the agent should complete with a fallback plan."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        llm = MockLLM(responses=[
            "not json at all!!!",                                               # plan (bad → fallback)
            make_tool_call_ai([("glob_search", {"pattern": "*.py"}, "cx")]),   # execute
            AIMessage(content="Task attempted."),                               # execute done
            _SUCCESS_REFLECTION,                                                # reflect
            "Summary: task was attempted.",                                     # finish
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "do something",
            "mode": "plan",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        # Should still complete with fallback
        assert final["phase"] == "done"

    @pytest.mark.asyncio
    async def test_max_iterations_halts(self, tmp_path):
        """If the agent loops too much, the iteration guard should halt it.

        Uses agent mode (not plan) because the max_iterations guard in
        route_after_execute fires directly without needing the reflect node path.
        In plan mode, the guard lives in reflect_node which requires a
        no-tool-call response to reach — making the test more brittle.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        # Agent mode: init → execute → tools/finish/replan/execute
        # With no tool_calls and no AGENT_STATUS signal, execute self-loops.
        # When iteration >= max_iterations, route_after_execute returns "finish".
        responses = []
        for _ in range(10):
            # Plain text, no tool_calls → route_after_execute returns "execute" (self-loop)
            responses.append(AIMessage(content="Still working..."))

        llm = MockLLM(responses=responses)

        graph = build_graph(llm, registry, mode="agent")

        initial_state = {
            "task": "test",
            "mode": "agent",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 3,  # very low limit
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        # Should halt — phase is "done" and iteration reached the max
        assert final["phase"] == "done"
        assert final.get("iteration", 0) == 3
        # In agent mode, when max_iterations is hit, route_after_execute
        # returns "finish" directly without setting error_message.
        # The agent stops cleanly with whatever final_answer it has.


# ═══════════════════════════════════════════════════════════════════
# Test: ClaudeCodeMini Agent Class
# ═══════════════════════════════════════════════════════════════════

class TestAgentClass:
    """Tests for the ClaudeCodeMini main class."""

    def test_agent_instantiation(self, tmp_path):
        from agent.agent import ClaudeCodeMini
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)
        llm = MockLLM()

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=llm,
        )
        assert agent.workspace is not None
        assert agent.tool_registry is not None
        assert agent.tool_registry.count == 6
        assert agent.graph is not None

    def test_agent_repr(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        llm = MockLLM()
        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)
        r = repr(agent)
        assert "ClaudeCodeMini" in r

    @pytest.mark.asyncio
    async def test_agent_run_completes(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan_json = json.dumps([
            {"id": "1", "description": "Check project"},
        ])

        llm = MockLLM(responses=[
            plan_json,
            AIMessage(content="Project looks good."),
            _SUCCESS_REFLECTION,
            "All done.",
        ])

        agent = ClaudeCodeMini(
            workspace_path=str(tmp_path),
            llm=llm,
        )
        result = await agent.run("Check the project")

        assert result["success"] is True
        assert len(result["final_answer"]) > 0

    @pytest.mark.asyncio
    async def test_agent_run_empty_task_raises(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        llm = MockLLM()
        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)

        with pytest.raises(ValueError, match="empty"):
            await agent.run("")

    @pytest.mark.asyncio
    async def test_agent_stream_yields_events(self, tmp_path):
        from agent.agent import ClaudeCodeMini

        plan_json = json.dumps([
            {"id": "1", "description": "Check"},
        ])

        llm = MockLLM(responses=[
            plan_json,
            AIMessage(content="OK."),
            _SUCCESS_REFLECTION,
            "Summary",
        ])

        agent = ClaudeCodeMini(workspace_path=str(tmp_path), llm=llm)

        events = []
        async for event in agent.stream("Check project"):
            events.append(event)

        assert len(events) > 0
        final = events[-1]
        assert final["phase"] == "done"


# ═══════════════════════════════════════════════════════════════════
# Test: BUG-001 Conversational query short-circuit (Plan mode)
# ═══════════════════════════════════════════════════════════════════

class TestBug001ConversationalShortCircuit:
    """Verify V3 model-driven handling of simple Q&A across modes."""

    @pytest.mark.asyncio
    async def test_conversational_query_no_tool_calls_plan_mode(self, tmp_path):
        """Agent mode + '你是谁' → model answers with task_complete, 0 tools,
        finish_node pass-through without re-summarization."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        # V3: Model answers + task_complete signal (no conversational shortcut needed)
        llm = MockLLM(responses=[
            AIMessage(
                content=(
                    "我是 Claude Code Mini，一个编码助手。\n\n"
                    "---AGENT_STATUS---\n"
                    '{"action": "task_complete", "reason": "Answered"}\n'
                    "---END_STATUS---"
                )
            ),
        ])

        graph = build_graph(llm, registry, mode="agent")

        initial_state = {
            "task": "你是谁",
            "mode": "agent",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        assert final["phase"] == "done"
        assert len(final["final_answer"]) > 0
        assert "Claude Code Mini" in final["final_answer"]
        assert len(final.get("tool_history", [])) == 0

    @pytest.mark.asyncio
    async def test_general_knowledge_direct_answer_plan_mode(self, tmp_path):
        """Agent mode + '中国首都是哪里' → brief answer, 0 tools, passthrough finish."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        llm = MockLLM(responses=[
            AIMessage(
                content=(
                    "中国的首都是北京。\n\n"
                    "---AGENT_STATUS---\n"
                    '{"action": "task_complete", "reason": "Answered"}\n'
                    "---END_STATUS---"
                )
            ),
        ])

        graph = build_graph(llm, registry, mode="agent")

        initial_state = {
            "task": "中国首都是哪里",
            "mode": "agent",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        assert final["phase"] == "done"
        assert "北京" in final["final_answer"]
        assert len(final.get("tool_history", [])) == 0

    @pytest.mark.asyncio
    async def test_coding_task_still_uses_plan_mode(self, tmp_path):
        """Plan mode + "读取 main.py 并解释" → plan generated + read_file called."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        (tmp_path / "main.py").write_text("# Main entry point")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "读取 main.py 文件"},
            {"id": "2", "description": "解释文件内容"},
        ])

        llm = MockLLM(responses=[
            plan_json,                                                              # plan
            _AUDIT_ENTAILMENT,                                                      # audit step 1
            _AUDIT_ENTAILMENT,                                                      # audit step 2
            make_tool_call_ai([("read_file", {"file_path": "main.py"}, "c1")]),    # execute step1
            AIMessage(content="文件包含注释 '# Main entry point'。"),               # execute done
            _SUCCESS_REFLECTION,                                                    # reflect
            AIMessage(content="文件很简单。"),                                      # execute step2
            _SUCCESS_REFLECTION,                                                    # reflect
            "## Summary\n\nRead and explained main.py.",                            # finish
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "读取 main.py 并解释",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        # Should have completed normally with plan + tools
        assert final["phase"] == "done"
        assert len(final["plan"]) == 2
        assert final["plan"][0]["status"] == "done"
        assert len(final["tool_history"]) >= 1
        # Verify read_file was called
        tools_called = [t["tool"] for t in final["tool_history"]]
        assert "read_file" in tools_called

    @pytest.mark.asyncio
    async def test_conversational_query_react_mode_unchanged(self, tmp_path):
        """React mode + "你是谁" behavior unchanged (free ReAct loop)."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        llm = MockLLM(responses=[
            AIMessage(
                content=(
                    "I am Claude Code Mini.\n\n"
                    "---AGENT_STATUS---\n"
                    '{"action": "task_complete", "reason": "Answered"}\n'
                    "---END_STATUS---"
                )
            ),
            "Summary.",
        ])

        graph = build_graph(llm, registry, mode="react")

        initial_state = {
            "task": "who are you",
            "messages": [],
            "plan": [],
            "current_step_index": 0,
            "tool_history": [],
            "phase": "init",
            "iteration": 0,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "error_message": "",
            "final_answer": "",
        }

        final = await graph.ainvoke(initial_state)

        # React mode should complete normally
        assert final["phase"] == "done"
        assert len(final.get("tool_history", [])) == 0
