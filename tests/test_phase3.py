"""Unit tests for Phase 3 — Enhanced Planner + Reflector + Replan.

Covers: plan validation, LLM-based reflection, retry logic,
replanning, and error recovery paths.

Run with:  pytest tests/test_phase3.py -v
"""

import pytest
import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class MockLLM:
    """Fake LLM that returns canned responses by index."""

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
        return AIMessage(content="Default.")

    def bind_tools(self, schemas):
        self._bound_tools = schemas
        return self

    def invoke(self, messages, **kwargs):
        return AIMessage(content="sync")


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


# ═══════════════════════════════════════════════════════════════════
# Test: Plan Validation
# ═══════════════════════════════════════════════════════════════════

class TestPlanValidation:
    """Tests for _validate_step and plan_node enhancements."""

    def test_valid_step_passes(self):
        from graph.nodes import _validate_step
        step = {"id": "1", "description": "Search for all Python files using glob_search"}
        warnings = _validate_step(step)
        assert len(warnings) == 0

    def test_vague_step_warns(self):
        from graph.nodes import _validate_step
        step = {"id": "1", "description": "look at the code and understand it"}
        warnings = _validate_step(step)
        assert len(warnings) >= 1

    def test_too_short_step_warns(self):
        from graph.nodes import _validate_step
        step = {"id": "1", "description": "do it"}
        warnings = _validate_step(step)
        assert len(warnings) >= 1

    def test_step_with_edit_hint_passes(self):
        from graph.nodes import _validate_step
        step = {"id": "2", "description": "edit the main.py to fix import"}
        warnings = _validate_step(step)
        assert len(warnings) == 0

    def test_step_with_read_hint_passes(self):
        from graph.nodes import _validate_step
        step = {"id": "3", "description": "Read the requirements.txt file"}
        warnings = _validate_step(step)
        assert len(warnings) == 0  # "read the" matches _REQUIRED_TOOL_KEYWORDS

    def test_step_with_no_tool_keyword_warns(self):
        from graph.nodes import _validate_step
        # "understand" is in FORBIDDEN, and no tool keyword
        step = {"id": "1", "description": "understand the project structure"}
        warnings = _validate_step(step)
        assert len(warnings) >= 1

    def test_step_with_shell_keyword_passes(self):
        from graph.nodes import _validate_step
        step = {"id": "1", "description": "Run the tests using shell_execute"}
        warnings = _validate_step(step)
        assert len(warnings) == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Enhanced Plan Node
# ═══════════════════════════════════════════════════════════════════

class TestPlanNodeEnhanced:
    """Tests for the enhanced plan_node."""

    @pytest.mark.asyncio
    async def test_plan_node_adds_retry_fields(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"id": "1", "description": "Search Python files with glob_search"},
            {"id": "2", "description": "Read main.py"},
        ])
        llm = MockLLM(responses=[plan_json])

        state = {"task": "find files", "messages": []}
        result = await plan_node(state, llm)

        for step in result["plan"]:
            assert "retry_count" in step
            assert step["retry_count"] == 0
            assert "max_retries" in step
            assert step["max_retries"] == 2

    @pytest.mark.asyncio
    async def test_plan_node_truncates_over_10(self):
        from graph.nodes import plan_node

        steps = [{"id": str(i), "description": f"Step {i} with glob_search"} for i in range(1, 16)]
        llm = MockLLM(responses=[json.dumps(steps)])

        state = {"task": "complex task", "messages": []}
        result = await plan_node(state, llm)

        assert len(result["plan"]) == 10
        assert "truncated" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_plan_node_strips_backtick_fences(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"id": "1", "description": "Search for files using glob_search"},
        ])
        llm = MockLLM(responses=[f"```json\n{plan_json}\n```"])

        state = {"task": "test", "messages": []}
        result = await plan_node(state, llm)

        assert len(result["plan"]) == 1

    @pytest.mark.asyncio
    async def test_plan_node_strips_bare_backticks(self):
        from graph.nodes import plan_node

        plan_json = json.dumps([
            {"id": "1", "description": "Read main.py file"},
        ])
        llm = MockLLM(responses=[f"```\n{plan_json}\n```"])

        state = {"task": "test", "messages": []}
        result = await plan_node(state, llm)

        assert len(result["plan"]) == 1

    @pytest.mark.asyncio
    async def test_plan_node_fallback_on_bad_json(self):
        from graph.nodes import plan_node

        llm = MockLLM(responses=["completely invalid {{["])

        state = {"task": "fix the thing", "messages": []}
        result = await plan_node(state, llm)

        assert len(result["plan"]) == 1
        assert "fell back" in result.get("error_message", "").lower()
        assert result["phase"] == "executing"

    @pytest.mark.asyncio
    async def test_plan_node_fallback_on_empty_array(self):
        """Empty array [] is now a valid plan (BUG-001: conversational queries may return [])."""
        from graph.nodes import plan_node

        llm = MockLLM(responses=["[]"])

        state = {"task": "fix bugs", "messages": []}
        result = await plan_node(state, llm)

        # Empty plan is accepted (not a fallback) — allows non-coding queries
        # that reach plan_node to pass through without forced tool steps
        assert len(result["plan"]) == 0
        assert result["phase"] == "executing"


# ═══════════════════════════════════════════════════════════════════
# Test: Enhanced Reflect Node (LLM-based)
# ═══════════════════════════════════════════════════════════════════

class TestReflectNodeEnhanced:
    """Tests for the LLM-powered reflect_node."""

    def _make_basic_state(self, **overrides):
        state = {
            "task": "test task",
            "messages": [
                HumanMessage(content="do task"),
                AIMessage(content="I completed the step successfully."),
            ],
            "plan": [
                {
                    "id": "1",
                    "description": "Search files",
                    "status": "in_progress",
                    "retry_count": 0,
                    "max_retries": 2,
                },
                {
                    "id": "2",
                    "description": "Read file",
                    "status": "pending",
                    "retry_count": 0,
                    "max_retries": 2,
                },
            ],
            "current_step_index": 0,
            "iteration": 5,
            "max_iterations": 30,
            "step_retry_count": 0,
            "max_retries_per_step": 2,
            "tool_history": [],
            "phase": "executing",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_step_done_success(self):
        from graph.nodes import create_reflect_node

        # LLM says: step done, success
        reflection_json = json.dumps({
            "step_done": True,
            "success": True,
            "error_type": "none",
            "reasoning": "Step completed successfully.",
            "should_retry": False,
            "should_replan": False,
            "retry_suggestion": "",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state()
        result = await reflect_node(state)

        assert result["phase"] == "executing"
        assert result["current_step_index"] == 1  # advanced to next
        assert result["plan"][0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_step_done_last_step(self):
        from graph.nodes import create_reflect_node

        reflection_json = json.dumps({
            "step_done": True,
            "success": True,
            "error_type": "none",
            "reasoning": "All done.",
            "should_retry": False,
            "should_replan": False,
            "retry_suggestion": "",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state(
            plan=[
                {
                    "id": "1",
                    "description": "Only step",
                    "status": "in_progress",
                    "retry_count": 0,
                    "max_retries": 2,
                },
            ],
            current_step_index=0,
        )
        result = await reflect_node(state)

        assert result["phase"] == "done"

    @pytest.mark.asyncio
    async def test_recoverable_error_retry(self):
        from graph.nodes import create_reflect_node

        reflection_json = json.dumps({
            "step_done": False,
            "success": False,
            "error_type": "recoverable",
            "reasoning": "Wrong search pattern. Try again with correct regex.",
            "should_retry": True,
            "should_replan": False,
            "retry_suggestion": "Use a simpler regex pattern",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state()
        result = await reflect_node(state)

        assert result["phase"] == "retry"
        assert result["step_retry_count"] == 1
        assert "simpler regex" in result["error_message"].lower()

    @pytest.mark.asyncio
    async def test_replan_triggered(self):
        from graph.nodes import create_reflect_node

        reflection_json = json.dumps({
            "step_done": False,
            "success": False,
            "error_type": "wrong_approach",
            "reasoning": "The project uses FastAPI, not Flask. Plan needs revision.",
            "should_retry": False,
            "should_replan": True,
            "retry_suggestion": "",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state()
        result = await reflect_node(state)

        assert result["phase"] == "replan"

    @pytest.mark.asyncio
    async def test_fatal_error_moves_on(self):
        from graph.nodes import create_reflect_node

        reflection_json = json.dumps({
            "step_done": False,
            "success": False,
            "error_type": "fatal",
            "reasoning": "File does not exist and cannot be created.",
            "should_retry": False,
            "should_replan": False,
            "retry_suggestion": "",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state()
        result = await reflect_node(state)

        assert result["phase"] == "executing"
        assert result["current_step_index"] == 1  # moved on
        assert result["plan"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_fatal_error_last_step_goes_to_done(self):
        from graph.nodes import create_reflect_node

        reflection_json = json.dumps({
            "step_done": False,
            "success": False,
            "error_type": "fatal",
            "reasoning": "Cannot proceed.",
            "should_retry": False,
            "should_replan": False,
            "retry_suggestion": "",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state(
            plan=[
                {
                    "id": "1",
                    "description": "Only step",
                    "status": "in_progress",
                    "retry_count": 0,
                    "max_retries": 2,
                },
            ],
            current_step_index=0,
        )
        result = await reflect_node(state)

        assert result["phase"] == "done"

    @pytest.mark.asyncio
    async def test_max_step_retries_exceeded(self):
        from graph.nodes import create_reflect_node

        # Even if LLM says retry, the guard should prevent it
        reflection_json = json.dumps({
            "step_done": False,
            "success": False,
            "error_type": "recoverable",
            "reasoning": "Try again.",
            "should_retry": True,
            "should_replan": False,
            "retry_suggestion": "Try again",
        })
        llm = MockLLM(responses=[reflection_json])
        reflect_node = create_reflect_node(llm)

        # Already at max retries, single step plan → goes to done
        state = self._make_basic_state(
            step_retry_count=3,
            plan=[
                {
                    "id": "1", "description": "Only step",
                    "status": "in_progress", "retry_count": 3, "max_retries": 2,
                },
            ],
            current_step_index=0,
        )
        result = await reflect_node(state)

        assert result["phase"] == "done"
        assert result["plan"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self):
        from graph.nodes import create_reflect_node

        llm = MockLLM()  # won't be called
        reflect_node = create_reflect_node(llm)

        state = self._make_basic_state(iteration=30, max_iterations=30)
        result = await reflect_node(state)

        assert result["phase"] == "done"
        assert "max iterations" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        """If the reflection LLM call raises, we fall back to assuming step done."""
        from graph.nodes import create_reflect_node

        class BrokenLLM:
            async def ainvoke(self, messages, **kwargs):
                raise RuntimeError("API error")

        reflect_node = create_reflect_node(BrokenLLM())

        state = self._make_basic_state()
        result = await reflect_node(state)

        # Should gracefully advance
        assert result["phase"] in ("executing", "done")
        assert result["plan"][0]["status"] == "done"


# ═══════════════════════════════════════════════════════════════════
# Test: Replan Node
# ═══════════════════════════════════════════════════════════════════

class TestReplanNode:
    """Tests for the replan_node."""

    @pytest.mark.asyncio
    async def test_replan_keeps_completed_rewrites_remaining(self):
        from graph.nodes import replan_node

        new_steps = json.dumps([
            {"id": "3", "description": "Revised: search with new pattern"},
            {"id": "4", "description": "Revised: verify with shell_execute"},
        ])
        llm = MockLLM(responses=[new_steps])

        state = {
            "task": "add feature",
            "plan": [
                {"id": "1", "description": "Find files", "status": "done", "retry_count": 0, "max_retries": 2},
                {"id": "2", "description": "Read config", "status": "done", "retry_count": 0, "max_retries": 2},
                {"id": "3", "description": "Bad: do something wrong", "status": "pending", "retry_count": 0, "max_retries": 2},
                {"id": "4", "description": "Bad: verify wrong thing", "status": "pending", "retry_count": 0, "max_retries": 2},
            ],
            "current_step_index": 2,
            "error_message": "The plan was wrong.",
            "tool_history": [
                {"tool": "glob_search", "args": {"pattern": "*.py"}, "result": "found files", "success": True},
            ],
        }
        result = await replan_node(state, llm)

        # Should have kept completed steps
        assert len(result["plan"]) == 4  # 2 completed + 2 new
        assert result["plan"][0]["status"] == "done"
        assert result["plan"][1]["status"] == "done"
        assert result["plan"][2]["status"] == "pending"  # new
        assert result["plan"][3]["status"] == "pending"  # new
        assert result["current_step_index"] == 2  # starts from first remaining
        assert result["phase"] == "executing"

    @pytest.mark.asyncio
    async def test_replan_fallback_on_bad_llm_output(self):
        from graph.nodes import replan_node

        llm = MockLLM(responses=["not a json array at all {{["])

        state = {
            "task": "fix bugs",
            "plan": [
                {"id": "1", "description": "Done step", "status": "done", "retry_count": 0, "max_retries": 2},
                {"id": "2", "description": "Bad step", "status": "pending", "retry_count": 0, "max_retries": 2},
            ],
            "current_step_index": 1,
            "error_message": "Needs replan",
            "tool_history": [],
        }
        result = await replan_node(state, llm)

        # Fallback should preserve structure
        assert len(result["plan"]) == 2  # 1 completed + 1 fallback
        assert result["plan"][0]["status"] == "done"
        assert result["plan"][1]["status"] == "pending"
        assert result["phase"] == "executing"


# ═══════════════════════════════════════════════════════════════════
# Test: JSON Extraction Helpers
# ═══════════════════════════════════════════════════════════════════

class TestJsonExtraction:
    """Tests for _extract_json_array and _extract_json_object."""

    def test_extract_json_array_clean(self):
        from graph.nodes import _extract_json_array
        result = _extract_json_array('[{"id": "1"}]')
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_extract_json_array_with_fences(self):
        from graph.nodes import _extract_json_array
        result = _extract_json_array('```json\n[{"id": "1", "desc": "test"}]\n```')
        assert len(result) == 1
        assert result[0]["desc"] == "test"

    def test_extract_json_array_buried_in_text(self):
        from graph.nodes import _extract_json_array
        # LLM sometimes outputs: "Here's the plan:\n[{\"id\":\"1\"}]\nThat's all."
        result = _extract_json_array('Here is the plan:\n\n[{"id": "1", "description": "search"}]\n\nHope this helps!')
        assert len(result) == 1
        assert result[0]["description"] == "search"

    def test_extract_json_array_raises_on_garbage(self):
        from graph.nodes import _extract_json_array
        with pytest.raises(ValueError):
            _extract_json_array("no json here at all")

    def test_extract_json_object_clean(self):
        from graph.nodes import _extract_json_object
        result = _extract_json_object('{"key": "value"}')
        assert result["key"] == "value"

    def test_extract_json_object_with_fences(self):
        from graph.nodes import _extract_json_object
        result = _extract_json_object('```\n{"ok": true}\n```')
        assert result["ok"] is True

    def test_extract_json_object_raises_on_garbage(self):
        from graph.nodes import _extract_json_object
        with pytest.raises(ValueError):
            _extract_json_object("no json at all")


# ═══════════════════════════════════════════════════════════════════
# Test: End-to-End with Retry and Replan
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndPhase3:
    """Integration tests for the full Phase 3 graph with retry/replan."""

    @pytest.mark.asyncio
    async def test_retry_flow(self, tmp_path):
        """Step fails → reflection triggers retry → second attempt succeeds."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        (tmp_path / "config.py").write_text("DEBUG=True")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read config.py"},
        ])

        # LLM response sequence:
        # 1. plan → plan_json
        # 2. execute (step 1, attempt 1) → tool_call (read_file with wrong path)
        # 3. execute (after tool error) → text "I couldn't read it"
        # 4. reflect → recoverable error → retry
        # 5. execute (step 1, attempt 2) → tool_call with correct path
        # 6. execute (after tool success) → text "Read successfully"
        # 7. reflect → success
        # 8. finish → summary

        # Pre-canned Intent Auditor response: entailment
        _AUDIT_ENTAILMENT = json.dumps({
            "label": "entailment", "score": 0.95,
            "reason": "Step directly serves the goal.",
        })

        llm = MockLLM(responses=[
            # plan
            plan_json,
            # audit_plan (1 step)
            _AUDIT_ENTAILMENT,
            # execute step 1 attempt 1: call tool with wrong path
            make_tool_call_ai([("read_file", {"file_path": "nonexistent.py"}, "c1")]),
            # execute after tool error → step didn't complete
            make_text_ai("I couldn't find the file. Need to try again."),
            # reflect → recoverable
            json.dumps({
                "step_done": False, "success": False, "error_type": "recoverable",
                "reasoning": "Wrong file path.", "should_retry": True,
                "should_replan": False, "retry_suggestion": "Try config.py instead",
            }),
            # execute step 1 attempt 2: correct tool call
            make_tool_call_ai([("read_file", {"file_path": "config.py"}, "c2")]),
            # execute after tool success
            make_text_ai("Successfully read config.py. It contains DEBUG=True."),
            # reflect → success
            json.dumps({
                "step_done": True, "success": True, "error_type": "none",
                "reasoning": "File read successfully.", "should_retry": False,
                "should_replan": False, "retry_suggestion": "",
            }),
            # finish
            "Task completed: read config.py successfully.",
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "Read the config file",
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
        # Step 1 should be marked done (retried then succeeded)
        assert final["plan"][0]["status"] == "done"

    @pytest.mark.asyncio
    async def test_replan_flow(self, tmp_path):
        """Step fails with wrong_approach → replan → continue with revised plan."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        original_plan = json.dumps([
            {"id": "1", "description": "Read Flask config"},
            {"id": "2", "description": "Add Flask route"},
        ])

        revised_steps = json.dumps([
            {"id": "2", "description": "Revised: Read FastAPI config instead"},
            {"id": "3", "description": "Revised: Add FastAPI endpoint"},
        ])

        # Pre-canned Intent Auditor response: entailment
        _AUDIT_ENTAILMENT2 = json.dumps({
            "label": "entailment", "score": 0.95,
            "reason": "Step directly serves the goal.",
        })

        llm = MockLLM(responses=[
            # plan
            original_plan,
            # audit_plan (2 steps)
            _AUDIT_ENTAILMENT2,
            _AUDIT_ENTAILMENT2,
            # execute step 1 → tool call
            make_tool_call_ai([("glob_search", {"pattern": "*.py"}, "c1")]),
            # execute → text response
            make_text_ai("This project uses FastAPI, not Flask. Can't find Flask config."),
            # reflect → wrong_approach → replan
            json.dumps({
                "step_done": False, "success": False, "error_type": "wrong_approach",
                "reasoning": "Project is FastAPI, plan assumed Flask.",
                "should_retry": False, "should_replan": True, "retry_suggestion": "",
            }),
            # replan → revised_steps
            revised_steps,
            # execute new step 1 → tool call
            make_tool_call_ai([("read_file", {"file_path": "main.py"}, "c2")]),
            # execute → text
            make_text_ai("Step completed with revised approach."),
            # reflect → success
            json.dumps({
                "step_done": True, "success": True, "error_type": "none",
                "reasoning": "OK.", "should_retry": False,
                "should_replan": False, "retry_suggestion": "",
            }),
            # reflect → last step done
            json.dumps({
                "step_done": True, "success": True, "error_type": "none",
                "reasoning": "OK.", "should_retry": False,
                "should_replan": False, "retry_suggestion": "",
            }),
            # finish
            "All done.",
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "Add a new endpoint",
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
        # Plan should have been revised
        assert len(final["plan"]) >= 2

    @pytest.mark.asyncio
    async def test_fatal_error_still_completes(self, tmp_path):
        """When a step has a fatal error, we mark it failed but finish normally."""
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read a file that doesn't exist"},
        ])

        # Pre-canned Intent Auditor response: entailment
        _AUDIT_ENTAILMENT3 = json.dumps({
            "label": "entailment", "score": 0.95,
            "reason": "Step directly serves the goal.",
        })

        llm = MockLLM(responses=[
            plan_json,
            # audit_plan (1 step)
            _AUDIT_ENTAILMENT3,
            make_tool_call_ai([("read_file", {"file_path": "missing.py"}, "c1")]),
            make_text_ai("The file doesn't exist."),
            # reflect → fatal
            json.dumps({
                "step_done": False, "success": False, "error_type": "fatal",
                "reasoning": "File not found and not recoverable.",
                "should_retry": False, "should_replan": False, "retry_suggestion": "",
            }),
            # finish
            "Task attempted. File was missing.",
        ])

        graph = build_graph(llm, registry, mode="plan")

        initial_state = {
            "task": "Read missing file",
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
        assert final["plan"][0]["status"] == "failed"
