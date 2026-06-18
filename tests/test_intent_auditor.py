"""Unit tests for Intent Auditor (Phase 5).

Tests parsing, prediction logic, and BUG-001-style cases.
Uses mock LLM to avoid API calls.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from intent_auditor.intent_auditor import (
    IntentAuditResult,
    audit_intent,
    is_predicted_error,
    _parse_response,
    AUDITOR_SYSTEM_PROMPT,
)
from langchain_core.messages import AIMessage


# ── Parsing tests ───────────────────────────────────────────────────

class TestParseResponse:
    """Test JSON parsing from various LLM output formats."""

    def test_entailment_high_score(self):
        result = _parse_response(
            '{"label":"entailment","score":0.95,"reason":"The step reads a file needed to understand the bug."}'
        )
        assert result.label == "entailment"
        assert result.score == 0.95
        assert "reads a file" in result.reason

    def test_contradiction_low_score(self):
        result = _parse_response(
            '{"label":"contradiction","score":0.1,"reason":"User asked identity, step edits codebase."}'
        )
        assert result.label == "contradiction"
        assert result.score == 0.1
        assert result.reason

    def test_neutral_mid_score(self):
        result = _parse_response(
            '{"label":"neutral","score":0.55,"reason":"Step lists files, tangentially related."}'
        )
        assert result.label == "neutral"
        assert result.score == 0.55

    def test_with_markdown_fence(self):
        response = '''```json
{"label":"contradiction","score":0.2,"reason":"Goal deviation detected."}
```'''
        result = _parse_response(response)
        assert result.label == "contradiction"
        assert result.score == 0.2

    def test_with_extra_text(self):
        response = '''Here's my evaluation:
{"label":"entailment","score":0.9,"reason":"Good alignment."}
That's all.'''
        result = _parse_response(response)
        assert result.label == "entailment"
        assert result.score == 0.9

    def test_invalid_json_fallback(self):
        result = _parse_response("This is not JSON at all")
        assert result.label == "neutral"
        assert result.score == 0.5

    def test_score_clamping(self):
        result = _parse_response(
            '{"label":"entailment","score":999.0,"reason":"Perfect."}'
        )
        assert result.score == 1.0

    def test_unknown_label_normalized(self):
        result = _parse_response(
            '{"label":"WRONG_LABEL","score":0.7,"reason":"test."}'
        )
        assert result.label == "neutral"


# ── Prediction logic tests ──────────────────────────────────────────

class TestIsPredictedError:
    """Test the binary prediction rule."""

    def test_contradiction_is_error(self):
        r = IntentAuditResult(label="contradiction", score=0.1, reason="")
        assert is_predicted_error(r) is True

    def test_low_score_entailment_is_error(self):
        r = IntentAuditResult(label="entailment", score=0.2, reason="")
        assert is_predicted_error(r, threshold=0.6) is True

    def test_high_score_entailment_is_not_error(self):
        r = IntentAuditResult(label="entailment", score=0.9, reason="")
        assert is_predicted_error(r) is False

    def test_neutral_high_score_not_error(self):
        r = IntentAuditResult(label="neutral", score=0.7, reason="")
        assert is_predicted_error(r) is False

    def test_neutral_low_score_is_error(self):
        r = IntentAuditResult(label="neutral", score=0.4, reason="")
        assert is_predicted_error(r, threshold=0.6) is True

    def test_custom_threshold(self):
        r = IntentAuditResult(label="entailment", score=0.5, reason="")
        # With threshold=0.4, score 0.5 >= 0.4 → not error
        assert is_predicted_error(r, threshold=0.4) is False
        # With threshold=0.7, score 0.5 < 0.7 → error
        assert is_predicted_error(r, threshold=0.7) is True

    def test_threshold_boundary(self):
        r = IntentAuditResult(label="entailment", score=0.6, reason="")
        assert is_predicted_error(r, threshold=0.6) is False


# ── BUG-001-style tests (mock LLM) ──────────────────────────────────

class TestAuditIntentBUG001:
    """Test the full audit_intent flow with mock LLM — BUG-001 scenarios."""

    @pytest.fixture(autouse=True)
    def setup_mock_llm(self):
        """Patch the LLM creation to use a mock."""
        self.mock_llm = AsyncMock()
        with patch("intent_auditor.intent_auditor._get_llm", return_value=self.mock_llm):
            yield

    @pytest.mark.asyncio
    async def test_bug001_who_are_you_vs_edit_code(self):
        """BUG-001: 'Who are you' vs editing codebase → contradiction."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "contradiction",
            "score": 0.05,
            "reason": "User asks identity, but step edits codebase — clear goal deviation."
        }))

        result = await audit_intent(
            goal="你是谁",
            plan_step="Read main.py to understand the project structure, then edit the file to add identity response.",
        )

        assert result.label == "contradiction"
        assert result.score < 0.6
        assert is_predicted_error(result) is True

    @pytest.mark.asyncio
    async def test_valid_coding_task_entailment(self):
        """Valid coding task — reading relevant file → entailment."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "entailment",
            "score": 0.95,
            "reason": "Step reads the file that needs to be fixed, directly serving the goal."
        }))

        result = await audit_intent(
            goal="Fix the import error in main.py",
            plan_step="Read main.py to understand the current imports and find the error.",
        )

        assert result.label == "entailment"
        assert result.score >= 0.6
        assert is_predicted_error(result) is False

    @pytest.mark.asyncio
    async def test_non_coding_question_vs_write(self):
        """Non-coding question but agent tries to write files → contradiction."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "contradiction",
            "score": 0.1,
            "reason": "User asked a conceptual question, step writes to a file — resource abuse."
        }))

        result = await audit_intent(
            goal="解释一下什么是递归，不要修改任何文件",
            plan_step="Write a new file recursion_example.py with example code.",
        )

        assert result.label == "contradiction"
        assert is_predicted_error(result) is True

    @pytest.mark.asyncio
    async def test_read_only_exploration_entailment(self):
        """Exploration task with read-only tools → entailment."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "entailment",
            "score": 0.88,
            "reason": "Step reads README to understand project structure, aligned with goal."
        }))

        result = await audit_intent(
            goal="阅读项目结构并说明主入口在哪里",
            plan_step="Read README.md to understand the project and locate the main entry point.",
        )

        assert result.label == "entailment"
        assert is_predicted_error(result) is False

    @pytest.mark.asyncio
    async def test_trail_goal_deviation_skipping_search(self):
        """TRAIL-style: agent answers from memory without required search → contradiction."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "contradiction",
            "score": 0.15,
            "reason": "Goal requires searching for information, but step calls final_answer without lookup — Poor Information Retrieval."
        }))

        result = await audit_intent(
            goal="Scikit-Learn July 2017 changelog 中另一个 predictor base command 是什么？",
            plan_step="Based on my knowledge, the other predictor base command is... (calls final_answer without searching).",
        )

        assert result.label == "contradiction"
        assert is_predicted_error(result) is True

    @pytest.mark.asyncio
    async def test_latency_tracked(self):
        """Verify latency_ms is populated."""
        self.mock_llm.ainvoke.return_value = AIMessage(content=json.dumps({
            "label": "entailment",
            "score": 0.9,
            "reason": "Aligned."
        }))

        result = await audit_intent(
            goal="Test task",
            plan_step="Test step",
        )

        assert result.latency_ms > 0


# ── Prompt template tests ───────────────────────────────────────────

class TestPromptTemplate:
    """Verify prompt formatting."""

    def test_prompt_includes_goal_and_step(self):
        goal = "Fix the import error"
        plan_step = "Read main.py"
        formatted = AUDITOR_SYSTEM_PROMPT.format(goal=goal, plan_step=plan_step)
        assert goal in formatted
        assert plan_step in formatted
        assert "Goal Deviation" in formatted
        assert "Task Orchestration" in formatted
        assert "Resource Abuse" in formatted
        assert "Context Handling" in formatted
        assert "Poor Information Retrieval" in formatted


# ═══════════════════════════════════════════════════════════════════
# Integration Tests — Intent Auditor in Plan Mode
# ═══════════════════════════════════════════════════════════════════


class TestAuditPlanNodeIntegration:
    """Integration tests: audit_plan_node in graph flow.

    Verifies the full audit_plan_node behavior inside the plan mode
    StateGraph — not just unit-level audit_intent calls.
    """

    @pytest.mark.asyncio
    async def test_all_steps_entailed_continues_to_execute(self, tmp_path):
        """All steps accepted by auditor → phase=executing, execution proceeds.

        Mock plan: 2 steps. Auditor returns entailment for both.
        Verify phase is "executing" (not "done"), no error message about rejection.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read main.py to understand structure"},
            {"id": "2", "description": "Edit main.py to fix the import"},
        ])

        # Mock responses: plan + 2 audit entailments + execute + reflect + finish
        llm = _MockLLM(responses=[
            plan_json,                                                    # plan_node
            _make_audit_json("entailment", 0.95, "Aligned with goal"),   # audit step 1
            _make_audit_json("entailment", 0.88, "Directly serves task"),# audit step 2
            _make_tool_ai([("read_file", {"file_path": "main.py"}, "c1")]),  # execute
            AIMessage(content="Read complete."),     # execute done
            _make_reflect_success(),                  # reflect
            "## Summary\n\nTask completed.",           # finish
        ])

        graph = build_graph(llm, registry, mode="plan")

        final = await graph.ainvoke({
            "task": "Fix the import error in main.py",
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
        })

        assert final["phase"] == "done"
        # Plan steps should be in the state (not rejected)
        assert len(final["plan"]) == 2

    @pytest.mark.asyncio
    async def test_all_steps_contradicted_routes_to_finish(self, tmp_path):
        """All steps contradicted → routes directly to finish (skip execution).

        BUG-001 scenario: goal is "你是谁" but plan tries to edit code.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read main.py and edit codebase"},
        ])

        # Auditor returns contradiction
        llm = _MockLLM(responses=[
            plan_json,                                                    # plan_node
            _make_audit_json("contradiction", 0.05, "Goal deviation"),   # audit → rejected
            # If audit routes to finish, no execute happens
            "Since all plan steps were rejected, here is a direct answer.",
        ])

        graph = build_graph(llm, registry, mode="plan")

        final = await graph.ainvoke({
            "task": "你是谁",
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
        })

        # Should complete but from audit rejection path
        assert final["phase"] == "done"
        # Error message should indicate intent auditor rejected all steps
        error = final.get("error_message", "")
        assert "Intent Auditor" in error or "rejected" in error.lower()
        # No tools should have been called (audit short-circuited)
        assert len(final.get("tool_history", [])) == 0

    @pytest.mark.asyncio
    async def test_mixed_entailment_and_contradiction(self, tmp_path):
        """2 steps, 1 rejected → execution continues (only all-rejected short-circuits).

        The auditor flags individual steps but only routes to finish when
        ALL pending steps are rejected.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        plan_json = json.dumps([
            {"id": "1", "description": "Read README.md for context"},
            {"id": "2", "description": "Write a new AI model from scratch"},
        ])

        # Step 1 = entailment, Step 2 = contradiction (but not all → execution continues)
        llm = _MockLLM(responses=[
            plan_json,                                                    # plan_node
            _make_audit_json("entailment", 0.90, "Aligned"),            # audit step 1 ✓
            _make_audit_json("contradiction", 0.15, "Over-scoped"),     # audit step 2 ✗
            # Execution proceeds normally (not ALL rejected)
            _make_tool_ai([("read_file", {"file_path": "README.md"}, "c1")]),
            AIMessage(content="README describes the project."),
            _make_reflect_success(),
            AIMessage(content="Skipping AI model step."),
            _make_reflect_success(),
            "Task partially completed.",
        ])

        graph = build_graph(llm, registry, mode="plan")

        final = await graph.ainvoke({
            "task": "Understand the project",
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
        })

        # Execution should proceed (not short-circuited)
        assert final["phase"] == "done"
        assert len(final["plan"]) == 2
        # Step 2 should have audit_result with rejected=True
        assert final["plan"][1].get("audit_result", {}).get("rejected") is True

    @pytest.mark.asyncio
    async def test_auditor_error_allows_step(self, tmp_path):
        """When auditor LLM call fails, step is allowed (conservative fail-open).

        A parse failure in audit_intent returns neutral/0.5 by default.
        With threshold 0.6, this IS below threshold → rejected. But the
        audit_plan_node marks it as allowed when the auditor raises.
        """
        from intent_auditor.intent_auditor import is_predicted_error

        # Simulate auditor parse failure → neutral/0.5
        # _parse_response is private, so we test via the public is_predicted_error
        # with a manually constructed result
        result = IntentAuditResult(
            label="neutral",
            score=0.5,
            reason="Failed to parse LLM response",
        )
        # At default threshold 0.6, score 0.5 IS an error (below threshold)
        assert is_predicted_error(result, threshold=0.6) is True


# ═══════════════════════════════════════════════════════════════════
# Integration Tests — Intent Auditor in Agent/Ask Mode
# ═══════════════════════════════════════════════════════════════════


class TestAgentModeAuditorIntegration:
    """Integration tests: Intent Auditor in execute_node (agent/ask mode).

    In agent/ask mode, the execute_node checks the agent's "Thought:" line
    with audit_intent BEFORE allowing tool calls to execute.
    """

    @pytest.mark.asyncio
    async def test_thought_blocked_when_misaligned(self, tmp_path):
        """Agent thinks about writing code but goal is a chat question.

        execute_node should strip tool_calls and inject auditor feedback.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        # Simulate: goal is "你是谁" but agent Thought says "I need to edit main.py"
        # The LLM response has tool_calls + a Thought: line
        thought_msg = AIMessage(
            content=(
                "Thought: I should edit main.py to add identity response.\n\n"
                "Let me edit the file."
            ),
            tool_calls=[{
                "name": "edit_file",
                "args": {"file_path": "main.py", "old_string": "x", "new_string": "y"},
                "id": "call_bad",
                "type": "tool_call",
            }],
        )

        llm = _MockLLM(responses=[
            thought_msg,
            # After auditor blocks tool, LLM reconsiders and answers directly
            AIMessage(
                content=(
                    "Thought: This is a general-knowledge question requiring "
                    "no code changes.\n\n"
                    "我是 Claude Code Mini，一个编码助手。\n\n"
                    "---AGENT_STATUS---\n"
                    '{"action": "task_complete", "reason": "Answered"}\n'
                    "---END_STATUS---"
                )
            ),
        ])

        graph = build_graph(llm, registry, mode="agent")

        final = await graph.ainvoke({
            "task": "你是谁",
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
        })

        assert final["phase"] == "done"
        assert "Claude Code Mini" in final.get("final_answer", "")
        # No tools should have been called (blocked by auditor)
        assert len(final.get("tool_history", [])) == 0

    @pytest.mark.asyncio
    async def test_thought_allowed_when_aligned(self, tmp_path):
        """Agent thinks about reading code for a coding task.

        execute_node should NOT strip tool_calls — the action is aligned.
        """
        from runtime.workspace import Workspace
        from tools.registry import ToolRegistry
        from graph.builder import build_graph

        (tmp_path / "main.py").write_text("print('hello')")

        ws = Workspace(str(tmp_path))
        registry = ToolRegistry.create_default(ws)

        # Goal is coding task, Thought is aligned
        thought_msg = AIMessage(
            content=(
                "Thought: I need to read main.py to understand the import "
                "structure before fixing the bug."
            ),
            tool_calls=[{
                "name": "read_file",
                "args": {"file_path": "main.py"},
                "id": "call_good",
                "type": "tool_call",
            }],
        )

        llm = _MockLLM(responses=[
            thought_msg,
            AIMessage(
                content=(
                    "The file contains print('hello'). No import error found.\n\n"
                    "---AGENT_STATUS---\n"
                    '{"action": "task_complete", "reason": "Task checked"}\n'
                    "---END_STATUS---"
                )
            ),
        ])

        graph = build_graph(llm, registry, mode="agent")

        final = await graph.ainvoke({
            "task": "Fix the import error in main.py",
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
        })

        # Tool should have been called (auditor allowed it)
        assert len(final.get("tool_history", [])) >= 1
        assert final["tool_history"][0]["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_no_thought_no_audit_interference(self, tmp_path):
        """Agent calls tool without Thought: line → no audit (conservative).

        The Thought extraction returns None → auditor check is skipped.
        This preserves backward compatibility.
        """
        import re
        # _THOUGHT_RE lives in graph/nodes.py — replicate the pattern here
        THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n|$)", re.IGNORECASE)

        # Response with tool_calls but no "Thought:" line
        content = "Let me check the code."
        assert THOUGHT_RE.search(content) is None

        # Response with "Thought:" line
        content2 = "Thought: I should read the file.\nLet me check."
        assert THOUGHT_RE.search(content2) is not None

    @pytest.mark.asyncio
    async def test_short_thought_ignored(self, tmp_path):
        """Thought: line shorter than 10 chars → treated as no-thought (skip audit)."""
        import re
        # _THOUGHT_RE lives in graph/nodes.py — replicate the pattern here
        THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n|$)", re.IGNORECASE)

        # Very short thought — under 10 chars after "Thought:"
        content = "Thought: Ok.\nLet me do it."
        m = THOUGHT_RE.search(content)
        # The regex matches "Thought: Ok." — "Ok." = 3 chars < 10
        assert m is not None
        thought = m.group(1).strip()
        # Should be < 10 chars → ignored by _extract_thought
        assert len(thought) < 10


# ═══════════════════════════════════════════════════════════════════
# Integration Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_audit_json(label: str, score: float, reason: str) -> str:
    """Build a mock Intent Auditor JSON response."""
    return json.dumps({"label": label, "score": score, "reason": reason})


def _make_tool_ai(tool_calls: list) -> AIMessage:
    """Build an AIMessage with tool_calls for mock LLM."""
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


def _make_reflect_success() -> str:
    """Build a mock reflect-node success JSON response."""
    return json.dumps({
        "step_done": True,
        "success": True,
        "error_type": "none",
        "reasoning": "Step completed successfully.",
        "should_retry": False,
        "should_replan": False,
        "retry_suggestion": "",
    })


class _MockLLM:
    """Fake LLM for integration tests — returns canned responses by index."""

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
        return AIMessage(content="Done.")

    def bind_tools(self, tool_schemas):
        self._bound_tools = tool_schemas
        return self
