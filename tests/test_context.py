"""Unit tests for V2 ContextManager — context window management.

Tests verify:
  - Token estimation (heuristic char/4)
  - Messages under budget → returned as-is
  - Messages over budget → compressed with summary
  - Long tool results truncated
  - System messages preserved during compression
  - Extra context appended correctly

Run with:  pytest tests/test_context.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def make_messages(count: int, base_content: str = "test message") -> list:
    """Create a list of HumanMessage + AIMessage pairs."""
    msgs = []
    for i in range(count):
        msgs.append(HumanMessage(content=f"{base_content} {i}"))
        msgs.append(AIMessage(content=f"response {i}"))
    return msgs


# ═══════════════════════════════════════════════════════════════════
# Test: Token Estimation
# ═══════════════════════════════════════════════════════════════════

class TestTokenEstimation:
    """Verify the char/4 heuristic."""

    def test_estimate_tokens_empty(self):
        from memory.context_manager import estimate_tokens
        assert estimate_tokens([]) == 1  # max(1, ...)

    def test_estimate_tokens_basic(self):
        from memory.context_manager import estimate_tokens
        # 400 chars → ~100 tokens
        msgs = [HumanMessage(content="a" * 400)]
        assert estimate_tokens(msgs) == 100

    def test_estimate_tokens_multiple(self):
        from memory.context_manager import estimate_tokens
        msgs = [
            HumanMessage(content="hello world"),  # 11 chars
            AIMessage(content="hi there"),          # 8 chars
        ]
        # 19 chars // 4 = 4
        assert estimate_tokens(msgs) == 4

    def test_estimate_tokens_includes_tool_calls(self):
        from memory.context_manager import estimate_tokens
        msg = AIMessage(
            content="Let me check.",
            tool_calls=[
                {"name": "read_file", "args": {"file_path": "/x/y.py"}, "id": "c1", "type": "tool_call"},
            ],
        )
        # content="Let me check." = 14 chars + tool_call args/name = ~47 chars
        # total ~61 // 4 = 15
        tokens = estimate_tokens([msg])
        assert tokens > 0

    def test_estimate_message_tokens(self):
        from memory.context_manager import estimate_message_tokens
        assert estimate_message_tokens(AIMessage(content="abcd")) == 1  # 4//4=1


# ═══════════════════════════════════════════════════════════════════
# Test: ContextManager
# ═══════════════════════════════════════════════════════════════════

class TestContextManager:
    """Verify ContextManager behavior."""

    @pytest.fixture
    def mock_llm(self):
        """An async LLM that returns a canned summary."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary: user asked to fix bugs."))
        return llm

    @pytest.mark.asyncio
    async def test_under_budget_no_compression(self, mock_llm):
        """Messages under budget → returned as-is with extra context appended."""
        from memory.context_manager import ContextManager

        cm = ContextManager(mock_llm, max_tokens=100_000, keep_recent_messages=10)
        msgs = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="task"),
            AIMessage(content="ok"),
        ]

        result = await cm.prepare_messages(msgs, extra_context="step 1")
        assert len(result) == len(msgs) + 1  # extra context appended
        assert result[-1].content == "step 1"

    @pytest.mark.asyncio
    async def test_over_budget_compresses(self, mock_llm):
        """Messages over a very low budget → compression triggered."""
        from memory.context_manager import ContextManager

        # Very small budget forces compression
        cm = ContextManager(mock_llm, max_tokens=200, reserve_tokens=50, keep_recent_messages=2)
        msgs = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="a" * 200),   # ~50 tokens
            AIMessage(content="b" * 200),       # ~50 tokens
            HumanMessage(content="c" * 200),    # ~50 tokens
            AIMessage(content="d" * 200),       # ~50 tokens — total ~200+ tokens
        ]

        result = await cm.prepare_messages(msgs, extra_context="")
        # Should be system + summary + recent 2 + extra
        assert len(result) <= 4  # system + summary + 2 recent

        # System message preserved
        assert any(isinstance(m, SystemMessage) and "system prompt" in str(m.content) for m in result)

        # Summary injected
        summary_msgs = [m for m in result if isinstance(m, SystemMessage) and "Context Summary" in str(m.content)]
        assert len(summary_msgs) == 1

    @pytest.mark.asyncio
    async def test_few_messages_no_compression(self, mock_llm):
        """Few messages (under keep_recent) → no compression even if over budget."""
        from memory.context_manager import ContextManager

        cm = ContextManager(mock_llm, max_tokens=100, reserve_tokens=10, keep_recent_messages=100)
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="task"),
            AIMessage(content="ok"),
        ]

        result = await cm.prepare_messages(msgs, extra_context="extra")
        assert len(result) == len(msgs) + 1  # all kept + extra

    @pytest.mark.asyncio
    async def test_system_messages_preserved(self, mock_llm):
        """System messages are always preserved during compression."""
        from memory.context_manager import ContextManager

        cm = ContextManager(mock_llm, max_tokens=150, reserve_tokens=50, keep_recent_messages=1)
        msgs = [
            SystemMessage(content="sys1"),
            SystemMessage(content="sys2"),
            HumanMessage(content="a" * 200),
            AIMessage(content="b" * 200),
            HumanMessage(content="c" * 200),
            AIMessage(content="d" * 200),
        ]

        result = await cm.prepare_messages(msgs, extra_context="")
        sys_count = sum(1 for m in result if isinstance(m, SystemMessage))
        assert sys_count >= 2  # both system messages preserved

    @pytest.mark.asyncio
    async def test_extra_context_appended(self, mock_llm):
        """Extra context is always appended."""
        from memory.context_manager import ContextManager

        cm = ContextManager(mock_llm, max_tokens=100_000, keep_recent_messages=10)
        msgs = [HumanMessage(content="task")]

        result = await cm.prepare_messages(msgs, extra_context="## Step Context\nDo something.")
        assert result[-1].content == "## Step Context\nDo something."

    @pytest.mark.asyncio
    async def test_empty_messages(self, mock_llm):
        """Empty message list returns empty list."""
        from memory.context_manager import ContextManager
        cm = ContextManager(mock_llm)
        result = await cm.prepare_messages([], extra_context="")
        assert result == []

    @pytest.mark.asyncio
    async def test_summary_accessible(self, mock_llm):
        """Summary property is populated after compression."""
        from memory.context_manager import ContextManager

        # Tight budget: 100 tokens minus 50 reserve = 50 available
        # Messages: "a"*200 + "b"*200 = ~100 tokens → over budget → compression
        cm = ContextManager(mock_llm, max_tokens=100, reserve_tokens=10, keep_recent_messages=1)
        msgs = [
            HumanMessage(content="a" * 200),
            AIMessage(content="b" * 200),
        ]

        await cm.prepare_messages(msgs, extra_context="")
        assert cm.summary == "Summary: user asked to fix bugs."

    @pytest.mark.asyncio
    async def test_summarization_error_graceful(self, mock_llm):
        """If summarization LLM call fails, old summary is kept."""
        from memory.context_manager import ContextManager

        # First call succeeds to set summary
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Initial summary."))
        cm = ContextManager(mock_llm, max_tokens=100, reserve_tokens=10, keep_recent_messages=1)

        msgs1 = [
            HumanMessage(content="a" * 200),
            AIMessage(content="b" * 200),
        ]
        await cm.prepare_messages(msgs1, extra_context="")
        assert "Initial summary" in cm.summary

        # Second call fails
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        msgs2 = msgs1 + [
            HumanMessage(content="c" * 200),
            AIMessage(content="d" * 200),
        ]
        result = await cm.prepare_messages(msgs2, extra_context="")
        # Should still have the old summary
        assert cm.summary == "Initial summary."


# ═══════════════════════════════════════════════════════════════════
# Test: Tool Result Truncation
# ═══════════════════════════════════════════════════════════════════

class TestToolResultTruncation:
    """Verify long tool results are truncated."""

    @pytest.mark.asyncio
    async def test_long_tool_result_truncated(self):
        """Tool results > 2000 chars get truncated."""
        from memory.context_manager import ContextManager

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        cm = ContextManager(llm, max_tokens=100_000)

        long_content = "x" * 3000
        msgs = [
            SystemMessage(content="sys"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            ToolMessage(content=long_content, tool_call_id="c1"),
        ]

        result = await cm.prepare_messages(msgs, extra_context="")

        # Find the ToolMessage
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        content = tool_msgs[0].content
        assert "[truncated" in content
        assert len(content) < len(long_content)

    @pytest.mark.asyncio
    async def test_short_tool_result_not_truncated(self):
        """Short tool results (under 2000 chars) are untouched."""
        from memory.context_manager import ContextManager

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        cm = ContextManager(llm, max_tokens=100_000)

        short_content = "file contents here"
        msgs = [
            SystemMessage(content="sys"),
            ToolMessage(content=short_content, tool_call_id="c1"),
        ]

        result = await cm.prepare_messages(msgs, extra_context="")
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == short_content
