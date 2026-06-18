"""ContextManager — intelligent context window management.

Replaces the V1 "discard after 40 messages" approach with:
  - Token budget tracking (configurable max_tokens)
  - Rolling summarization of older messages
  - Preservation of system messages, task, and recent context
  - Truncation of long tool results
"""

from typing import List, Optional

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel


# ═══════════════════════════════════════════════════════════════════
# Token estimation utilities
# ═══════════════════════════════════════════════════════════════════

def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Estimate total tokens for a list of messages.

    Uses character-count / 4 heuristic — provider-agnostic, fast,
    and accurate enough for budget management (±15%).

    Args:
        messages: List of LangChain messages.

    Returns:
        Approximate token count.
    """
    total_chars = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total_chars += len(str(content))
        # Tool calls also have structured args that consume tokens
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                total_chars += len(str(tc.get("args", {})))
                total_chars += len(str(tc.get("name", "")))
    return max(1, total_chars // 4)


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate tokens for a single message."""
    content = getattr(message, "content", "") or ""
    return max(1, len(str(content)) // 4)


# ═══════════════════════════════════════════════════════════════════
# ContextManager
# ═══════════════════════════════════════════════════════════════════

# Maximum characters for a tool result before truncation
_MAX_TOOL_RESULT_CHARS = 2000

# Default summary prompt
_SUMMARY_SYSTEM_PROMPT = (
    "You are a context summarizer. Given a conversation between a coding agent "
    "and the user/tools, produce a concise rolling summary that preserves:\n"
    "- The user's original task\n"
    "- Key files that were read or modified\n"
    "- Any errors encountered and how they were addressed\n"
    "- Current progress in the task plan\n"
    "- Important discoveries about the project structure\n\n"
    "Output ONLY the summary paragraph. No introduction, no meta-commentary."
)


class ContextManager:
    """Manages context window budget via token estimation and summarization.

    When the message history exceeds the token budget, instead of
    discarding messages (V1), this class:
      1. Keeps all SystemMessages (identity + tool descriptions)
      2. Keeps the most recent N messages intact
      3. Compresses older messages into a rolling summary
      4. Injects the summary as a SystemMessage

    This preserves critical context while staying within budget.

    Usage:
        cm = ContextManager(llm, max_tokens=120_000)
        prepared = await cm.prepare_messages(state.messages, extra_context)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_tokens: int = 120_000,
        reserve_tokens: int = 8_000,
        keep_recent_messages: int = 20,
    ):
        """Initialize the context manager.

        Args:
            llm: LLM for generating summaries (can be the same as the main agent LLM).
            max_tokens: Maximum token budget for the full message list.
            reserve_tokens: Tokens reserved for the LLM response + tool schemas.
            keep_recent_messages: Number of most recent messages to always keep intact.
        """
        self._llm = llm
        self._max_tokens = max_tokens
        self._reserve_tokens = reserve_tokens
        self._keep_recent = keep_recent_messages
        self._summary: str = ""
        self._summary_token_estimate: int = 0

    # ── Public API ────────────────────────────────────────────────

    async def prepare_messages(
        self,
        messages: List[BaseMessage],
        extra_context: str = "",
    ) -> List[BaseMessage]:
        """Prepare messages for the next LLM call, compressing if needed.

        Args:
            messages: The full message history from AgentState.
            extra_context: Additional context to append (step context, etc.).

        Returns:
            A (possibly compressed) list of messages ready for LLM invocation.
        """
        if not messages:
            return []

        # ── Truncate long tool results ──────────────────────
        messages = self._truncate_long_tool_results(messages)

        # ── Estimate current tokens ─────────────────────────
        # Include extra_context in the estimate
        extra_tokens = len(extra_context) // 4 if extra_context else 0
        current_tokens = estimate_tokens(messages) + extra_tokens

        available = self._max_tokens - self._reserve_tokens

        if current_tokens <= available:
            # Under budget — return as-is with extra context appended
            result = list(messages)
            if extra_context:
                result.append(HumanMessage(content=extra_context))
            return result

        # ── Over budget — compress ──────────────────────────
        return await self._compress(messages, extra_context, available)

    async def _compress(
        self,
        messages: List[BaseMessage],
        extra_context: str,
        available_tokens: int,
    ) -> List[BaseMessage]:
        """Compress message history when over budget.

        Strategy:
          1. Separate system messages
          2. Keep the most recent N non-system messages
          3. Summarize older messages via LLM
          4. Combine: system + summary + recent + extra_context
        """
        # Separate system from non-system messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # If few enough messages, just keep recent
        if len(other_msgs) <= self._keep_recent:
            result = list(messages)
            if extra_context:
                result.append(HumanMessage(content=extra_context))
            return result

        # Split: older messages to compress, recent messages to keep
        recent_msgs = other_msgs[-self._keep_recent:]
        older_msgs = other_msgs[:-self._keep_recent]

        # Generate or update rolling summary
        await self._update_summary(older_msgs)

        # Build compressed message list
        result = list(system_msgs)

        if self._summary:
            result.append(SystemMessage(
                content=f"[Context Summary]\n{self._summary}"
            ))

        result.extend(recent_msgs)

        if extra_context:
            result.append(HumanMessage(content=extra_context))

        return result

    async def _update_summary(self, older_msgs: List[BaseMessage]):
        """Generate or update the rolling summary from older messages.

        Extracts key information about task progress, file changes, and errors.
        Uses the existing summary as prior context (rolling update).
        """
        # Build a compact representation of older messages
        older_text = self._format_for_summarization(older_msgs)

        if not older_text.strip():
            return

        prompt = _SUMMARY_SYSTEM_PROMPT
        if self._summary:
            prompt += f"\n\n## Previous Summary\n{self._summary}\n\n## New Messages to Incorporate"

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=older_text),
            ])
            self._summary = response.content.strip()
            self._summary_token_estimate = estimate_message_tokens(
                SystemMessage(content=self._summary)
            )
        except Exception:
            # If summarization fails, keep old summary (graceful degradation)
            pass

    # ── Helpers ────────────────────────────────────────────────────

    def _truncate_long_tool_results(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """Truncate ToolMessage content that exceeds the character limit.

        Keeps first 1800 chars + appends '[truncated, N total chars]'.
        """
        result = []
        for m in messages:
            if isinstance(m, ToolMessage) and m.content:
                content = m.content
                if len(content) > _MAX_TOOL_RESULT_CHARS:
                    truncated = (
                        content[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n[truncated, {len(content)} total chars]"
                    )
                    result.append(ToolMessage(
                        content=truncated,
                        tool_call_id=m.tool_call_id if hasattr(m, "tool_call_id") else "",
                    ))
                else:
                    result.append(m)
            else:
                result.append(m)
        return result

    def _format_for_summarization(
        self, messages: List[BaseMessage]
    ) -> str:
        """Format older messages for the summarization LLM call.

        Extracts the essential content — truncates very long messages,
        skips pure tool-call AIMessages (no content).
        """
        lines = []
        for m in messages:
            role = type(m).__name__.replace("Message", "").upper()

            if isinstance(m, ToolMessage):
                content = m.content or ""
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"[{role}] {content}")
            elif isinstance(m, AIMessage):
                content = m.content or ""
                if m.tool_calls:
                    tools = [tc.get("name", "?") for tc in m.tool_calls]
                    lines.append(f"[AI → calls: {', '.join(tools)}]")
                if content:
                    if len(content) > 500:
                        content = content[:500] + "..."
                    lines.append(f"[AI] {content}")
            elif isinstance(m, HumanMessage):
                content = m.content or ""
                if len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"[USER] {content}")
            elif isinstance(m, SystemMessage):
                # Skip system messages in the compressed representation
                # (they're already preserved separately)
                pass

        return "\n".join(lines)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def summary(self) -> str:
        """Return the current rolling summary (may be empty)."""
        return self._summary

    @property
    def max_tokens(self) -> int:
        """Return the token budget."""
        return self._max_tokens

    @property
    def keep_recent(self) -> int:
        """Return the number of recent messages kept intact."""
        return self._keep_recent
