"""Base tool abstractions — ToolResult and BaseTool."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Unified return format for all tool executions.

    Every tool must return a ToolResult so the Agent can
    uniformly process success, failure, and metadata.
    """

    success: bool = Field(description="Whether the tool executed successfully")
    output: str = Field(default="", description="Human-readable output on success")
    error: Optional[str] = Field(default=None, description="Error message on failure")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra data (line_count, file_size, exit_code, etc.)",
    )

    def to_summary(self) -> str:
        """Produce a one-line summary for the Agent's observation log."""
        if self.success:
            return f"[OK] {self.output[:120]}"
        return f"[FAIL] {self.error[:120] if self.error else 'Unknown error'}"

    def to_langchain_message(self) -> str:
        """Format the result as a string suitable for LangChain ToolMessage content."""
        if self.success:
            msg = self.output
        else:
            msg = f"Error: {self.error}"
        # Append metadata if non-empty
        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            msg += f"\n[metadata: {meta_str}]"
        return msg


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must define:
      - name: str         — unique tool identifier (e.g. "read_file")
      - description: str  — description shown to the LLM
      - parameters: dict  — JSON Schema for function-calling parameters
      - execute(**kwargs) — the actual tool logic
    """

    name: str
    description: str
    parameters: dict

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters.

        All implementations must:
          1. Validate inputs
          2. Execute the operation
          3. Return a ToolResult (success or failure)
        """
        ...

    def to_openai_schema(self) -> dict:
        """Return OpenAI/Anthropic-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_langchain_tool_dict(self) -> dict:
        """Alias for to_openai_schema — compatible with LangChain tool format."""
        return self.to_openai_schema()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
