"""ToolRegistry — singleton registry for all tools."""

from typing import List, Dict, Optional
from tools.base import BaseTool
from tools.file_read import ReadFileTool
from tools.file_write import WriteFileTool
from tools.file_edit import EditFileTool
from tools.search_grep import GrepSearchTool
from tools.search_glob import GlobSearchTool
from tools.shell import ShellTool


class ToolRegistry:
    """Central registry for managing tool instances.

    Singleton pattern — there should only be one registry per Agent instance.
    Provides registration, lookup, and LangChain-compatible export.

    Usage:
        registry = ToolRegistry.create_default(workspace)
        schemas = registry.get_langchain_tool_schemas()
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    # ── Registration ────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: A BaseTool subclass instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Existing: {type(self._tools[tool.name]).__name__}"
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    # ── Lookup ──────────────────────────────────────────────────

    def get(self, name: str) -> BaseTool:
        """Get a tool by name.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found. Available: {self.tool_names}"
            )
        return self._tools[name]

    def get_all(self) -> List[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> List[str]:
        """Return list of registered tool names."""
        return sorted(self._tools.keys())

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    # ── LLM Integration ─────────────────────────────────────────

    def get_openai_schemas(self) -> List[dict]:
        """Return OpenAI-compatible function-calling schemas for all tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_langchain_tool_dicts(self) -> List[dict]:
        """Alias — return tool dicts suitable for LangChain tool binding."""
        return self.get_openai_schemas()

    def to_langchain_tools(self) -> List:
        """Convert all tools to LangChain StructuredTool objects.

        This enables direct use with LangChain's tool-calling
        and LangGraph's ToolNode.
        """
        from langchain_core.tools import StructuredTool
        import inspect

        structured_tools = []
        for tool in self._tools.values():
            # Create a wrapper function for LangChain
            async def _execute_wrapper(_tool=tool, **kwargs):
                """Wrapper to execute tool and format result for LangChain."""
                result = await _tool.execute(**kwargs)
                return result.to_langchain_message()

            # Build parameter schema from the tool's JSON Schema
            sig = inspect.signature(tool.execute)

            # Create a StructuredTool
            st = StructuredTool(
                name=tool.name,
                description=tool.description,
                func=None,  # sync version
                coroutine=_execute_wrapper,  # async version
                args_schema=_build_pydantic_schema(tool.name, tool.parameters),
            )
            structured_tools.append(st)

        return structured_tools

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def create_default(cls, workspace) -> "ToolRegistry":
        """Create a registry with all 6 V1 tools pre-registered.

        Alias for create_for_mode(workspace, "agent") — backward compatible.

        Args:
            workspace: A Workspace instance (from runtime.workspace).

        Returns:
            ToolRegistry with ReadFile, WriteFile, EditFile, GrepSearch,
            GlobSearch, and Shell tools.
        """
        return cls.create_for_mode(workspace, "agent")

    @classmethod
    def create_for_mode(cls, workspace, mode: str = "agent") -> "ToolRegistry":
        """Create a mode-aware ToolRegistry.

        V3: Three-mode tool sets (aligns with Cursor Ask/Agent/Plan):
          - ask   → 3 read-only tools (read_file, grep_search, glob_search)
          - agent → all 6 tools (default model-driven agent loop)
          - plan  → all 6 tools (Plan-and-Execute with user-visible plan)

        Args:
            workspace: A Workspace instance.
            mode: "ask", "agent", or "plan". "react" is accepted as "agent" alias.

        Returns:
            ToolRegistry with the appropriate tools for the mode.
        """
        # Normalize mode: react → agent (deprecated alias)
        if mode == "react":
            mode = "agent"

        registry = cls()
        # Read-only tools — always registered
        registry.register(ReadFileTool(workspace))
        registry.register(GrepSearchTool(workspace))
        registry.register(GlobSearchTool(workspace))

        # Write tools — ask mode excludes these (physically prevents modification)
        if mode != "ask":
            registry.register(WriteFileTool(workspace))
            registry.register(EditFileTool(workspace))
            registry.register(ShellTool(workspace))

        return registry

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"


def _build_pydantic_schema(name: str, json_schema: dict):
    """Build a Pydantic model from a JSON Schema for StructuredTool args_schema.

    This is a utility to bridge our BaseTool.parameters (JSON Schema dict)
    into LangChain's StructuredTool (which expects a Pydantic model).
    """
    from pydantic import BaseModel, Field, create_model

    properties = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))

    fields = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_type_to_python(prop_schema.get("type", "string"))
        default = ... if prop_name in required else None
        description = prop_schema.get("description", "")
        fields[prop_name] = (python_type, Field(default, description=description))

    model = create_model(
        f"{name.title().replace('_', '')}Args",
        **fields,
    )
    return model


def _json_type_to_python(json_type: str):
    """Map JSON Schema types to Python types for Pydantic."""
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)
