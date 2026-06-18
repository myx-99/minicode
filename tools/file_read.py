"""ReadFileTool — read file contents with optional offset/limit."""

from pathlib import Path
from tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    """Read a file from the workspace and return its content.

    Supports reading from a specific line offset with a line limit
    to avoid overflowing context with large files.
    """

    name = "read_file"
    description = (
        "Read the contents of a file from the workspace. "
        "Use the 'offset' parameter to start reading from a specific line number "
        "(1-indexed), and 'limit' to read only a certain number of lines. "
        "If the file is large, always use offset and limit to avoid reading "
        "too much at once. "
        "Always read a file before editing it — the Edit tool requires exact "
        "string matching."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative to workspace).",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed). Default: 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. If omitted, reads the entire file.",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, workspace):
        self._workspace = workspace

    async def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = None,
    ) -> ToolResult:
        try:
            resolved = self._workspace.resolve_path(file_path)

            if not resolved.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {file_path}",
                )
            if not resolved.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {file_path}",
                )

            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            start = max(0, offset - 1)

            if start >= total_lines:
                return ToolResult(
                    success=True,
                    output=f"(file has {total_lines} lines, offset {offset} is past end)",
                    metadata={"total_lines": total_lines, "path": str(resolved)},
                )

            if limit is not None:
                selected = all_lines[start : start + limit]
            else:
                selected = all_lines[start:]

            # Format output with line numbers (like cat -n)
            numbered_lines = []
            for i, line in enumerate(selected, start=start + 1):
                numbered_lines.append(f"{i}\t{line.rstrip()}")

            output = "\n".join(numbered_lines)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "total_lines": total_lines,
                    "lines_read": len(selected),
                    "offset": offset,
                    "path": self._workspace.relative(resolved),
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=f"Cannot read {file_path}: file is not UTF-8 encoded (binary file?)",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read {file_path}: {e}",
            )
