"""WriteFileTool — create or overwrite files in the workspace."""

from tools.base import BaseTool, ToolResult


class WriteFileTool(BaseTool):
    """Create a new file or overwrite an existing one.

    Important: this tool overwrites the entire file. To modify an existing
    file, prefer the edit_file tool which does precise string replacement.
    """

    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing file with new content. "
        "Use this for creating NEW files. For modifying existing files, prefer the "
        "edit_file tool which performs precise string replacement. "
        "This tool will overwrite the entire file — any content not included "
        "in the 'content' parameter will be lost."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path where the file should be created/overwritten (relative to workspace).",
            },
            "content": {
                "type": "string",
                "description": "The complete content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, workspace):
        self._workspace = workspace

    async def execute(self, file_path: str, content: str) -> ToolResult:
        try:
            resolved = self._workspace.resolve_path(file_path)

            # Ensure parent directory exists
            resolved.parent.mkdir(parents=True, exist_ok=True)

            existed = resolved.exists()
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)

            line_count = content.count("\n") + (1 if content else 0)
            action = "Updated" if existed else "Created"

            return ToolResult(
                success=True,
                output=f"{action} file: {self._workspace.relative(resolved)} ({line_count} lines)",
                metadata={
                    "path": self._workspace.relative(resolved),
                    "lines": line_count,
                    "chars": len(content),
                    "existed_before": existed,
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write {file_path}: {e}",
            )
