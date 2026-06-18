"""EditFileTool — precise string replacement in files (the EDIT tool is the soul of a Coding Agent)."""

from tools.base import BaseTool, ToolResult


class EditFileTool(BaseTool):
    """Modify a file by replacing an exact string match.

    This is the most critical tool in the Coding Agent — it enables precise,
    surgical edits without rewriting entire files. The LLM must provide the
    exact old_string to match (including whitespace and indentation).
    """

    name = "edit_file"
    description = (
        "Make a precise edit to a file by replacing one string with another. "
        "CRITICAL: the 'old_string' must match EXACTLY — including all whitespace, "
        "indentation, blank lines, and surrounding code. Copy it verbatim from "
        "the file after reading it. If the string appears multiple times, only "
        "the first occurrence is replaced (unless replace_all is true). "
        "Use this tool to modify existing files instead of write_file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit (relative to workspace).",
            },
            "old_string": {
                "type": "string",
                "description": (
                    "The exact text to find and replace. Must match character-for-character "
                    "including indentation, line endings, and surrounding code. "
                    "Read the file first to get the exact text."
                ),
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace ALL occurrences. Default: false (replace only the first).",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(self, workspace):
        self._workspace = workspace

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
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

            with open(resolved, "r", encoding="utf-8") as f:
                original = f.read()

            # Guard: empty old_string
            if not old_string:
                return ToolResult(
                    success=False,
                    error="old_string cannot be empty",
                )

            # Check for match
            count = original.count(old_string)
            if count == 0:
                # Provide a helpful error: show surrounding context
                return ToolResult(
                    success=False,
                    error=(
                        f"old_string not found in {file_path}. "
                        f"The text must match exactly — check whitespace, indentation, "
                        f"and blank lines. Try reading the file again to verify the "
                        f"exact content."
                    ),
                    metadata={"occurrences": 0},
                )

            if replace_all:
                modified = original.replace(old_string, new_string)
                replaced_count = count
            else:
                modified = original.replace(old_string, new_string, 1)
                replaced_count = 1

            with open(resolved, "w", encoding="utf-8") as f:
                f.write(modified)

            plural = "s" if replaced_count > 1 else ""
            return ToolResult(
                success=True,
                output=(
                    f"Edited {self._workspace.relative(resolved)}: "
                    f"replaced {replaced_count} occurrence{plural}"
                ),
                metadata={
                    "path": self._workspace.relative(resolved),
                    "occurrences_found": count,
                    "occurrences_replaced": replaced_count,
                    "replace_all": replace_all,
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to edit {file_path}: {e}",
            )
