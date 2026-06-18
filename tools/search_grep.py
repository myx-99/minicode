"""GrepSearchTool — regex-based content search across workspace files."""

import re
import os
from pathlib import Path
from tools.base import BaseTool, ToolResult


# Directories to skip during search
DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".eggs", "build", "dist", ".mypy_cache", ".pytest_cache",
    ".claude", ".idea", ".vscode",
}


class GrepSearchTool(BaseTool):
    """Search file contents using regex patterns (like ripgrep/grep).

    This is the Agent's primary way to locate code, function definitions,
    class names, imports, and error patterns across the project.
    """

    name = "grep_search"
    description = (
        "Search for a regular expression pattern in file contents across the workspace. "
        "Returns matching file paths and the matching lines with line numbers. "
        "Use this to find function/class definitions, imports, error messages, "
        "or any text pattern in the codebase. "
        "Supports glob filtering (e.g., '*.py' for Python files only)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Sub-directory to search in (relative to workspace). Default: workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Filename filter using glob pattern, e.g. '*.py' or '*.{js,ts}'. Default: all files.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case-sensitive. Default: false.",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace):
        self._workspace = workspace

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_sensitive: bool = False,
    ) -> ToolResult:
        try:
            search_root = self._workspace.resolve_path(path)
            if not search_root.exists():
                return ToolResult(
                    success=False,
                    error=f"Search path not found: {path}",
                )

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return ToolResult(
                    success=False,
                    error=f"Invalid regex pattern: {e}",
                )

            results: list[str] = []
            files_searched = 0
            matches_found = 0
            max_results = 200  # cap to avoid overwhelming context

            for root, dirs, files in os.walk(str(search_root)):
                # Skip ignored directories
                dirs[:] = [
                    d for d in dirs if d not in DEFAULT_IGNORE_DIRS
                    and not d.startswith(".")
                ]

                # Apply glob filter
                if glob:
                    import fnmatch
                    files = [f for f in files if fnmatch.fnmatch(f, glob)]

                for fname in files:
                    if matches_found >= max_results:
                        break

                    fpath = os.path.join(root, fname)
                    files_searched += 1

                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            for lineno, line in enumerate(f, 1):
                                if regex.search(line):
                                    relative_path = self._workspace.relative(
                                        Path(fpath)
                                    )
                                    results.append(
                                        f"{relative_path}:{lineno}: {line.rstrip()[:200]}"
                                    )
                                    matches_found += 1
                                    if matches_found >= max_results:
                                        break
                    except (OSError, UnicodeDecodeError):
                        continue

            if not results:
                return ToolResult(
                    success=True,  # "no results" is not a failure
                    output=f"No matches found for pattern '{pattern}' in {files_searched} files.",
                    metadata={
                        "files_searched": files_searched,
                        "matches": 0,
                        "pattern": pattern,
                    },
                )

            truncated = matches_found >= max_results
            header = (
                f"Found {matches_found} match(es) for '{pattern}' "
                f"in {files_searched} files"
            )
            if truncated:
                header += f" (results truncated at {max_results})"

            output = header + "\n" + "\n".join(results)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "files_searched": files_searched,
                    "matches": matches_found,
                    "truncated": truncated,
                    "pattern": pattern,
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {e}",
            )
