"""GlobSearchTool — filename pattern matching across the workspace."""

from pathlib import Path
from tools.base import BaseTool, ToolResult

# Directories to skip during walk
DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".eggs", "build", "dist", ".mypy_cache", ".pytest_cache",
    ".claude", ".idea", ".vscode",
}

# Directories that start with "." are also skipped


class GlobSearchTool(BaseTool):
    """Find files by glob pattern (e.g. '**/*.py', 'src/**/*.ts').

    This is the Agent's primary way to discover project structure — finding
    all Python files, all config files, all test files, etc.
    """

    name = "glob_search"
    description = (
        "Find files matching a glob pattern. Supports '**' for recursive directory "
        "matching. Examples: '**/*.py' (all Python files), 'src/**/*.ts' (TypeScript "
        "files under src/), '*.md' (markdown files in root). "
        "Use this to explore project structure and locate relevant files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match, e.g. '**/*.py'. Supports ** for recursive matching.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (relative to workspace). Default: workspace root.",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace):
        self._workspace = workspace

    async def execute(self, pattern: str, path: str = ".") -> ToolResult:
        try:
            search_root = self._workspace.resolve_path(path)
            if not search_root.exists():
                return ToolResult(
                    success=False,
                    error=f"Search path not found: {path}",
                )
            if not search_root.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Search path is not a directory: {path}",
                )

            results: list[str] = []
            files_found = 0
            max_results = 500

            # Use pathlib's native glob which correctly handles ** (recursive)
            # rglob with a relative pattern does recursive glob
            if "**" in pattern:
                # Remove leading **/ if present — rglob expects a filename pattern
                clean_pattern = pattern
                if pattern.startswith("**/"):
                    clean_pattern = pattern[3:]  # strip "**/"
                matched = search_root.rglob(clean_pattern)
            else:
                matched = search_root.glob(pattern)

            for p in matched:
                if files_found >= max_results:
                    break

                if not p.is_file():
                    continue

                # Skip ignored directories: check if any path part is ignored
                parts = p.relative_to(search_root).parts
                if any(
                    part in DEFAULT_IGNORE_DIRS or part.startswith(".")
                    for part in parts
                ):
                    continue

                rel_str = str(p.relative_to(search_root)).replace("\\", "/")
                results.append(rel_str)
                files_found += 1

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No files matching '{pattern}' found.",
                    metadata={"matches": 0, "pattern": pattern},
                )

            results.sort()
            truncated = files_found >= max_results
            header = f"Found {files_found} file(s) matching '{pattern}'"
            if truncated:
                header += f" (truncated at {max_results})"

            output = header + "\n" + "\n".join(results)

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "matches": files_found,
                    "truncated": truncated,
                    "pattern": pattern,
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Glob search failed: {e}",
            )
