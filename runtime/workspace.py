"""Workspace management — path resolution and security validation."""

import os
from pathlib import Path
from typing import Optional


class Workspace:
    """Manages the project workspace directory.

    Responsibilities:
      - Resolve relative paths to absolute paths inside the workspace
      - Prevent tools from reading/writing outside the workspace
      - Track current working directory for shell execution
    """

    def __init__(self, root_path: str = "."):
        """Initialize workspace.

        Args:
            root_path: Path to the workspace root (absolute or relative).
                       Defaults to current directory.
        """
        self._root = Path(root_path).resolve()
        if not self._root.exists():
            raise ValueError(f"Workspace root does not exist: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {self._root}")

    @property
    def root(self) -> Path:
        """Return the absolute workspace root path."""
        return self._root

    @property
    def root_str(self) -> str:
        """Return the workspace root as a string."""
        return str(self._root)

    def resolve_path(self, file_path: str) -> Path:
        """Resolve a file path relative to the workspace root.

        Rules:
          1. Absolute paths — used as-is, but must be inside workspace
          2. Relative paths — resolved against workspace root
          3. ".." traversal — resolved, then checked

        Raises:
            ValueError: If the resolved path escapes the workspace.
        """
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = self._root / candidate

        resolved = candidate.resolve()

        # Check containment: the resolved path must start with workspace root
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Access denied: '{file_path}' resolves to '{resolved}' "
                f"which is outside the workspace '{self._root}'"
            )

        return resolved

    def validate_safe_path(self, file_path: str) -> Path:
        """Alias for resolve_path — validate and return safe absolute path."""
        return self.resolve_path(file_path)

    def relative(self, absolute_path: Path) -> str:
        """Convert an absolute path back to workspace-relative form for display."""
        try:
            return str(absolute_path.relative_to(self._root))
        except ValueError:
            return str(absolute_path)

    def exists(self, file_path: str) -> bool:
        """Check if a workspace-relative path exists."""
        try:
            resolved = self.resolve_path(file_path)
            return resolved.exists()
        except ValueError:
            return False

    def is_dir(self, file_path: str) -> bool:
        """Check if a workspace-relative path is a directory."""
        try:
            resolved = self.resolve_path(file_path)
            return resolved.is_dir()
        except ValueError:
            return False

    def list_dir(self, dir_path: str = ".") -> list[str]:
        """List entries in a workspace directory (names only)."""
        try:
            resolved = self.resolve_path(dir_path)
            if resolved.is_dir():
                return sorted(
                    [str(p.relative_to(resolved)) for p in resolved.iterdir()]
                )
            return []
        except (ValueError, OSError):
            return []

    def __repr__(self) -> str:
        return f"Workspace(root={self._root})"
