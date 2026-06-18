"""Shell platform detection and command guidance for agents and tools."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class ShellEnvironment:
    """Runtime shell facts for the machine running the agent."""

    system: str
    shell_name: str
    is_windows: bool

    @property
    def tool_description_suffix(self) -> str:
        """Extra lines appended to shell_execute tool description for the LLM."""
        common = (
            f"\n\n## Execution environment\n"
            f"- OS: {self.system}\n"
            f"- Shell: {self.shell_name}\n"
            f"- Working directory: workspace root (do not `cd /workspace` — that path does not exist here)\n"
        )
        if self.is_windows:
            return common + (
                "\n## Windows command rules (required)\n"
                "- Use **cmd.exe** syntax, not bash.\n"
                "- Pipe stdin: `echo 5 3 8 | python script.py` — never use `<<<`.\n"
                "- Find executables: `where python` (not `which`).\n"
                "- List files: `dir` or `dir /s` (not `ls` unless you know it exists).\n"
                "- Run Python: prefer `python` or `py`; test with `python --version`.\n"
                "- Avoid: `/workspace`, `2>/dev/null`, `$(...)`, heredocs `<<`, `pwd` (use `cd` with no args).\n"
                "- Chain commands: `command1 && command2` is OK in cmd.\n"
            )
        return common + (
            "\n## Unix command rules\n"
            "- Shell is **bash** (`/bin/bash`).\n"
            "- Pipe stdin: `echo \"5 3 8\" | python script.py` or heredoc where appropriate.\n"
            "- Find executables: `which python3`.\n"
        )

    @property
    def agent_prompt_section(self) -> str:
        """Bullet list for system / planning prompts."""
        if self.is_windows:
            return (
                f"- **Host OS**: Windows ({self.shell_name})\n"
                f"- Commands run in the **workspace root** via cmd.exe — use Windows syntax only.\n"
                f"- Feed script input: `echo 5 3 8 1 9 | python sort_numbers.py` (not `<<<`).\n"
                f"- Do not use Linux-only paths like `/workspace`, or bash-only operators like `<<<`, `2>/dev/null`.\n"
                f"- Locate Python: `where python` or `python --version`.\n"
            )
        return (
            f"- **Host OS**: {self.system} ({self.shell_name})\n"
            f"- Commands run in the **workspace root** via bash.\n"
            f"- Use standard Unix/bash syntax for pipes, redirects, and `which`.\n"
        )


def get_shell_environment() -> ShellEnvironment:
    """Detect the shell environment for the current process."""
    system = platform.system()
    if system == "Windows":
        return ShellEnvironment(
            system=system,
            shell_name="cmd.exe (Windows)",
            is_windows=True,
        )
    return ShellEnvironment(
        system=system,
        shell_name="bash (/bin/bash)",
        is_windows=False,
    )
