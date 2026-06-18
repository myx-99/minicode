"""ShellTool — execute shell commands in the workspace."""

import asyncio
import platform
from tools.base import BaseTool, ToolResult
from runtime.shell_platform import get_shell_environment

_SHELL_BASE_DESCRIPTION = (
    "Execute a shell command in the workspace directory and return its output "
    "(stdout and stderr combined). "
    "Use this tool to: run tests, install packages, start/stop servers, "
    "run git commands, check file properties, or any other command-line operation. "
    "Commands have a timeout to prevent hanging — avoid commands that run "
    "indefinitely (like starting a server without backgrounding). "
    "The command's working directory is already the workspace root."
)


class ShellTool(BaseTool):
    """Execute shell commands and capture output.

    This is how the Agent installs dependencies, runs tests, starts servers,
    and performs any operation that requires the command line.
    """

    name = "shell_execute"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 60, max: 300).",
            },
        },
        "required": ["command"],
    }

    # Commands that will be blocked for safety
    BLOCKED_COMMANDS = [
        "rm -rf /",
        "mkfs.",
        "dd if=",
        ":(){ :|:& };:",  # fork bomb
    ]

    def __init__(self, workspace):
        self._workspace = workspace
        self._shell_env = get_shell_environment()
        self.description = (
            _SHELL_BASE_DESCRIPTION + self._shell_env.tool_description_suffix
        )

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        try:
            # Basic safety check
            for blocked in self.BLOCKED_COMMANDS:
                if blocked in command:
                    return ToolResult(
                        success=False,
                        error=f"Blocked dangerous command (matched pattern: '{blocked}')",
                    )

            # Apply timeout cap
            timeout = min(max(timeout, 1), 300)

            cwd = self._workspace.root_str
            if self._shell_env.is_windows:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    executable="/bin/bash",
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout}s: {command[:80]}...",
                    metadata={"timeout": timeout, "exit_code": None},
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

            exit_code = process.returncode

            # Combine stdout and stderr
            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")

            output = "\n".join(parts) if parts else "(no output)"

            success = exit_code == 0
            status = "OK" if success else f"FAIL (exit={exit_code})"

            return ToolResult(
                success=success,
                output=f"[{status}] {command}\n{output}",
                metadata={
                    "exit_code": exit_code,
                    "stdout_len": len(stdout_bytes),
                    "stderr_len": len(stderr_bytes),
                    "shell": self._shell_env.shell_name,
                    "platform": platform.system(),
                },
            )

        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Command execution failed: {e}",
            )
