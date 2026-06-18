"""Rich-powered interactive CLI for Claude Code Mini.

Displays real-time streaming output:
  - Agent banner on startup
  - Plan display with step progress
  - Tool call tracking as they happen
  - Error / retry / replan notifications
  - Final summary in a bordered panel

V2.1: Shared MemoryManager across REPL turns for cross-turn recall.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.markdown import Markdown

from agent.agent import ClaudeCodeMini
from config.settings import settings


# ═══════════════════════════════════════════════════════════════════
# Icons & colour helpers
# ═══════════════════════════════════════════════════════════════════

PHASE_ICONS = {
    "init": "🔌", "planning": "🧠", "executing": "⚡",
    "retry": "🔁", "replan": "📋", "done": "✅", "error": "❌",
}

STEP_ICONS = {
    "pending": "⏳", "in_progress": "🔄", "done": "✅", "failed": "❌",
}

TOOL_COLORS = {
    "read_file": "cyan", "write_file": "green", "edit_file": "yellow",
    "grep_search": "magenta", "glob_search": "blue", "shell_execute": "red",
}


def _error_panel(message: str, *, title: str = "Error") -> Panel:
    """Show error text without Rich markup parsing (brackets in message are safe)."""
    return Panel(Text(str(message), style="red"), title=title, border_style="red")


# ═══════════════════════════════════════════════════════════════════
# CLI Application
# ═══════════════════════════════════════════════════════════════════

class AgentCLI:
    """Rich-based interactive CLI for Claude Code Mini.

    V2.1: Shared MemoryManager across REPL turns — agent remembers previous tasks.

    Usage:
        cli = AgentCLI(workspace_path=".", mode="react")
        await cli.run_interactive()       # REPL loop
        await cli.run_task("Fix import")  # single streaming task
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        mode: str = "agent",
        memory_enabled: bool = True,
    ):
        self._console = Console()
        self._workspace_path = workspace_path or settings.workspace_path
        self._mode = mode
        self._memory_enabled = memory_enabled

        # V2.1: Create shared MemoryManager (lives for the entire REPL session)
        self._memory_manager = None
        if memory_enabled:
            from memory.manager import MemoryManager
            ws_root = Path(self._workspace_path).resolve()
            self._memory_manager = MemoryManager(
                workspace_root=ws_root,
                enabled=True,
                session_turns=getattr(settings, "memory_session_turns", 10),
                project_recent=getattr(settings, "memory_project_recent", 5),
                project_search_k=getattr(settings, "memory_search_top_k", 3),
                max_turns=getattr(settings, "memory_max_turns", 200),
            )

        try:
            self._agent = ClaudeCodeMini(
                workspace_path=self._workspace_path,
                mode=self._mode,
                memory_enabled=self._memory_enabled,
                memory_manager=self._memory_manager,
            )
        except Exception as e:
            self._console.print(Text(f"Failed to initialize agent: {e}", style="red"))
            sys.exit(1)

    # ── Public API ────────────────────────────────────────────────

    async def run_interactive(self):
        """Start an interactive REPL loop."""
        self._print_banner()

        while True:
            try:
                task = self._console.input(
                    "\n[bold cyan]Task[/bold cyan] "
                    "[dim](or 'quit'/'exit'/'q', /mode ask|agent|plan, /memory)[/dim]: "
                )
            except (EOFError, KeyboardInterrupt):
                self._console.print("\n[yellow]Goodbye![/yellow]")
                break

            task = task.strip()
            if not task:
                continue
            if task.lower() in ("quit", "exit", "q"):
                self._console.print("[yellow]Goodbye![/yellow]")
                break

            # ── V2.1: REPL commands ─────────────────────────
            if task.startswith("/mode"):
                await self._handle_mode_command(task)
                continue

            if task.startswith("/memory"):
                await self._handle_memory_command(task)
                continue

            await self.run_task(task)

    async def _handle_mode_command(self, task: str):
        """Handle /mode ask|agent|plan command — preserve MemoryManager across switch.

        V3: Supports ask/agent/plan. "react" maps to "agent" with deprecated warning.
        """
        parts = task.split()
        if len(parts) < 2 or parts[1] not in ("ask", "agent", "plan", "react"):
            self._console.print(
                Text("Usage: /mode ask|agent|plan", style="yellow")
            )
            return

        new_mode = parts[1]

        # ── Deprecated alias: react → agent ────────────────
        if new_mode == "react":
            self._console.print(
                Text(
                    "⚠ 'react' is deprecated and will be removed in a future version. "
                    "Use 'agent' instead.",
                    style="yellow",
                )
            )
            new_mode = "agent"

        if new_mode == self._mode:
            self._console.print(
                Text(f"Already in {new_mode} mode.", style="dim")
            )
            return

        self._mode = new_mode
        try:
            # Rebuild agent but KEEP the same MemoryManager (don't lose session turns)
            self._agent = ClaudeCodeMini(
                workspace_path=self._workspace_path,
                mode=self._mode,
                memory_enabled=self._memory_enabled,
                memory_manager=self._memory_manager,
            )
            self._console.print(
                Text(
                    f"Switched to {new_mode} mode. "
                    f"Memory ({len(self._memory_manager.get_session_turns())} turns) preserved.",
                    style="green",
                )
            )
        except Exception as e:
            self._console.print(
                Text(f"Failed to switch mode: {e}", style="red")
            )

    async def _handle_memory_command(self, task: str):
        """Handle /memory [clear] command — show session + project state."""
        parts = task.split()

        if self._memory_manager is None:
            self._console.print(
                Text("Memory is disabled. Enable with --memory flag (remove --no-memory).", style="yellow")
            )
            return

        if len(parts) >= 2 and parts[1] == "clear":
            await self._memory_manager.clear_all()
            self._console.print(
                Text("Session and project memory cleared.", style="green")
            )
            return

        # Display memory status
        session_turns = self._memory_manager.get_session_turns()
        project_count = self._memory_manager.get_project_turn_count()
        session_id = self._memory_manager.session_id

        self._console.print()
        self._console.print(
            Panel(
                "\n".join([
                    f"Mode:       [bold]{self._mode}[/bold]",
                    f"Memory:     [green]enabled[/green]",
                    f"Session ID: [dim]{session_id[:20]}...[/dim]",
                    f"Session turns: [bold]{len(session_turns)}[/bold]",
                    f"Project turns: [bold]{project_count}[/bold]",
                    "",
                    "[dim]Use /memory clear to clear both session and project memory.[/dim]",
                ]),
                title="[bold]Memory Status[/bold]",
                border_style="cyan",
            )
        )

        # Show session turn list
        if session_turns:
            self._console.print("[bold]Session turns:[/bold]")
            for i, t in enumerate(session_turns, 1):
                status_icon = "✅" if t.success else "❌"
                self._console.print(
                    Text(f"  {status_icon} Turn {i} [{t.mode}] ", style="bold")
                    + Text(t.user_task[:100], style="dim")
                )
                if t.files_changed:
                    self._console.print(
                        Text(f"      Files: {', '.join(t.files_changed[:5])}", style="dim")
                    )
                if t.final_answer:
                    self._console.print(
                        Text(f"      Result: {t.final_answer[:120]}", style="dim")
                    )
        else:
            self._console.print(
                Text("No session turns yet. Run a task to populate memory.", style="dim")
            )

    async def run_task(self, task: str):
        """Run a single task and stream its progress to the console."""
        self._console.print()
        self._console.print(
            Panel(Text(task, style="bold white"), title="[bold]Task[/bold]", border_style="cyan")
        )

        tool_count = 0

        try:
            final_state = None
            async for state in self._agent.stream(task):
                final_state = state
                # Detect new tool calls
                history = state.get("tool_history", [])
                new_tools = history[tool_count:]
                for t in new_tools:
                    self._print_tool_call(t)
                tool_count = len(history)

                # Detect plan just generated
                plan = state.get("plan", [])
                phase = state.get("phase", "")

                if phase == "executing" and plan and tool_count == 0:
                    # First time we see the plan in execution
                    self._print_plan(plan, state.get("current_step_index", 0))

            if final_state is not None:
                final = self._agent._result_from_state(final_state)
                self._print_final(final)

        except Exception as e:
            self._console.print(_error_panel(e))

    # ── Rendering helpers ─────────────────────────────────────────

    def _print_banner(self):
        ws = self._agent.workspace
        tools = self._agent.tool_registry.tool_names
        tool_desc = f"[dim]{', '.join(tools)}[/dim]"

        # V3: tool permission hint
        if self._mode == "ask":
            tool_hint = "🔒  [yellow]Read-only[/yellow] — no write/edit/shell tools"
        elif self._mode == "agent":
            tool_hint = "🔓  [green]Full Control[/green] — all 6 tools available"
        else:
            tool_hint = "📋  [cyan]Plan-and-Execute[/cyan] — plan first, then execute"

        mem_info = ""
        if self._memory_manager is not None:
            pt = self._memory_manager.get_project_turn_count()
            mem_info = f"\n🧠  Memory:     [green]enabled[/green] ([dim]{pt} project turns on disk[/dim])"
        else:
            mem_info = "\n🧠  Memory:     [yellow]disabled[/yellow]"

        banner = Panel(
            "\n".join([
                "🤖  [bold cyan]Claude Code Mini[/bold cyan]  v3.0.0",
                "    [dim]Three Modes — Model-Driven Agent Loop[/dim]",
                "",
                f"📁  Workspace: [green]{ws.root_str}[/green]",
                f"🎯  Mode:       [bold]{self._mode}[/bold]",
                f"    {tool_hint}",
                f"🔧  Tools:      {tool_desc}",
                mem_info,
                "",
                "[dim]/mode ask|agent|plan to switch • /memory to view state • quit to exit[/dim]",
            ]),
            border_style="cyan",
            box=box.ROUNDED,
        )
        self._console.print(banner)

    def _print_plan(self, plan: List[dict], current_idx: int):
        """Print the plan with step status icons."""
        if not plan:
            return

        self._console.print()
        self._console.print("[bold]Plan:[/bold]")
        for s in plan:
            sid = s.get("id", "?")
            status = s.get("status", "pending")
            desc = s.get("description", "")
            retry = s.get("retry_count", 0)
            is_now = (
                str(sid) == str(current_idx + 1)
                if plan and current_idx < len(plan)
                else False
            )

            icon = STEP_ICONS.get(status, "❓")

            style = ""
            if status == "done":
                style = "green"
            elif status == "failed":
                style = "red"
            elif is_now:
                style = "yellow"

            line = Text(f"  {icon} ")
            label = f"Step {sid}: "
            if style:
                line.append(label, style=style)
            else:
                line.append(label)
            line.append(desc)
            if retry > 0:
                line.append(f" (retry {retry})", style="dim")
            if is_now:
                line.append(" ◀", style="bold yellow")
            self._console.print(line)

    def _print_tool_call(self, tool: dict):
        """Print a single tool invocation and its result."""
        tool_name = tool.get("tool", "?")
        args = tool.get("args", {})
        success = tool.get("success", True)
        result = str(tool.get("result", ""))[:200].replace("\n", " ")

        color = TOOL_COLORS.get(tool_name, "white")
        status_icon = "✓" if success else "✗"
        status_color = "green" if success else "red"

        args_str = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items())

        line = Text(f"  {status_icon} ", style=status_color)
        line.append(tool_name, style=color)
        line.append(f" ({args_str})", style="dim")
        self._console.print(line)

    def _print_final(self, result: Dict[str, Any]):
        """Print the final task result."""
        success = result.get("success", False)
        final_answer = result.get("final_answer", "")
        error_message = result.get("error_message", "")
        plan = result.get("plan", [])

        # Re-print plan with final statuses (could have changed from stream)
        if plan:
            self._print_plan(plan, result.get("current_step_index", -1))

        self._console.print()

        if error_message and not success:
            self._console.print(Text(f"Note: {error_message[:300]}", style="yellow"))
            self._console.print()

        border_style = "green" if success else "red"
        title = "✅ Task Complete" if success else "❌ Task Failed"

        self._console.print(
            Panel(
                Markdown(final_answer) if final_answer else Text("(no output)", style="dim"),
                title=title,
                border_style=border_style,
            )
        )


# ═══════════════════════════════════════════════════════════════════
# Standalone result printer
# ═══════════════════════════════════════════════════════════════════

def print_result(result: Dict[str, Any], console: Optional[Console] = None):
    """Format and print an agent.run() result dict.

    Args:
        result: The dict returned by agent.run().
        console: Optional Rich Console (auto-creates if None).
    """
    if console is None:
        console = Console()

    success = result.get("success", False)
    final_answer = result.get("final_answer", "")
    plan = result.get("plan", [])
    tool_history = result.get("tool_history", [])
    error_message = result.get("error_message", "")

    # Plan table
    if plan:
        plan_table = Table(title="Plan Execution", box=box.SIMPLE, padding=(0, 2))
        plan_table.add_column("Step", style="dim")
        plan_table.add_column("Description")
        plan_table.add_column("Status")
        for s in plan:
            icon = STEP_ICONS.get(s.get("status", ""), "?")
            plan_table.add_row(
                s.get("id", "?"),
                s.get("description", "")[:70],
                f"{icon} {s.get('status', '?')}",
            )
        console.print(plan_table)

    # Tool summary
    if tool_history:
        console.print(f"\n[dim]Tools used: {len(tool_history)}[/dim]")
        for t in tool_history[-10:]:
            success_t = t.get("success", True)
            color = "green" if success_t else "red"
            dot = "●" if success_t else "✗"
            tool_name = t.get("tool", "?")
            tool_color = TOOL_COLORS.get(tool_name, "white")
            args_short = str(t.get("args", {}))[:70]
            line = Text(f"  {dot} ", style=color)
            line.append(tool_name, style=tool_color)
            line.append(f" {args_short}", style="dim")
            console.print(line)

    # Result panel
    border = "green" if success else "red"
    title = "✅ Task Complete" if success else "❌ Task Failed"

    if error_message:
        final_answer += f"\n\n_Note: {error_message}_"

    console.print()
    console.print(
        Panel(
            Markdown(final_answer) if final_answer else Text("(no output)", style="dim"),
            title=title,
            border_style=border,
        )
    )
