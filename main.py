#!/usr/bin/env python3
"""Claude Code Mini — main entry point.

A weekend-buildable Coding Agent powered by LangChain + LangGraph.

Usage:
    python main.py                          # Interactive mode
    python main.py "Fix the import bug"     # Single task
    python main.py --workspace /path        # Custom workspace
    python main.py --help                   # Show options
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Load environment variables from .env before anything else
from dotenv import load_dotenv
load_dotenv()

from cli.app import AgentCLI, print_result
from agent.agent import ClaudeCodeMini
from config.settings import settings
from rich.console import Console
from rich.text import Text


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Code Mini — A weekend-buildable Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Interactive REPL
  python main.py "Add logging to all files"   Run a single task
  python main.py -w /my/project               Use custom workspace
  python main.py -m gpt-4o-mini               Use a different model
  python main.py --max-iters 50               Increase iteration limit
        """,
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task to execute (omit for interactive mode)",
    )
    parser.add_argument(
        "-w", "--workspace",
        default=None,
        help="Project workspace directory (default: current directory)",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Override LLM model (e.g., gpt-4o, claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--mode",
        choices=["ask", "agent", "plan", "react"],
        default=None,
        help="Execution mode: ask (read-only), agent (default), plan (Plan-and-Execute). "
             "'react' is a deprecated alias for 'agent'.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Maximum ReAct loop iterations (default: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries per step (default: 2)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable long-term memory (V2)",
    )
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        default=None,
        help="Context window token budget (V2, default: 120000)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw result JSON instead of formatted display",
    )
    return parser.parse_args()


def _resolve_mode(mode: str | None) -> str:
    """Resolve mode, warning on deprecated 'react' alias."""
    if mode is None:
        return settings.agent_mode
    if mode == "react":
        import sys
        print("⚠ 'react' is deprecated — use 'agent' instead.", file=sys.stderr)
        return "agent"
    return mode


async def run_single_task(args):
    """Execute a single task and print results."""
    console = Console()

    workspace_path = args.workspace or settings.workspace_path
    mode = _resolve_mode(args.mode)

    try:
        agent = ClaudeCodeMini(
            workspace_path=workspace_path,
            mode=mode,
            max_iterations=args.max_iters,
            max_retries_per_step=args.max_retries,
            memory_enabled=not args.no_memory,
            context_max_tokens=args.context_max_tokens or settings.context_max_tokens,
        )

        if args.raw:
            result = await agent.run(args.task)
            import json
            console.print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Use CLI streaming with memory support
            cli = AgentCLI(
                workspace_path=workspace_path,
                mode=mode,
                memory_enabled=not args.no_memory,
            )
            await cli.run_task(args.task)

    except Exception as e:
        console.print(Text(f"Fatal error: {e}", style="red"))
        sys.exit(1)


async def run_interactive(args):
    """Launch the interactive REPL."""
    mode = _resolve_mode(args.mode)
    cli = AgentCLI(
        workspace_path=args.workspace,
        mode=mode,
        memory_enabled=not getattr(args, "no_memory", False),
    )
    await cli.run_interactive()


def main():
    """Entry point."""
    args = parse_args()

    if args.model:
        settings.openai_model = args.model
        settings.anthropic_model = args.model

    if args.task:
        asyncio.run(run_single_task(args))
    else:
        asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
