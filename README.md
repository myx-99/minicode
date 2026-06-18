# Claude Code Mini V3.0

A weekend-buildable Coding Agent powered by LangChain + LangGraph.

```
LLM + Mode-Aware Tools (Ask/Agent/Plan) + Cross-Turn Memory + Model-Driven Agent Loop = Claude Code Mini V3.0
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 3. Run (default: agent mode — model decides tool usage)
python main.py "Fix the bug in main.py"

# 4. Run in ask mode (read-only exploration)
python main.py --mode ask "Explain the project architecture"

# 5. Run in plan mode (Plan-and-Execute, user opt-in)
python main.py --mode plan "Refactor the auth module"
```

## Features

### V3 (new — Product Alignment)
- **Three Modes**: `ask` (read-only, 3 tools) / `agent` (full control, default) / `plan` (Plan-and-Execute, opt-in)
- **Model-Driven Agent Loop**: Agent decides whether and when to use tools — no regex pre-classification
- **Mode-Aware Tool Registry**: `ask` mode physically prevents write/edit/shell calls
- **Smart Finish**: `len(tool_history)==0` → direct answer pass-through; tools used → structured coding summary
- **Backward Compatible**: `--mode react` maps to `agent` with deprecation warning

### V2.1 (baseline)
- **Cross-Turn Memory**: REPL remembers previous tasks in the same session — follow-up questions like "delete what I just did" work
- **Two-Layer Memory**: `SessionMemory` (in-process turn history) + `ProjectMemory` (disk-persisted turns.jsonl)
- **Meta Query Detection**: Auto-detects questions about previous tasks ("刚才完成了什么") and recalls session history
- **Memory Manager**: Unified `MemoryManager` shared across REPL turns — mode switch preserves session

Intent Auditor experiment

### V2 (baseline)
- **LLM Signal Protocol**: Agent can declare task_complete or request replan autonomously
- **Context Window Management**: Token-budget-based rolling summarization — no more "discard after 40 messages"
- **6 Core Tools**: read_file, write_file, edit_file, grep_search, glob_search, shell_execute

### V1 (baseline)
- **Plan + Execute**: LLM decomposes tasks into steps, then executes with tool calling
- **Self-Reflection**: LLM evaluates each step — recovers from errors, retries, or replans
- **Streaming CLI**: Real-time progress display with Rich
- **Workspace Security**: All file ops confined to project root

## Architecture

```
                          ┌─ ask  (read-only) ─── no write/edit/shell ─┐
V3 Three Modes ───────────┼─ agent (default) ──── full 6 tools ────────┤
                          └─ plan  (opt-in) ────── plan → execute → reflect ──┘

Ask / Agent mode graph (model-driven React loop):
    START → [init] → [execute] ⇄ [tools] → [finish] → END
    (Model autonomously decides tool usage. Simple Q&A = 0 tools.)

Plan mode graph (Plan-and-Execute, user opt-in):
    START → [init] → [plan] → [execute] ⇄ [tools] → [reflect] → [finish] → END
```

## Project Structure

```
owncode/
├── agent/          # Agent core + State (V3: mode="ask"|"agent"|"plan")
├── graph/          # LangGraph nodes + builder + routing (V3: three-mode)
├── memory/         # SessionMemory + ProjectMemory + MemoryManager
├── tools/          # 6 tools + mode-aware registry (V3: create_for_mode)
├── prompts/        # System prompt + templates (V3: mode-aware)
├── runtime/        # Workspace management + shell platform detection
├── config/         # Settings + LLM factory (V3: default agent_mode="agent")
├── cli/            # Rich CLI (V3: /mode ask|agent|plan)
├── tests/          # 276 tests
├── main.py         # Entry point (V3: --mode ask|agent|plan)
├── report/v3/      # V3 alignment report
└── requirements.txt
```

## Usage

```bash
# Interactive REPL (default: agent mode)
python main.py

# Interactive REPL (ask mode — read-only)
python main.py --mode ask

# Interactive REPL (plan mode — user reviews plan first)
python main.py --mode plan

# Single task (agent mode — model-driven)
python main.py "Add logging to all modules"

# Single task (ask mode — explore without modifying)
python main.py --mode ask "How does the auth module work?"

# Custom workspace + model
python main.py -w /my/project -m gpt-4o-mini "Fix import errors"

# Disable memory
python main.py --no-memory "Temporary task"

# Custom context budget
python main.py --context-max-tokens 50000 "Read many files"

# Options
python main.py --help
```

### REPL Commands

| Command | Action |
|---------|--------|
| `/mode plan` | Switch to Plan mode (V1 behavior) |
| `/mode react` | Switch to React mode (free ReAct) |
| `/memory` | Show session turn list + project turn count |
| `/memory clear` | Clear both session and project memory |
| `quit`/`exit`/`q` | Exit REPL |

## Run Tests

```bash
pytest tests/ -v
```

**252 tests** covering tools, graph, planner, reflector, replan, dual-mode routing, context management, cross-turn memory, session memory, CLI, and integration.

## What's New in V2.1

| V2 Issue | V2.1 Solution |
|----------|---------------|
| B1: REPL can't reference previous tasks | `SessionMemory` shared across turns in same process |
| B2: "What did I just do" fails | Meta query detection + auto-inject recent session turns |
| B3: Memory entries written with empty summary | `finish_node` writes memory AFTER `final_answer` generation |
| V2 `MemoryEntry` fragments | Unified `TurnRecord` — complete task record per turn |

## What's New in V2

| V1 Issue | V2 Solution |
|----------|-------------|
| P0-1: LLM can't autonomously finish | React mode + `task_complete` signal routing |
| P0-2: Plan is static | execute→replan active routing (both modes) |
| P1-1: Messages > 40 discarded | ContextManager: rolling summarization |
| P1-2: Extra LLM call for reflect | React mode skips reflect entirely |
| Cross-session memory | LongTermMemory + `.agent/memory/` |

## Design Principles

- **Working First, Architecture Second** — MVP runs before any abstraction
- **Simplicity** — 6 tools, 7-5 nodes (plan/react), ~11K lines
- **LangChain + LangGraph** — Production-grade primitives, zero lock-in
- **Extensible** — V2.1-V5 roadmap clear, interfaces clean
- **Backward Compatible** — Plan mode = V1 behavior, all V1 tests still pass

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| V1 | 6 tools + ReAct Loop + Planning + Reflection | ✅ Complete |
| V2 | Dual mode + Context management + Long-term memory | ✅ Complete |
| V2.1 | Cross-turn memory + Session/Project layers + Meta query detection | ✅ Complete |
| V3 | RAG, vector search, embedding-based retrieval | 🔲 Planned |
| V4 | Multi-agent, MCP protocol | 🔲 Planned |
| V5 | Plugin system, IDE integration | 🔲 Planned |

## License

MIT
