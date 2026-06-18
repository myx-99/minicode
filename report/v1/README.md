# Claude Code Mini

A weekend-buildable Coding Agent powered by LangChain + LangGraph.

```
LLM + (ReadFile + WriteFile + EditFile + GrepSearch + GlobSearch + Shell) = Claude Code Mini
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 3. Run
python main.py "Find and fix the bug in main.py"
```

## Features

- **6 Core Tools**: read_file, write_file, edit_file, grep_search, glob_search, shell_execute
- **Plan + Execute**: LLM decomposes tasks into steps, then executes with tool calling
- **Self-Reflection**: LLM evaluates each step — recovers from errors, retries, or replans
- **Streaming CLI**: Real-time progress display with Rich
- **Workspace Security**: All file ops confined to project root

## Architecture

```
START → [init] → [plan] → [execute] ←──────┐
                              │              │
                      ┌───────┴──────┐       │
                      ▼               ▼       │
                   [tools]        [reflect]   │
                      │          ┌──┼──┐      │
                      │     done │ret│ry│replan
                      │          │   │  │      │
                      └→ execute ←┘   │  │      │
                             ▲        ▼  │      │
                             │    [replan] ────┘
                             │        │
                             └────────┘
                                      ▼
                                  [finish] → END
```

## Project Structure

```
owncode/
├── agent/          # Agent core + State
├── graph/          # LangGraph nodes + builder
├── tools/          # 6 tools + registry
├── prompts/        # System prompt + templates
├── runtime/        # Workspace management
├── config/         # Settings + LLM factory
├── cli/            # Rich CLI
├── tests/          # 149 tests
├── main.py         # Entry point
└── ARCHITECTURE.md # Full design doc
```

## Usage

```bash
# Interactive REPL
python main.py

# Single task
python main.py "Add logging to all modules"

# Custom workspace + model
python main.py -w /my/project -m gpt-4o-mini "Fix import errors"

# Options
python main.py --help
```

## Run Tests

```bash
pytest tests/ -v
```

149 tests covering tools, graph, planner, reflector, replan, CLI, and integration.

## Design Principles

- **Working First, Architecture Second** — MVP runs before any abstraction
- **Simplicity** — 6 tools, 7 nodes, ~1500 lines
- **LangChain + LangGraph** — Production-grade primitives, zero lock-in
- **Extensible** — V2-V5 roadmap clear, interfaces clean

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| V1 | 6 tools + ReAct Loop + Planning + Reflection | ✅ Complete |
| V2 | Docker sandbox, session persistence | 🔲 Planned |
| V3 | RAG, project memory, vector search | 🔲 Planned |
| V4 | Multi-agent, MCP protocol | 🔲 Planned |
| V5 | Plugin system, IDE integration | 🔲 Planned |

## License

MIT
