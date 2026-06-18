# Claude Code Mini — V2 实现 Prompt

> **用途**: 将本文档整体作为 Claude Code 的执行指令，在 `owncode/` 目录下完成 V2 迭代。
> **基线**: V1 已打包为 `claude-code-mini-v1.zip`，所有改动基于当前工作区代码。
> **原则**: Working First, Architecture Second — 先跑通，再优化；保持 V1 全部 154 测试通过并新增 V2 测试。

---

## 0. 执行前必读

在开始任何代码改动前，**完整阅读**以下文件以建立上下文：

| 文件 | 目的 |
|------|------|
| `V1_REPORT.md` | V1 能力清单、已知 P0/P1 问题、工程指标 |
| `AUDIT.md` | Agent Loop 审计、P0-1/P0-2 修复方案（§修改清单） |
| `ARCHITECTURE.md` | §8 Memory 设计、§5 LangGraph 设计 |
| `graph/builder.py` | 当前路由逻辑（`route_after_execute` / `route_after_reflect`） |
| `graph/nodes.py` | 7 个节点实现，尤其 `execute_node` 第 255–263 行的消息截断 |
| `agent/state.py` | AgentState 字段 |
| `agent/agent.py` | 入口与 initial_state 构造 |
| `config/settings.py` | 配置扩展点 |
| `main.py` / `cli/app.py` | CLI 参数与交互 |

运行基线测试确认 V1 健康：

```bash
python -m pytest tests/ -v
```

---

## 1. V2 目标（三大能力 + 双模式）

### 1.1 总目标

在 **不破坏 V1 Plan 模式** 的前提下，实现：

1. **上下文窗口管理** — 替换「超过 40 条消息直接丢弃」的粗暴截断
2. **项目长期记忆** — 跨会话持久化项目级知识，新任务自动召回
3. **双执行模式可切换** — Plan 模式（V1）+ 完全 ReAct 模式（LLM 自主宣告完成 / 主动重新规划）

### 1.2 成功标准（Acceptance Criteria）

| # | 能力 | 验收标准 |
|---|------|---------|
| AC-1 | Plan 模式 | `--mode plan`（默认）行为与 V1 一致：init→plan→execute↔tools→reflect→…；154 个现有测试全部通过 |
| AC-2 | ReAct 模式 | `--mode react` 跳过 plan_node（或 plan 为空时走自由循环）；LLM 输出完成信号 → 直达 finish，不经 reflect 强制推步 |
| AC-3 | 自主完成 | ReAct 模式下 LLM 可在任意轮次宣告任务完成（见 §3.2 信号协议） |
| AC-4 | 自主重规划 | 两种模式均支持 LLM 主动触发 replan（ReAct 从 execute 直达 replan；Plan 模式保留 reflect 触发 + execute 主动触发） |
| AC-5 | 上下文管理 | 长对话（>40 条消息）不丢关键上下文；有摘要/压缩日志；token 预算可配置 |
| AC-6 | 长期记忆 | 会话结束后写入 `.agent/memory/`；新会话 init 时自动注入相关记忆；提供 `memory_recall` / `memory_store` 工具或等效 API |
| AC-7 | 模式切换 | CLI `--mode plan\|react`、Settings `agent_mode`、交互 REPL 命令 `/mode react` 三处均可切换 |
| AC-8 | 测试 | 新增 `tests/test_memory.py`、`tests/test_context.py`、`tests/test_dual_mode.py`；全量 pytest 通过 |

---

## 2. 架构设计

### 2.1 新增模块与目录

```
owncode/
├── memory/                          # 【新增】记忆子系统
│   ├── __init__.py
│   ├── context_manager.py           # 上下文窗口管理（压缩/摘要/预算）
│   ├── long_term.py                 # 长期记忆存储与检索
│   ├── store.py                     # 持久化后端（JSON/SQLite，优先轻量 JSON）
│   └── types.py                     # MemoryEntry, ContextBudget 等类型
├── graph/
│   ├── builder.py                   # 【改】双模式图构建 + 新路由
│   ├── nodes.py                     # 【改】节点接入 ContextManager / Memory
│   └── routing.py                   # 【新增】信号解析 + route 函数（从 builder 抽出）
├── tools/
│   ├── memory_recall.py             # 【新增】可选：LLM 主动召回记忆
│   └── memory_store.py              # 【新增】可选：LLM 主动写入记忆
├── prompts/
│   ├── system.py                    # 【改】双模式行为说明
│   └── templates.py                 # 【改】ReAct 模式 prompt + 信号协议
├── agent/
│   ├── state.py                     # 【改】新增 mode, memory_context 等字段
│   └── agent.py                     # 【改】传入 mode，构建对应 graph
├── config/settings.py               # 【改】V2 配置项
├── main.py                          # 【改】--mode 参数
├── cli/app.py                       # 【改】/mode 命令 + 记忆状态展示
└── tests/
    ├── test_memory.py               # 【新增】
    ├── test_context.py              # 【新增】
    └── test_dual_mode.py            # 【新增】
```

持久化目录（运行时生成，不入 git）：

```
<workspace>/.agent/
├── memory/
│   ├── entries.jsonl                # 长期记忆条目
│   └── sessions/                    # 可选：会话摘要
└── CLAUDE.md                        # 【可选 V2.1】项目级指令，init 时注入
```

### 2.2 AgentState 扩展

在 `agent/state.py` 的 `AgentState` 中新增：

```python
# ── Mode ─────────────────────────────────────────────
mode: Literal["plan", "react"]
"""Execution mode: plan = V1 Plan-and-Execute; react = free ReAct loop."""

# ── Memory ───────────────────────────────────────────
memory_context: str
"""Retrieved long-term memory snippets injected at init / before execute."""

context_summary: str
"""Rolling summary of compressed older messages."""

messages_token_estimate: int
"""Approximate token count of current messages (for budget tracking)."""
```

保留所有 V1 字段，确保向后兼容。

### 2.3 双模式图结构

#### Plan 模式（默认，= V1 + 增强）

```
START → [init] → [plan] → [execute] ←──────────────────┐
                              │                         │
                    ┌─────────┴─────────┐               │
                    ▼                   ▼               │
                 [tools]            [route_execute]      │
                    │                   │               │
                    │         ┌─────────┼─────────┐     │
                    │         ▼         ▼         ▼     │
                    │    [reflect]  [replan]  [finish]  │
                    │         │         │               │
                    └──→ execute ←──────┘               │
                              ▲                         │
                              └─────────────────────────┘
```

- `reflect_node` **保留**，步骤级评估逻辑不变
- **新增**: `route_after_execute` 在 Plan 模式下也识别 LLM 主动 `REPLAN` / `TASK_COMPLETE` 信号（见 §3.2），作为 reflect 的快捷路径

#### ReAct 模式（新增）

```
START → [init] → [execute] ←─────────────────────────────┐
                    │                                     │
          ┌─────────┴─────────┐                         │
          ▼                   ▼                         │
       [tools]          [route_execute]                 │
          │                   │                         │
          │         ┌─────────┼─────────┐               │
          │         ▼         ▼         ▼               │
          │      [replan]  [finish]  [execute]          │
          │         │         │      (继续思考)          │
          └──→ execute ←──────┘                         │
                    ▲                                     │
                    └─────────────────────────────────────┘
```

- **跳过** `plan_node` 和 `reflect_node`
- `init_node` 根据 `mode=="react"` 设置 `phase="executing"`，`plan=[]`
- LLM 在 `execute_node` 中完全自主：调工具 / 继续推理 / 宣告完成 / 请求重规划

#### 图构建 API

```python
# graph/builder.py
def build_graph(
    llm: BaseChatModel,
    tool_registry,
    mode: Literal["plan", "react"] = "plan",
    context_manager: ContextManager | None = None,
    long_term_memory: LongTermMemory | None = None,
) -> CompiledGraph:
    ...
```

`ClaudeCodeMini.__init__` 增加 `mode` 参数，传给 `build_graph`。

---

## 3. 核心机制详细规格

### 3.1 上下文窗口管理 (`memory/context_manager.py`)

**替换** `graph/nodes.py` 中 `execute_node` 的硬编码逻辑：

```python
MAX_MSG_COUNT = 40  # 当前 V1 实现 — 必须移除
```

**新设计 — `ContextManager`**:

```python
class ContextManager:
    def __init__(
        self,
        llm: BaseChatModel,
        max_tokens: int = 120_000,      # 可配置，默认保守值
        reserve_tokens: int = 8_000,      # 预留给回复 + 工具 schema
        keep_recent_messages: int = 20,   # 始终保留最近 N 条原始消息
    ): ...

    async def prepare_messages(
        self,
        messages: list[BaseMessage],
        extra_context: str = "",          # step_context / memory_context
    ) -> list[BaseMessage]:
        """
        1. 估算 token（tiktoken 或 len/4 启发式，与 provider 无关即可）
        2. 若未超预算 → 原样返回 + extra_context
        3. 若超限 →
           a. 保留所有 SystemMessage
           b. 保留最近 keep_recent_messages 条
           c. 对其余消息调用 LLM 生成 rolling summary → 写入 context_summary
           d. 将 summary 作为一条 SystemMessage 注入
        4. 返回裁剪后的 invoke_messages
        """
```

**要求**:
- 摘要时**保留**：用户原始任务、已修改文件列表、关键错误、当前 plan 进度
- 工具结果过长时（>2000 字符）在 `tool_node` 或 ContextManager 内截断并标注 `[truncated]`
- 配置项写入 `settings.py`: `context_max_tokens`, `context_keep_recent`
- 新增测试：50+ 条消息后仍能访问任务描述和最近工具结果

### 3.2 LLM 信号协议（双模式共用）

在 `prompts/templates.py` 定义结构化状态块，LLM 在**不调用工具**的文本回复末尾附加：

```
---AGENT_STATUS---
{"action": "continue" | "step_done" | "task_complete" | "replan", "reason": "..."}
---END_STATUS---
```

| action | Plan 模式路由 | ReAct 模式路由 |
|--------|--------------|----------------|
| `continue` | → reflect（评估是否推步） | → execute（继续循环） |
| `step_done` | → reflect | N/A（ReAct 无步骤概念，视为 continue） |
| `task_complete` | → finish | → finish |
| `replan` | → replan | → replan（ReAct 下 replan 可生成轻量 plan 或清空 plan 继续自由执行） |

**实现** `graph/routing.py`:

```python
def parse_agent_status(content: str) -> AgentStatus | None: ...
def route_after_execute(state: AgentState) -> Literal["tools", "reflect", "replan", "finish", "execute"]: ...
```

**路由优先级**（`route_after_execute`）:
1. 有 `tool_calls` → `"tools"`
2. 解析到 `task_complete` → `"finish"`
3. 解析到 `replan` → `"replan"`
4. Plan 模式 + 无信号 / `step_done` / `continue` → `"reflect"`
5. ReAct 模式 + 无信号 / `continue` → `"execute"`（自由循环）
6. 达到 `max_iterations` → 强制 `"finish"` 并写入 error_message

**向后兼容**: 若 LLM 未输出 status block，Plan 模式行为 = V1（进 reflect）；ReAct 模式 = 继续 execute。

### 3.3 项目长期记忆 (`memory/long_term.py`)

**存储模型**:

```python
@dataclass
class MemoryEntry:
    id: str
    content: str
    category: Literal["fact", "decision", "file_change", "error_pattern", "preference"]
    tags: list[str]           # 如 ["auth", "main.py", "bugfix"]
    created_at: str
    session_id: str
    relevance_score: float = 0.0  # 检索时填充
```

**写入时机**（自动，无需用户干预）:
- `finish_node` 执行后：LLM 生成 session summary → 拆分为 1–5 条 MemoryEntry 写入
- `replan_node` 触发时：记录 replan 原因
- 可选：`memory_store` 工具供 LLM 主动写入

**读取时机**:
- `init_node`：用 `task` 作为 query，检索 top-k（默认 k=5）相关记忆 → 填入 `memory_context`
- `execute_node`：将 `memory_context` 注入 step_context / system prompt
- 可选：`memory_recall` 工具供 LLM 主动查询

**检索策略（V2 轻量版，不引入向量 DB）**:
- 关键词匹配 + TF-IDF 或简单 BM25（可用 `rank_bm25` 或自实现）
- 按 category 加权（`file_change` / `decision` 优先）
- 为 V3 向量检索预留 `LongTermMemory.search()` 接口

```python
class LongTermMemory(ABC):
    async def add(self, entry: MemoryEntry) -> None: ...
    async def search(self, query: str, k: int = 5) -> list[MemoryEntry]: ...
    async def summarize_session(self, state: AgentState) -> list[MemoryEntry]: ...
```

**持久化**: `LongTermMemoryJSON` 写 `<workspace>/.agent/memory/entries.jsonl`，append-only，启动时加载到内存索引。

### 3.4 ReAct 模式 Prompt 差异

在 `prompts/templates.py` 新增 `REACT_CONTEXT_TEMPLATE`:

- 不注入 plan / step_index
- 强调：「你是完全自主的 ReAct Agent，自行决定何时完成或重新规划」
- 说明信号协议（§3.2）
- 注入 `memory_context` 和 `task`

Plan 模式继续使用现有 `STEP_CONTEXT_TEMPLATE` / `RETRY_CONTEXT_TEMPLATE`。

---

## 4. 配置与 CLI

### 4.1 `config/settings.py` 新增字段

```python
agent_mode: Literal["plan", "react"] = "plan"
context_max_tokens: int = 120_000
context_keep_recent: int = 20
memory_enabled: bool = True
memory_max_entries: int = 500          # 超出时 LRU 淘汰
memory_search_top_k: int = 5
```

### 4.2 `main.py` 新增参数

```
--mode {plan,react}     执行模式（默认 plan）
--no-memory             禁用长期记忆
--context-max-tokens N  上下文 token 预算
```

### 4.3 交互 REPL（`cli/app.py`）

```
/mode plan|react    切换模式（影响下一次任务）
/memory             显示当前召回的记忆条目
/memory clear       清空长期记忆（需确认）
```

---

## 5. 分阶段实现顺序

按顺序执行，**每阶段结束运行 pytest**：

### Phase 1: 路由与双模式骨架（~2h）

1. 新增 `graph/routing.py`（信号解析 + `route_after_execute` 重构）
2. 扩展 `AgentState`（`mode` 字段）
3. 修改 `graph/builder.py`：`build_graph(mode=...)` 构建不同边集
4. ReAct 模式：`init → execute`，跳过 plan/reflect
5. Plan 模式：保持 V1 图 + 新增 execute→finish/replan 快捷路由
6. 更新 `prompts/templates.py` 加入信号协议
7. 新增 `tests/test_dual_mode.py`（Mock LLM 验证路由）

**Phase 1 完成标志**: Plan 模式 154 测试仍全过；ReAct 模式能完成简单任务并自主 finish。

### Phase 2: 上下文窗口管理（~2h）

1. 实现 `memory/context_manager.py`
2. 在 `execute_node`、`create_reflect_node`、`finish_node` 中接入（替换 MAX_MSG_COUNT 截断）
3. `settings.py` + CLI 参数
4. 新增 `tests/test_context.py`

**Phase 2 完成标志**: 模拟 60+ 消息会话，任务描述和最近工具结果仍可见。

### Phase 3: 长期记忆（~3h）

1. 实现 `memory/types.py`、`memory/store.py`、`memory/long_term.py`
2. `init_node` 召回、`finish_node` 写入
3. 可选工具 `memory_recall` / `memory_store` 注册到 ToolRegistry
4. CLI `/memory` 命令
5. 新增 `tests/test_memory.py`

**Phase 3 完成标志**: 两次独立 `agent.run()` 调用，第二次能引用第一次 session 写入的项目事实。

### Phase 4: 集成 polish（~1h）

1. 更新 `README.md` V2 章节（简短）
2. 创建 `V2_REPORT.md`（能力清单 + 与 V1 对比表）
3. 全量测试 + 手动冒烟：

```bash
# Plan 模式（应与 V1 一致）
python main.py --mode plan "List all Python files in tools/"

# ReAct 模式（自由循环）
python main.py --mode react "Read main.py and summarize what it does"

# 长期记忆（两次会话）
python main.py --mode react "Remember: this project uses LangGraph for orchestration"
python main.py --mode react "What orchestration library does this project use?"
```

---

## 6. 测试要求

### 6.1 必须新增的测试用例

**`tests/test_dual_mode.py`**:
- Plan 模式图包含 plan/reflect 节点；ReAct 模式不包含
- `task_complete` 信号 → finish，不经过 reflect（ReAct）
- `replan` 信号 → replan_node（两种模式）
- 无信号时 Plan→reflect、ReAct→execute

**`tests/test_context.py`**:
- 消息未超预算 → 不压缩
- 消息超预算 → 生成 summary，保留 system + recent
- 工具结果超长截断

**`tests/test_memory.py`**:
- add + search 往返
- JSON 持久化 reload
- init_node 注入 memory_context
- finish_node 写入条目
- memory_enabled=False 时不读写

### 6.2 回归

```bash
python -m pytest tests/ -v --tb=short
```

**不允许**为通过测试而删除或弱化现有断言。若 V1 测试与 V2 行为冲突，优先保证 Plan 模式（默认）与 V1 一致。

---

## 7. 约束与非目标

### 7.1 必须遵守

- **不删除** V1 的 6 个核心工具（read/write/edit/grep/glob/shell）
- **不引入** Docker 沙箱、向量数据库、MCP（留给后续版本）
- **不破坏** `benchmark_runner/` 现有接口（adapter 需兼容新 `mode` 参数，默认 plan）
- 新增依赖控制在 2 个以内（如 `rank_bm25`；tiktoken 可选）
- 代码风格匹配 V1：async node、factory 模式、Rich CLI、Pydantic settings

### 7.2 V2 非目标（明确不做）

- RAG 代码索引 / embedding 检索（V3）
- 多 Agent 协作（V4）
- Session 断点续跑 checkpoint（可 V2.1）
- 并行工具调用

---

## 8. 关键文件改动清单（Checklist）

执行完成后逐项自检：

- [ ] `agent/state.py` — mode, memory_context, context_summary 字段
- [ ] `agent/agent.py` — mode 参数，initial_state，build_graph 传参
- [ ] `graph/routing.py` — 【新】parse_agent_status, route_after_execute
- [ ] `graph/builder.py` — 双模式图，conditional edges 扩展
- [ ] `graph/nodes.py` — 移除 MAX_MSG_COUNT=40；接入 ContextManager + Memory；init/finish 记忆钩子
- [ ] `memory/context_manager.py` — 【新】
- [ ] `memory/long_term.py` — 【新】
- [ ] `memory/store.py` — 【新】
- [ ] `memory/types.py` — 【新】
- [ ] `prompts/templates.py` — REACT_CONTEXT_TEMPLATE + AGENT_STATUS 协议
- [ ] `prompts/system.py` — 双模式说明
- [ ] `config/settings.py` — V2 配置项
- [ ] `main.py` — --mode, --no-memory, --context-max-tokens
- [ ] `cli/app.py` — /mode, /memory 命令
- [ ] `tools/registry.py` — 注册 memory 工具（若实现）
- [ ] `tests/test_dual_mode.py` — 【新】
- [ ] `tests/test_context.py` — 【新】
- [ ] `tests/test_memory.py` — 【新】
- [ ] `README.md` — V2 使用说明（简短）
- [ ] `V2_REPORT.md` — 【新】完成报告

---

## 9. V1 已知问题 → V2 映射

| V1 问题 | V2 解决方案 |
|---------|------------|
| P0-1 假 ReAct（LLM 不能自主 finish） | ReAct 模式 + `task_complete` 信号路由 |
| P0-2 计划锁死 | execute→replan 主动路由（两种模式） |
| P1-1 消息超 40 条丢弃 | ContextManager 摘要压缩 |
| P1-2 reflect 额外 LLM 开销 | ReAct 模式跳过 reflect；Plan 模式保留（V2 不强制合并） |
| 跨会话记忆未实现 | LongTermMemory + `.agent/memory/` |
| 无 CLAUDE.md | 可选：init 时读取 `<workspace>/CLAUDE.md` 注入 system prompt |

---

## 10. 给 Claude Code 的执行指令（直接复制）

```
你是 Claude Code Mini 项目的 V2 实现工程师。工作目录是 owncode/。

请严格按照 V2_IMPLEMENTATION_PROMPT.md 全文执行 V2 迭代：

1. 先阅读 §0 列出的所有文件和 V1_REPORT.md / AUDIT.md
2. 运行 pytest 确认 V1 基线 154 测试通过
3. 按 §5 四个 Phase 顺序实现，每 Phase 结束运行 pytest
4. 实现 §2 架构（memory/ 模块、双模式图、ContextManager、LongTermMemory）
5. 实现 §3 核心机制（信号协议、路由、记忆读写时机）
6. 完成 §6 全部测试 + §8 checklist
7. 创建 V2_REPORT.md 总结改动

硬性要求：
- Plan 模式（默认）必须与 V1 行为一致，154 测试全过
- ReAct 模式必须支持 LLM 自主 task_complete 和 replan
- 替换 execute_node 中 MAX_MSG_COUNT=40 的粗暴截断
- 长期记忆持久化到 <workspace>/.agent/memory/
- 不要引入 Docker、向量 DB、MCP
- 新增依赖 ≤2 个
- 不要 commit，除非我明确要求

开始执行 Phase 1。
```

---

> **文档版本**: V2-Prompt-1.0
> **生成日期**: 2026-06-03
> **基线快照**: `claude-code-mini-v1.zip`
