# Claude Code Mini — V2 完成报告

> **项目**: Claude Code Mini V2 — Coding Agent with Dual Mode + Memory
> **技术栈**: LangChain + LangGraph + Rich + JSONL
> **报告日期**: 2026-06-03
> **基线**: V1 (154 tests, 6 tools, Plan-and-Execute)

---

## 1. V2 目标达成

### 1.1 三大能力验收

| # | 能力 | 状态 |
|---|------|:--:|
| AC-1 | Plan 模式行为与 V1 一致，154 test 全过 | ✅ |
| AC-2 | React 模式跳过 plan/reflect，LLM 完全自主 | ✅ |
| AC-3 | 自主完成 — `task_complete` 信号直达 finish | ✅ |
| AC-4 | 自主重规划 — `replan` 信号直达 replan（双模式） | ✅ |
| AC-5 | 上下文管理 — ContextManager 替代 MAX_MSG_COUNT=40 截断 | ✅ |
| AC-6 | 长期记忆 — `.agent/memory/entries.jsonl` 持久化 | ✅ |
| AC-7 | 模式切换 — CLI `--mode` / Settings `agent_mode` / REPL `/mode` | ✅ |
| AC-8 | 测试 — +67 new tests (31 dual + 15 context + 21 memory) | ✅ |

### 1.2 测试统计

| 指标 | V1 | V2 | 变化 |
|------|-----|------|------|
| **测试总数** | 154 | 221 | +67 |
| **通过率** | 100% | 100% | - |
| **测试文件** | 5 | 8 | +3 |
| **新增测试文件** | - | test_dual_mode, test_context, test_memory | - |

---

## 2. 新增模块清单

```
memory/                          # 【新增】记忆子系统
├── __init__.py                  # 模块初始化
├── context_manager.py           # ContextManager — 上下文窗口管理（~280行）
├── long_term.py                 # LongTermMemoryKeyword — 关键词检索记忆（~220行）
├── store.py                     # MemoryStore — JSONL 持久化后端（~100行）
└── types.py                     # MemoryEntry + 分类权重（~70行）

graph/
├── routing.py                   # 【新增】信号解析 + V2 双模式路由（~150行）
├── builder.py                   # 【改】双模式图构建
└── nodes.py                     # 【改】init_node 模式感知 + execute_node ContextManager集成 + finish_node 记忆写入

prompts/
├── system.py                    # 【改】双模式行为描述
└── templates.py                 # 【改】+REACT_CONTEXT_TEMPLATE + AGENT_STATUS 协议

agent/
├── state.py                     # 【改】+mode, +memory_context, +context_summary, +messages_token_estimate
└── agent.py                     # 【改】mode 参数，ContextManager/LongTermMemory 集成

config/settings.py               # 【改】+agent_mode, +context_max_tokens, +memory_* 配置

main.py                          # 【改】+--mode, +--no-memory, +--context-max-tokens

cli/app.py                       # 【改】+/mode, +/memory 命令，v2.0.0 banner

tests/
├── test_dual_mode.py            # 【新增】31 tests — 信号解析 + 双模式路由
├── test_context.py              # 【新增】15 tests — 上下文管理
└── test_memory.py               # 【新增】21 tests — 长期记忆
```

---

## 3. 核心机制实现

### 3.1 双模式图结构

**Plan 模式** (default, = V1 + enhanced):
```
START → [init] → [plan] → [execute] ←──────────┐
                              │                  │
                    ┌─────────┴─────────┐        │
                    ▼                   ▼        │
                 [tools]          [route_execute] │
                    │                   │        │
                    │         ┌─────────┼────┐   │
                    │         ▼         ▼    ▼   │
                    │    [reflect]  [replan] [finish]
                    │         │         │        │
                    └──→ execute ←──────┘        │
                              ▲                  │
                              └──────────────────┘
```

**React 模式** (new — free ReAct):
```
START → [init] → [execute] ←─────────────────────┐
                        │                        │
              ┌─────────┴─────────┐              │
              ▼                   ▼              │
           [tools]          [route_execute]      │
              │                   │              │
              │         ┌─────────┼────┐         │
              │         ▼         ▼    ▼         │
              │      [replan]  [finish] [execute]
              │         │         │              │
              └──→ execute ←──────┘              │
                        ▲                        │
                        └────────────────────────┘
```

### 3.2 AGENT_STATUS 信号协议

LLM 在非工具调用回复末尾附加结构化状态块：

```
---AGENT_STATUS---
{"action": "continue" | "step_done" | "task_complete" | "replan", "reason": "..."}
---END_STATUS---
```

| action | Plan 模式 | React 模式 |
|--------|-----------|------------|
| `continue` | → reflect | → execute (继续循环) |
| `step_done` | → reflect | → execute |
| `task_complete` | → finish | → finish |
| `replan` | → replan | → replan |

路由优先级：`tool_calls > task_complete > replan > mode_default > max_iterations`

### 3.3 ContextManager

替换 V1 的 `MAX_MSG_COUNT = 40` 硬截断：

- **Token 预算**: 可配置 `max_tokens`（默认 120K），`reserve_tokens`（默认 8K）
- **保留策略**: 所有 SystemMessage + 最近 20 条原始消息
- **压缩策略**: 超出预算时，旧消息 → LLM 生成 rolling summary → 注入为 SystemMessage
- **工具截断**: 超过 2000 字符的工具结果自动截断并标记 `[truncated]`
- **优雅降级**: 摘要 LLM 调用失败时保留旧摘要

### 3.4 LongTermMemory

**存储模型** — 5 类 MemoryEntry：
- `fact` (w=1.0): 项目事实
- `decision` (w=1.5): 关键决策
- `file_change` (w=1.3): 文件修改记录
- `error_pattern` (w=1.3): 错误模式
- `preference` (w=0.8): 偏好设置

**写入时机**：
- `finish_node` 执行后自动调用 `summarize_session()` → 写入 1-5 条 MemoryEntry
- 自动提取：任务结果、文件变更、错误模式、已完成步骤

**读取时机**：
- `init_node` 启动时用 `task` 查询 top-5 相关记忆 → 注入 `memory_context`
- 可通过 `memory_context` 注入 system prompt / step context

**检索策略** (V2 轻量，无向量DB)：
- Token 交集匹配 + Jaccard 相似度
- 按 category 加权
- 预留 `LongTermMemory` 抽象接口供 V3 向量检索替换

### 3.5 AgentState 扩展

| 新字段 | 类型 | 用途 |
|--------|------|------|
| `mode` | `str` | "plan" / "react" |
| `memory_context` | `str` | 从长期记忆召回的内容 |
| `context_summary` | `str` | 滚动压缩摘要 |
| `messages_token_estimate` | `int` | 当前 token 估算 |

---

## 4. V1 已知问题 → V2 解决方案映射

| V1 问题 | 严重度 | V2 解决 | 状态 |
|---------|:------:|---------|:--:|
| P0-1: LLM 不能自主 finish | P0 | React 模式 + `task_complete` → finish 路由 | ✅ |
| P0-2: Plan 锁死 | P0 | execute → replan 主动快捷路由 | ✅ |
| P1-1: 消息超 40 条丢弃 | P1 | ContextManager rolling summary | ✅ |
| P1-2: Reflect 额外 LLM 开销 | P1 | React 模式跳过 reflect | ✅ |
| 跨会话记忆未实现 | Feature | LongTermMemory + .agent/memory/ | ✅ |

---

## 5. 工程指标

| 指标 | V1 | V2 | 变化 |
|------|-----|------|------|
| **总代码行数** | ~8,300 | ~10,300 | +2,000 |
| **Python 模块数** | 35 | 40 | +5 |
| **核心依赖** | 6 | 6 | 0（无新增依赖） |
| **测试用例** | 154 | 221 | +67 |
| **Graph 节点** | 7 | 7 (plan) / 5 (react) | 动态 |
| **工具数** | 6 | 6 | 不变 |
| **LLM 提供商** | 2 | 2 | 不变 |

---

## 6. 交付物清单

| 交付物 | 文件 | 状态 |
|--------|------|:--:|
| 双模式路由 | `graph/routing.py` | ✅ |
| 上下文管理 | `memory/context_manager.py` | ✅ |
| 长期记忆 | `memory/long_term.py` + `memory/store.py` + `memory/types.py` | ✅ |
| 模式感知 init | `agent/state.py` + `graph/nodes.py` (init_node) | ✅ |
| 记忆写入 finish | `graph/nodes.py` (finish_node) | ✅ |
| React prompt | `prompts/templates.py` (REACT_CONTEXT_TEMPLATE) | ✅ |
| 信号协议 | `prompts/templates.py` + `graph/routing.py` | ✅ |
| V2 配置 | `config/settings.py` | ✅ |
| CLI 参数 | `main.py` + `cli/app.py` | ✅ |
| 双模式测试 | `tests/test_dual_mode.py` (31 tests) | ✅ |
| 上下文测试 | `tests/test_context.py` (15 tests) | ✅ |
| 记忆测试 | `tests/test_memory.py` (21 tests) | ✅ |
| V1 回归测试 | 154 个 V1 测试全部通过 | ✅ |
| 使用说明 | `README.md` | ✅ |
| 完成报告 | `V2_REPORT.md` | ✅ |

---

## 7. 下一步 (V3+)

| 版本 | 功能 | 说明 |
|------|------|------|
| V3 | RAG 代码索引 | 向量检索替代关键词，Chroma/FAISS 集成 |
| V3 | 项目记忆增强 | 基于 embedding 的语义记忆检索 |
| V4 | 多 Agent | 子 Agent 分发，并行工具执行 |
| V4 | MCP 协议 | 外部工具服务器接入 |
| V5 | 生产化 | 插件系统、IDE 集成、指标监控 |

---

> **结论**: Claude Code Mini V2 成功实现了三大核心能力：**双执行模式**（Plan/React）、**上下文窗口管理**（ContextManager）和**项目长期记忆**（LongTermMemory）。所有 V1 已知 P0/P1 问题已解决。221 个测试全部通过。V2 已准备好进入 V3 迭代（RAG + 向量检索）。
