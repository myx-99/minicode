# V3 产品对标改造报告

> **日期**: 2026-06-04
> **从**: Claude Code Mini V2.2
> **到**: Claude Code Mini V3.0.0
> **对标**: Cursor (Ask/Agent/Plan) + Claude Code (模型驱动 Agent Loop)

---

## 1. 背景

V2.2 采用「默认 Plan + 正则意图分类」架构，存在以下问题：

1. **与成熟产品路径不同** — Cursor/Claude Code 靠模式切换 + 模型自主决策，不靠正则猜意图
2. **规则有盲区** — `is_direct_answer_query()` 正则难以覆盖「分析一下项目架构」等边界 case
3. **默认 Plan 过重** — 简单问题曾误触发多步 plan（BUG-001）
4. **finish 报告体** — 简单问答不应走编码任务五段总结

V3 核心原则：**简单问题简单答，复杂问题复杂做 — 靠模式 + 模型，不靠正则预判。**

---

## 2. V3 修复原理

### 2.1 从「正则分类」到「模式+模型驱动」

**删除/降级的 V2.2 机制：**

| 机制 | 文件 | 处置 |
|------|------|------|
| `init_node` 中 `is_direct_answer_query()` → `intent_class=conversational` | `graph/nodes.py` | 删除，不再做自动分类 |
| `route_after_init` conversational → skip plan 分支 | `graph/routing.py` | 删除整个函数 |
| `route_after_execute` 中 intent_class==conversational 特殊路由 | `graph/routing.py` | 删除 |
| `_should_pass_through_finish` 依赖 `intent_class` | `graph/nodes.py` | 改为 `len(tool_history)==0` |

`is_conversational_query()` / `is_direct_answer_query()` 保留在 `memory/project.py` 中，仅用于 Memory 检索优化，**不再驱动图路由**。

### 2.2 三模式产品定义（对标 Cursor）

| 模式 | CLI | 默认? | 工具权限 | 图结构 | 对标 |
|------|-----|-------|----------|--------|------|
| **ask** | `/mode ask` | 否 | 只读 3 工具 | React loop | Cursor Ask |
| **agent** | `/mode agent` | **是** | 全部 6 工具 | React loop | Cursor Agent + Claude Code 默认 |
| **plan** | `/mode plan` | 否 (opt-in) | 全部 6 工具 | Plan-and-Execute | Cursor Plan |

### 2.3 模型驱动工具使用（对标 Claude Code）

- **ask/agent 模式**：模型在 execute loop 内自主决定 0 次或 N 次工具调用；无工具时自然文本回复
- **ask 模式工具层硬约束**：`ToolRegistry.create_for_mode("ask")` 不注册 write/edit/shell，物理禁止误改
- **plan 模式**：用户主动选择，接受 plan→execute→reflect 重量级流程

### 2.4 finish 节点：按 tool_history 决定输出格式

```
if len(tool_history) == 0:
    final_answer = agent_response (pass through, no LLM re-summarization)
else:
    final_answer = llm.summarize(plan, tools, agent_response)
```

所有模式统一此规则，不再依赖 `intent_class`。

---

## 3. 关键改动

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `report/v3/V3_ALIGNMENT_REPORT.md` | 本报告 |

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| `config/settings.py` | `agent_mode` 默认 "agent"，字面量 `Literal["ask","agent","plan"]` |
| `tools/registry.py` | 新增 `create_for_mode(workspace, mode)` — ask 3 工具，agent/plan 6 工具 |
| `agent/state.py` | `intent_class` 标记 deprecated；新增可选 `plan_phase` |
| `prompts/system.py` | 三模式感知 system prompt，模型主导工具使用 |
| `prompts/templates.py` | 新增 `ASK_CONTEXT_TEMPLATE`；更新 `REACT_CONTEXT_TEMPLATE` 到 agent 术语；`CONVERSATIONAL_CONTEXT_TEMPLATE` 保留为模板但不再用作路由分支 |
| `graph/routing.py` | 删除 `route_after_init`；新增 `normalize_mode()`；`route_after_execute` 移除 intent_class 分支，agent/ask→execute 循环 |
| `graph/nodes.py` | `init_node` 删除 intent 分类；`execute_node` 新增 ask/agent 模式分支；`finish_node` 透传规则 `len(tool_history)==0` |
| `graph/builder.py` | 三模式图：ask/agent=React loop，plan=Plan-and-Execute；删除 BUG-001 conversational 短路 |
| `agent/agent.py` | 接受 ask/agent/plan；`create_for_mode` 模式感知工具；react→agent 归一化 |
| `cli/app.py` | `/mode ask|agent|plan` + react deprecated 警告；V3 banner |
| `main.py` | `--mode ask|agent|plan|react` + deprecation warning |
| `tests/test_dual_mode.py` | 重写为 V3 三模式测试（48 项） |
| `tests/test_graph.py` | 更新为 V3 语义 |
| `tests/test_phase3.py` | `build_graph(..., mode="plan")` |
| `tests/test_integration.py` | `mode="plan"` |
| `README.md` | V3 更新（见第 5 节） |
| `bug/BUG-001-plan-mode-intent-drift.md` | 追加 V3 后续（见第 6 节） |

---

## 4. 迁移指南

### 4.1 模式变化

```
V2.2:  default plan, 可选 react
V3:    default agent, 可选 ask/plan, react 映射到 agent (deprecated)
```

### 4.2 行为变化

| 场景 | V2.2 | V3 |
|------|------|-----|
| "中国首都是哪里" | plan 模式: 正则检测→跳过 plan→直接回答 | agent 模式: 模型自主决定不调工具→直接回答 |
| "你是谁" | plan 模式: 正则检测→跳过 plan→直接回答 | agent 模式: 模型自主决定不调工具→直接回答 |
| "读取 main.py 并解释" | plan 模式: 生成 plan→execute | agent 模式: 模型调 read_file→回答；plan 模式: plan→execute |
| 简单问答 finish | 透传 (V2.2 fix) | 透传 (tool_history==0 规则) |
| 编码任务 finish | LLM 结构化总结 | LLM 结构化总结 (tool_history > 0) |

### 4.3 API 变化

```python
# V2.2
agent = ClaudeCodeMini(mode="react")   # react mode
agent = ClaudeCodeMini(mode="plan")    # plan mode (default)

# V3
agent = ClaudeCodeMini(mode="agent")   # agent mode (default, 替代 react)
agent = ClaudeCodeMini(mode="ask")     # ask mode (read-only, 新增)
agent = ClaudeCodeMini(mode="plan")    # plan mode (opt-in)
agent = ClaudeCodeMini(mode="react")   # 仍然可用 → 映射到 agent + deprecated warning
```

---

## 5. README 更新

V3 README 已更新（见 `README.md`）：
- 版本号 v2.1.0 → v3.0.0
- 三模式架构图
- `/mode ask|agent|plan` 使用说明

---

## 6. BUG-001 后续

V3 已将 BUG-001 的正则分类方案替换为产品对标方案（见 `bug/BUG-001-plan-mode-intent-drift.md` §V3 后续说明）。

---

## 7. 测试

```
276 passed, 0 failed in 7.95s
```

- 新增测试: ask tool schema, three-mode graph construction, normalize_mode, ask/agent routing
- 移除测试: route_after_init, intent_class conversational routing (V2.2-specific)
- 更新测试: 所有需要 plan 模式的测试显式传 `mode="plan"`

---

## 8. 验收

- [x] ask 模式仅 3 个只读工具
- [x] agent 模式默认 + 模型自主决定工具使用
- [x] plan 模式保留完整 Plan-and-Execute 流程
- [x] `/mode react` 映射 agent + deprecated warning
- [x] `finish_node` 按 `tool_history==0` 透传，不依赖 intent_class
- [x] 276 个测试全部通过
- [x] Memory / ContextManager 无回归
- [x] 零额外 LLM 调用引入
- [x] 向后兼容: `--mode react` 仍可用（带 deprecated 警告）
