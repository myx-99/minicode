# V3 产品对标改造 — Claude Code 执行提示词

> **目标**：对标 Cursor（Ask/Agent/Plan 模式分离）与 Claude Code（模型驱动 Agent Loop + 用户 opt-in Plan），重构 Claude Code Mini 的意图处理与模式体系。
>
> 将下方 `---PROMPT START---` 到 `---PROMPT END---` 之间的内容**完整复制**交给 Claude Code 执行。

---PROMPT START---

## 任务概述

将 **Claude Code Mini V2.2** 从「默认 Plan + 正则意图分类」架构，升级为 **V3 产品对标架构**：

- **对标 Cursor**：Ask（只读）/ Agent（全工具）/ Plan（先规划再执行）三模式，由用户选择
- **对标 Claude Code**：默认单一 Agent Loop，**模型自主决定**是否/何时调工具，而非入口正则硬分流
- **保留 V2 能力**：Memory、ContextManager、AGENT_STATUS 协议、现有测试基线

请先通读项目：`README.md`、`graph/builder.py`、`graph/routing.py`、`graph/nodes.py`、`memory/project.py`、`cli/app.py`、`config/settings.py`、`tools/registry.py`。

---

## 背景：为什么要改

V2.2 用 `is_direct_answer_query()` 正则规则在 `init_node` 做意图分类，存在：

1. **与成熟产品路径不同** — Cursor/Claude Code 不靠正则猜意图
2. **规则有盲区** — 「分析一下项目架构」等边界 case 难覆盖
3. **默认 Plan 过重** — 简单问题曾误触发多步 plan（BUG-001 用规则补丁缓解）
4. **finish 报告体** — 简单问答不应走编码任务五段总结（已部分修复，需系统化）

V3 核心原则：**简单问题简单答，复杂问题复杂做 —— 靠模式 + 模型，不靠正则预判。**

---

## V3 目标架构

### 1. 三模式产品定义（对标 Cursor）

| 模式 | CLI 命令 | 默认? | 工具权限 | 图结构 | 适用场景 |
|------|----------|-------|----------|--------|----------|
| **ask** | `/mode ask` | 否 | 只读：`read_file`, `grep_search`, `glob_search` | React loop（无 plan/reflect） | 理解代码、问答、探索，**禁止改文件/跑 shell** |
| **agent** | `/mode agent` | **是（新默认）** | 全部 6 工具 | React loop（无 plan/reflect） | 实现功能、修 bug、重构（对标 Cursor Agent + Claude Code 默认） |
| **plan** | `/mode plan` | 否（用户 opt-in） | Plan 阶段只读；Execute 阶段全工具 | Plan-and-Execute + reflect | 多步复杂任务、用户要先审计划 |

**向后兼容映射：**
- 现有 `react` → 重命名为/别名 `agent`（保留 `--mode react` 作为 deprecated alias，至少一个版本周期）
- 现有 `plan` → 行为保留，但**不再是默认**

### 2. 意图处理：从「规则分类」到「模型驱动」

**删除或降级以下 V2.2 机制（不要完全删测试覆盖的历史，可标记 deprecated）：**

- `init_node` 中 `is_direct_answer_query()` → `intent_class=conversational` 的**自动短路**
- `route_after_init` 的 conversational → skip plan 分支（Plan 模式恢复统一 `init → plan → execute`，但 ask/agent 模式无 plan 节点）
- `route_after_execute` 中基于 `intent_class==conversational` 的特殊 finish 路由

**替代方案：**

- **ask/agent 模式**：模型在 execute loop 内自主决定 0 次或 N 次工具调用；无工具时自然文本回复
- **ask 模式工具层硬约束**：registry 不注册 write/edit/shell，从物理上禁止误改（对标 Cursor Ask + Claude Code Plan 沙箱思路）
- **plan 模式**：用户主动选择，表示接受 plan→execute→reflect 重量级流程

`is_direct_answer_query()` / `is_conversational_query()` 可保留为 **Memory 检索优化**（meta query 类似），但**不再驱动图路由**。

### 3. finish 节点：按「是否真正动了代码」决定输出格式

```python
# 目标逻辑（伪代码）
if len(tool_history) == 0:
    # 无工具调用 → 透传 execute 的自然回答（1-3 句），不调用 FINISH LLM
    final_answer = strip_status_block(last_ai_message)
else:
    # 有工具调用 → 结构化编码任务总结（可保留现有 FINISH_SYSTEM_PROMPT）
    final_answer = llm_summarize(...)
```

**所有模式**（ask/agent/plan）统一此规则。不再依赖 `intent_class`。

### 4. 默认模式变更

```python
# config/settings.py
agent_mode: Literal["ask", "agent", "plan"] = Field(default="agent", ...)
# 保留 react 作为 agent 的 alias（解析时 react → agent）
```

CLI banner、README、`main.py --help` 同步更新。

---

## 详细实现要求

### A. ToolRegistry — 模式感知工具集

**文件：** `tools/registry.py`, `agent/agent.py`, `graph/builder.py`

1. 新增 `ToolRegistry.create_for_mode(workspace, mode)` 或在 `create_default` 加 `mode` 参数：
   - `ask` → 仅注册 read_file, grep_search, glob_search
   - `agent` / `react`(alias) / `plan`(execute 阶段) → 全部 6 工具
2. **plan 模式两阶段工具**（推荐方案，对标 Claude Code Plan Mode）：
   - **Phase 1 — planning**：execute 节点绑定只读工具（或 plan_node 独立只读 LLM）
   - **Phase 2 — executing**：用户可见 plan 后，execute 绑定全工具
   - 实现方式任选其一，须在代码注释和文档中说明；优先「plan 完成后切换 tool registry」
3. ask 模式下 LLM 若尝试调用 write/edit/shell → 不应发生（工具未 bind）；加测试验证 schema 中无写工具

### B. Graph Builder — 三模式图

**文件：** `graph/builder.py`

```
ask / agent 图（相同结构，不同 tool registry）:
  START → init → execute ⇄ tools → finish → END
  （无 plan, 无 reflect）

plan 图（保留 V2 Plan-and-Execute）:
  START → init → plan → execute ⇄ tools → reflect → ... → finish → END
  （移除 route_after_init 的 conversational 短路）
```

- `build_graph(..., mode=...)` 接受 `"ask" | "agent" | "plan"`
- mode 切换时 CLI 重建 agent + graph（保留 MemoryManager，已有逻辑）

### C. Nodes — 简化 init / execute / finish

**文件：** `graph/nodes.py`, `prompts/templates.py`, `prompts/system.py`

1. **init_node**：
   - 移除 `intent_class` 自动分类（或固定为 `"agent"`，不再用于路由）
   - 注入 `mode` 到 state
   - Memory 注入逻辑不变

2. **execute_node**：
   - ask 模式使用新模板 `ASK_CONTEXT_TEMPLATE`（强调只读、可探索代码、禁止修改）
   - agent 模式使用现有 `REACT_CONTEXT_TEMPLATE`（改名将 react → agent 术语）
   - plan 模式保持 `STEP_CONTEXT_TEMPLATE`；plan 阶段只读时注入 `PLAN_READONLY_CONTEXT_TEMPLATE`
   - **删除** `CONVERSATIONAL_CONTEXT_TEMPLATE` 作为路由分支（内容可合并进 system prompt 的「简单问题直接答，不要调工具」）

3. **finish_node**：
   - 实现 `_should_pass_through_finish(state) → len(tool_history)==0`
   - 移除对 `intent_class` / `is_direct_answer_query` 的依赖

4. **system.py**：
   - 增加模式说明：「你当前处于 ask/agent/plan 模式，工具权限如下…」
   - 明确：「简单问答无需调工具；只有需要读代码/改代码/跑命令时才调工具」

### D. Routing — 精简

**文件：** `graph/routing.py`

1. **删除** `route_after_init`（plan 模式恢复固定 `init → plan`；ask/agent 固定 `init → execute`）
2. **route_after_execute** 优先级保持：
   ```
   tool_calls → tools
   task_complete → finish
   replan → replan (plan/agent 均可用)
   react/ask: 无信号 → execute（iteration guard）
   plan: 无信号 → reflect
   ```
3. 移除 `intent_class` / `is_direct_answer_query` 相关分支

### E. CLI — 三模式切换

**文件：** `cli/app.py`, `main.py`

1. `/mode ask|agent|plan`（保留 `/mode react` 映射到 agent 并打印 deprecated 警告）
2. Banner 显示当前模式 + 可用工具列表
3. 提示行更新：`(or 'quit', /mode ask|agent|plan, /memory)`

### F. State — 清理

**文件：** `agent/state.py`

- `intent_class` 标记 deprecated 或删除（若无其他引用）
- 可选新增 `plan_phase: "planning" | "executing"` 用于 plan 模式两阶段工具切换

### G. 测试

**必须新增/更新：**

| 测试 | 期望 |
|------|------|
| ask 模式 tool schema | 仅 3 个读工具 |
| ask 模式「读取 main.py 解释」 | 可调 read_file，不可 bind write |
| agent 模式「中国首都是哪里」 | 0 工具，finish 透传简短回答 |
| agent 模式「修复 bug」 | 正常调工具 + 结构化 finish |
| plan 模式编码任务 | 仍走 plan→reflect，行为与 V1 一致 |
| `/mode react` | 映射 agent，警告 deprecated |
| 全量 pytest | 全部通过 |

更新 `tests/test_dual_mode.py` 中与 `intent_class` / conversational 路由相关的测试 — 改为 V3 语义。

### H. 文档

**必须更新（不要新建多余 md，只改已有）：**

1. **`README.md`** — V3 三模式说明、新默认 agent、架构图
2. **`bug/BUG-001-plan-mode-intent-drift.md`** — 追加 §「V3 后续」说明 BUG-001 规则分类已被产品对标方案取代
3. **`report/v2/V2_REPORT.md`** 或新建 **`report/v3/V3_ALIGNMENT_REPORT.md`**（二选一）— 记录 V3 改造方案与验收

在报告/文档中必须包含 **「V3 修复原理」** 章节，说明：
- 为何从正则分类改为模式+模型驱动
- 三模式与 Cursor/Claude Code 的对应关系
- finish 透传的新规则
- 迁移指南（旧 `plan` 默认 → 新 `agent` 默认）

---

## 约束

- **最小破坏**：现有 276+ tests 必须全过（允许更新测试断言以反映 V3 语义）
- **Memory / ContextManager 不得回归**
- **Plan 模式编码任务行为与 V1 一致**（有 plan + reflect 的完整流程）
- **不要**在 V3 再引入新的 LLM 意图分类调用（零额外 API 成本）
- 不修改 git config，不主动 commit
- 代码注释与命名用 `agent` 而非 `react`（react 仅作 alias）

---

## 涉及文件清单

```
config/settings.py          # default mode → agent
main.py                     # --mode ask|agent|plan
cli/app.py                  # /mode 三模式 + banner
tools/registry.py           # mode-aware tool sets
agent/agent.py              # mode 传递 + graph rebuild
agent/state.py              # 清理 intent_class
graph/builder.py            # 三模式图
graph/routing.py            # 移除 intent 路由
graph/nodes.py              # 简化 init/finish，plan 两阶段
prompts/system.py           # 模式感知 system prompt
prompts/templates.py        # ASK模板，清理 CONVERSATIONAL 路由依赖
memory/project.py           # is_direct_answer_query 降级为 memory-only（可选）
tests/test_dual_mode.py     # V3 路由测试
tests/test_graph.py         # 三模式集成测试
README.md                   # 用户文档
report/v3/V3_ALIGNMENT_REPORT.md  # 改造报告（新建）
bug/BUG-001-plan-mode-intent-drift.md  # 追加 V3 后续说明
```

---

## 验收标准（自测）

```bash
# 1. 全量测试
python -m pytest tests/ -q

# 2. 默认 agent 模式 REPL
python main.py
# 期望 banner 显示 Mode: agent

# 3. 简单问答（agent 模式，不应调工具，不应五段报告）
中国首都是哪里
# 期望：简短回答，如「中国的首都是北京。」

# 4. ask 模式只读
/mode ask
读取 main.py 并解释
# 期望：read_file 可调用，无 write/edit/shell

# 5. plan 模式 opt-in
/mode plan
修复 README 里的 typo
# 期望：生成 plan，走 reflect，finish 有结构化总结

# 6. 向后兼容
python main.py --mode react "hello"
# 期望：等同 agent，打印 deprecated 警告
```

---

## 交付物

1. V3 代码 + 测试（276+ pass）
2. `report/v3/V3_ALIGNMENT_REPORT.md`（含修复原理、模式对照表、迁移指南）
3. `README.md` 更新
4. `bug/BUG-001-plan-mode-intent-drift.md` 追加 V3 后续说明
5. 简要改动摘要（5-8 句）

---PROMPT END---

---

## 附：V3 与成熟产品对照（供执行时参考）

| Claude Code Mini V3 | Cursor | Claude Code |
|---------------------|--------|-------------|
| ask | Ask mode | — (无直接对应，只读探索) |
| agent（默认） | Agent mode | 默认 Agent Loop |
| plan（opt-in） | Plan mode | Shift+Tab Plan Mode |
| 模型决定调工具 | 模式决定工具上限 | 模型决定调工具 |
| finish 看 tool_history | 无固定报告模板 | 自然总结 |
| 用户 `/mode` 切换 | Shift+Tab 切换 | Shift+Tab permission mode |
