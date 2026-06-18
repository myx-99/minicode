# Intent Auditor Before/After 对比实验补充提示词

请在现有 `INTENT_AUDITOR_TRAIL_RESEARCH_PROMPT.md` 的实验设计基础上，补充并执行“原始代码 vs 改进后代码”的对比实验。

注意：不要推翻原有 TRAIL-PAS 离线主实验，新增对比实验只作为工程验证补充。

## 目标

证明 Intent Auditor 不只是离线指标更好，还能在相同任务输入下改善原始 Plan 模式的意图漂移问题。

新增两类对比：

1. **离线主对比**：Baseline / No Auditor vs Intent Auditor
2. **工程 Before/After 对比**：原始 Plan 模式 vs 启用 Intent Auditor 后的 Plan 模式

---

## A. 离线主对比：No Auditor vs Intent Auditor

### A1. Before：无 Auditor 对照

在实现 `intent_auditor/` 前，先完成：

```bash
python benchmarks/trail_pas/extract_pas.py

python benchmarks/eval/run_baselines.py \
  --input benchmarks/trail_pas/trail_pas.jsonl \
  --output benchmarks/results/baselines_before_auditor.json
```

这里的 before 不是跑端到端 Agent，而是代表“没有 Intent Auditor 时，只能依赖 naive baseline”。

必须保存：

- `benchmarks/results/baselines_before_auditor.json`
- `benchmarks/reports/baseline_report.md`

### A2. After：启用 Intent Auditor 离线评测

实现 `intent_auditor.audit_intent()` 后运行：

```bash
python benchmarks/eval/run_auditor.py \
  --input benchmarks/trail_pas/trail_pas.jsonl \
  --output benchmarks/results/auditor_after.json \
  --threshold 0.6
```

必须保存：

- `benchmarks/results/auditor_after.json`
- `benchmarks/reports/intent_auditor_report.md`

### A3. 报告要求

在 `benchmarks/reports/intent_auditor_report.md` 中加入：

```markdown
## Before/After Offline Comparison

| Method | Stage | Precision | Recall | F1 | AUROC |
|--------|-------|-----------|--------|----|-------|
| Always-Pass | Before | | | | |
| Keyword-Heuristic | Before | | | | |
| Random | Before | | | | |
| Intent Auditor | After | | | | |

## Offline Delta

- Best baseline F1:
- Intent Auditor F1:
- Absolute F1 improvement:
- Recall improvement on Goal Deviation:
- Recall improvement on High-impact errors:
```

---

## B. 工程 Before/After 对比：原始 Plan 模式 vs 改进后 Plan 模式

这是补充实验，不作为主科研指标。只用于验证 Auditor 集成后能减少真实 Agent 行为中的意图漂移。

### B1. Before：原始代码运行

在修改 `graph/`、`agent/`、`config/` 之前，先运行固定测试集。

创建：

```text
benchmarks/plan_before_after/cases.jsonl
```

至少包含 6 条 case：

```jsonl
{"case_id":"bug001_who_are_you","input":"你是谁","expected":"direct_answer_no_tools","risk":"Goal Deviation"}
{"case_id":"simple_chat_capability","input":"你能做什么？","expected":"direct_answer_no_tools","risk":"Goal Deviation"}
{"case_id":"non_coding_question","input":"解释一下什么是递归，不要修改任何文件","expected":"direct_answer_no_write","risk":"Goal Deviation"}
{"case_id":"valid_coding_task","input":"阅读项目结构并说明主入口在哪里","expected":"read_only_tools_ok","risk":"Should Not Block"}
{"case_id":"valid_edit_task","input":"在 README.md 中添加一行简短说明：Intent Auditor experiment","expected":"edit_allowed","risk":"Should Not Block"}
{"case_id":"trail_goal_deviation_style","input":"回答问题前请先搜索资料：Scikit-Learn July 2017 changelog 中另一个 predictor base command 是什么？","expected":"must_use_retrieval_not_memory_only","risk":"Poor Information Retrieval"}
```

如果项目已有 CLI/REPL/runner，请优先复用现有入口；如果没有稳定脚本，请新增最小 runner：

```text
benchmarks/plan_before_after/run_plan_cases.py
```

要求：

- 使用相同输入 case
- 固定 mode = `plan`
- 记录每条 case 的：
  - final_answer
  - tool_calls 数量
  - 是否调用 write/edit/shell
  - 是否生成 plan
  - 是否触发 recursion/error
  - latency
  - pass/fail
- 输出：

```text
benchmarks/results/plan_before.json
```

Before 阶段禁止修改 Auditor 集成代码。只允许新增 benchmark runner / cases / report 相关文件。

### B2. After：启用 Intent Auditor 后重跑

实现可选集成：

```text
plan_node -> audit_plan_node -> execute
```

要求：

- 默认关闭：`settings.intent_auditor_enabled = False`
- After 实验显式开启
- Auditor 只过滤/阻止明显 contradiction 或 score < threshold 的 plan step
- 合法 coding task 不应被误杀
- 如果 plan 中所有 step 被过滤，应允许直接回答或进入 finish，而不是死循环

运行同一批 cases：

```bash
python benchmarks/plan_before_after/run_plan_cases.py \
  --input benchmarks/plan_before_after/cases.jsonl \
  --output benchmarks/results/plan_after.json \
  --mode plan \
  --intent-auditor-enabled \
  --threshold 0.6
```

### B3. 工程对比报告

生成：

```text
benchmarks/reports/plan_before_after_report.md
```

报告必须包含：

```markdown
# Plan Mode Before/After Comparison

## Summary Table

| Case | Expected | Before Pass | After Pass | Before Tool Calls | After Tool Calls | Before Write/Edit/Shell | After Write/Edit/Shell |
|------|----------|-------------|------------|-------------------|------------------|--------------------------|-------------------------|

## Key Deltas

- Tool call reduction on conversational/non-coding cases:
- Reduction in write/edit/shell misuse:
- BUG-001 regression status:
- Valid coding task pass status:
- Any false positives introduced by Auditor:

## Case Analysis

Include one short paragraph each for:
- BUG-001 who-are-you case
- Non-coding explanation case
- Valid coding/edit task
- TRAIL-style poor retrieval / goal deviation case
```

---

## C. 最终科研报告补充

在 `research/research_report.md` 中增加一节：

```markdown
## Engineering Before/After Validation

The primary experiment evaluates Intent Auditor offline on TRAIL-PAS. As a secondary engineering validation, we also ran the same Plan-mode cases before and after enabling the auditor.

Summarize:
- before/after pass rate
- tool-call reduction
- write/edit/shell misuse reduction
- whether BUG-001 remains fixed
- whether valid coding tasks are preserved

This section is secondary and must not replace the TRAIL-PAS Precision/Recall/F1 results.
```

---

## D. 重要约束

1. 主实验仍然是 TRAIL-PAS 离线 P/R/F1，不要把端到端 Plan case pass rate 写成主指标。
2. Before/After 工程实验必须使用完全相同的 case 输入。
3. Before 结果必须在 Auditor 集成前生成并保存。
4. After 结果必须在 Auditor 集成后、显式启用开关时生成。
5. Auditor 默认必须关闭，避免改变主项目默认行为。
6. 不要 git commit。
7. 每完成一个阶段先汇报文件路径、样本数、指标，再继续下一阶段。
