你是一名资深 Agent Benchmark Engineer。

请对当前项目进行一次完整的 Coding Agent Benchmark 评估。

项目目标：

当前项目是一个 Claude Code Mini。

已经具备：

* Task Planning
* SubTask Decomposition
* ReAct Loop
* Tool Calling
* File Operation
* Shell Execution
* Error Retry
* Max Iteration Protection

现在需要评估：

Agent 是否具备真实的软件开发能力。

---

# Benchmark原则

请不要让我手动寻找Benchmark。

请自动完成：

1. 搜索适合Coding Agent的Benchmark
2. 下载Benchmark
3. 集成Benchmark
4. 运行Benchmark
5. 分析结果
6. 输出报告

---

# Benchmark选择要求

优先考虑：

## 第一优先级

HumanEval

评估：

* Python编码能力
* 算法实现能力
* 函数实现能力

---

## 第二优先级

MBPP

评估：

* 基础编程能力
* 指令遵循能力
* 函数实现能力

---

## 第三优先级

SWE-Bench Lite

评估：

* 代码库理解能力
* Bug Fix能力
* Agent工具调用能力
* 多轮推理能力

---

如果存在更适合Agent Coding System的Benchmark：

请说明原因后替换。

---

# 自动执行要求

请自动完成：

## Step1

分析当前项目结构。

识别：

* Agent入口
* Planner
* Executor
* Tool Layer

---

## Step2

设计Benchmark Adapter。

使Benchmark能够自动调用：

Agent.run(task)

而不是直接调用LLM。

---

## Step3

下载并集成Benchmark。

要求：

优先使用官方实现。

避免重复造轮子。

---

## Step4

执行Benchmark。

记录：

* 成功率
* Pass@1
* Pass@k
* 平均执行时间
* Token消耗
* Tool调用次数
* 平均ReAct轮数
* 平均Planner步骤数

---

## Step5

失败案例分析。

对每个失败案例分析：

* Planning失败
* Reasoning失败
* Tool调用失败
* Observation理解失败
* Reflection失败
* 代码能力不足

属于哪一种。

---

# 输出内容

请生成：

benchmark_report.md

包含：

## 1. Benchmark简介

为什么选择这些Benchmark。

---

## 2. 运行配置

模型

温度

最大步数

工具配置

运行环境

---

## 3. Benchmark结果

表格展示：

Benchmark

Total

Pass

Fail

Pass Rate

Average Steps

Average Tokens

Average Tool Calls

---

## 4. 错误分类统计

例如：

Planning Error

Reasoning Error

Tool Error

Code Generation Error

Verification Error

---

## 5. Agent能力雷达图

评估：

* Coding
* Planning
* Tool Usage
* Reflection
* Repository Understanding
* Debugging

---

## 6. Claude Code对比分析

从架构角度分析：

当前项目与Claude Code相比：

缺少哪些能力。

---

## 7. 优先级改进建议

输出：

P0

P1

P2

三级优化建议。

---

# 特别要求

不要只运行Benchmark。

请实现：

benchmark_runner/

目录。

包含：

benchmark_runner/
├── adapters/
├── datasets/
├── reports/
├── scripts/
└── benchmark_runner.py

使未来能够持续评测不同版本Agent。

---

最终目标：

建立一个长期可复用的Agent Benchmark Framework。

以后每次修改Agent后：

python benchmark_runner.py

即可自动运行评测并生成报告。
