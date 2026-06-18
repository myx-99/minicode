# Claude Code Mini — 跨轮次记忆重写 Prompt

> **用途**: 将本文档整体作为 Claude Code 的执行指令，在 `owncode/` 目录下**重写记忆子系统**，实现 REPL 跨 Task 记忆。
> **基线**: V2 已完成（221 tests、双模式、ContextManager）。V1 快照：`claude-code-mini-v1.zip`。
> **范围**: **重写 `memory/` 记忆模块**（保留 `context_manager.py` 不动）；联动修改 `agent/`、`graph/`、`cli/`、`config/`、测试。
> **原则**: Working First — 先让 REPL 三轮对话场景跑通，再 polish。

---

## 0. 背景：当前问题（必须理解）

V2 实测暴露三类问题（详见此前分析）：

| # | 现象 | 根因 |
|---|------|------|
| B1 | REPL 第二轮 Task「删除刚刚完成的任务」，Agent 不知道「刚刚」指矩阵乘法 | 每轮 `stream(task)` 都 `messages: []` 全新 graph，**无 Session 级 state** |
| B2 | 第三轮「刚才完成了什么」，Agent 说无法访问对话历史 | 同上；`LongTermMemory.search(当前task)` 对 meta 问题召回失败 |
| B3 | `.agent/memory/entries.jsonl` 写入 `"encountered issues"` 且摘要为空 | `finish_node` 在 `phase=done` / `final_answer` 赋值**之前**调用 `summarize_session` |

**关键区分（不要混淆）**:

| 模块 | 职责 | 本次是否改动 |
|------|------|:------------:|
| `ContextManager` | **单次 Task 内** messages 超 token 预算时的压缩/摘要 | ❌ **保留不动** |
| 记忆子系统（待重写） | **跨 Task / 跨 REPL 轮次 / 跨进程** 记住做过什么 | ✅ **全部重写** |

---

## 1. 目标与验收标准

### 1.1 核心目标

实现 **两层记忆 + 统一管理器**：

```
┌─────────────────────────────────────────────────────────┐
│                    MemoryManager                         │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  SessionMemory    │    │  ProjectMemory            │  │
│  │  REPL 跨轮次       │    │  跨进程持久化              │  │
│  │  (当前会话 N 轮)   │    │  (.agent/memory/turns)   │  │
│  └────────┬─────────┘    └────────────┬─────────────┘  │
│           └──────────┬────────────────┘                 │
│                      ▼                                    │
│           build_context_for_task(task)                  │
│           record_turn(state, final_answer)                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Acceptance Criteria（必须全部通过）

| # | 场景 | 预期 |
|---|------|------|
| AC-1 | REPL 连续 3 轮：`/mode react` → Task1 建文件夹 → Task2「删除刚刚完成的任务」→ Task3「刚才完成了什么」 | Task2/3 能引用 Task1 的 `matrix_multiplication/` 或等价描述，**无需**重新 glob 全项目猜 |
| AC-2 | `python main.py "task A"` 单次模式 | 正常工作，无 session 污染（或仅 1 turn） |
| AC-3 | `--no-memory` | 禁用 Session + Project 写入与召回 |
| AC-4 | `/memory` | 显示当前 session turn 列表 + project turn 数量 |
| AC-5 | `/memory clear` | 清空 session + project 持久化 |
| AC-6 | 记忆写入 | `finish_node` 完成后写入，content 含 **完整 task + final_answer + 修改的文件列表 + success 状态** |
| AC-7 | 回归 | 221 个现有测试全过（`test_memory.py` 可重写以匹配新 API） |
| AC-8 | 新增测试 | `tests/test_session_memory.py` ≥ 15 个用例 |

### 1.3 冒烟脚本（实现后必须手动验证）

```bash
python -m pytest tests/ -v --tb=short

python main.py
# 交互：
/mode react
新建一个文件夹 matrix_demo 并写一个 hello.py
删除刚刚完成的任务
刚刚我要求你完成了什么任务
/memory
quit
```

---

## 2. 架构设计

### 2.1 新目录结构（重写 memory/）

```
memory/
├── __init__.py              # 导出 MemoryManager, TurnRecord
├── context_manager.py       # 【保留不动】单次 Task 上下文窗口管理
├── types.py                 # 【重写】TurnRecord, SessionState
├── store.py                 # 【重写】turns.jsonl + session.json 持久化
├── session.py               # 【新增】SessionMemory — REPL 跨轮次
├── project.py               # 【新增】ProjectMemory — 跨进程持久化
└── manager.py               # 【新增】MemoryManager — 统一入口
```

**删除**（或清空后替换，不留 dead code）:
- `memory/long_term.py` — 被 `project.py` + `manager.py` 取代

### 2.2 数据模型 (`memory/types.py`)

```python
@dataclass
class TurnRecord:
    """一次完整 Task 的执行记录 — 记忆的最小单元。"""
    id: str                          # turn_20260603_abc123
    user_task: str                   # 用户原始输入
    final_answer: str                # finish_node 生成的摘要（必填，非空才算有效 turn）
    success: bool                    # phase == "done"
    mode: str                        # "plan" | "react"
    files_changed: List[str]         # 从 tool_history 提取 write/edit 路径
    tools_used: List[str]            # 工具名列表（去重）
    created_at: str                  # ISO timestamp
    session_id: str                  # REPL session UUID（同一次 python main.py 相同）

@dataclass
class SessionState:
    """当前 REPL 会话状态（内存 + 可选落盘）。"""
    session_id: str
    turns: List[TurnRecord]          # 有序，最新在末尾
    started_at: str
```

**不再使用** V2 的 `MemoryEntry` + `category` 碎片模型；统一为 `TurnRecord`。

### 2.3 SessionMemory (`memory/session.py`)

**职责**: 管理**当前 REPL 进程内**的 turn 列表。

```python
class SessionMemory:
    def __init__(self, session_id: str | None = None): ...

    def add_turn(self, turn: TurnRecord) -> None: ...
    def get_recent_turns(self, n: int = 10) -> List[TurnRecord]: ...
    def clear(self) -> None: ...
    def format_for_prompt(self, turns: List[TurnRecord]) -> str: ...
```

`format_for_prompt` 输出示例：

```markdown
## Recent Session History (newest last)

### Turn 1 (2026-06-03 15:23, react, ✅)
**User asked:** 新建一个文件夹并实现一个矩阵乘法
**Result:** Created matrix_multiplication/matrix_mul.py with manual and numpy multiply functions.
**Files changed:** matrix_multiplication/matrix_mul.py

### Turn 2 (2026-06-03 15:30, react, ✅)
**User asked:** 删除刚刚完成的任务
...
```

### 2.4 ProjectMemory (`memory/project.py`)

**职责**: 跨进程持久化，写入 `<workspace>/.agent/memory/turns.jsonl`。

```python
class ProjectMemory:
    def __init__(self, workspace_root: Path, max_turns: int = 200): ...

    def add_turn(self, turn: TurnRecord) -> None: ...
    def load_recent(self, n: int = 20) -> List[TurnRecord]: ...
    def search(self, query: str, k: int = 5) -> List[TurnRecord]: ...
    def clear(self) -> None: ...
```

**检索策略（V2.1 轻量，无向量 DB）**:

1. **始终注入** `load_recent(n=5)` — 解决「刚才/刚刚/上一个」类 meta 问题
2. **关键词补充** `search(query, k=3)` — 针对当前 task 相关的历史 turn
3. **Meta 问题检测** — 若 task 匹配以下模式，加大 recent 数量到 10，跳过 keyword：
   ```python
   META_PATTERNS = [
       r"刚才", r"刚刚", r"上一个", r"之前", r"上次",
       r"what did i", r"what was", r"previous task", r"last task",
       r"完成了什么", r"做了什么", r"删.*刚刚",
   ]
   ```

### 2.5 MemoryManager (`memory/manager.py`)

**统一入口**，供 `agent/`、`graph/nodes`、`cli/` 调用：

```python
class MemoryManager:
    def __init__(
        self,
        workspace_root: Path,
        enabled: bool = True,
        session: SessionMemory | None = None,
        project: ProjectMemory | None = None,
        llm: BaseChatModel | None = None,  # 可选：turn 超长时 LLM 压缩 session 块
    ): ...

    def build_context_for_task(self, task: str) -> str:
        """合并 session + project 记忆，返回注入 prompt 的 markdown 字符串。"""

    async def record_completed_turn(
        self,
        state: dict,
        final_answer: str,
        *,
        success: bool,
    ) -> TurnRecord:
        """Task 完成后调用。final_answer 必须已生成。"""

    def get_session_turns(self) -> List[TurnRecord]: ...
    async def clear_all(self) -> None: ...
    @property
    def session_id(self) -> str: ...
```

**`build_context_for_task` 逻辑**:

```
if not enabled: return ""

session_block = session.format_for_prompt(session.get_recent(N))
project_recent = project.load_recent(5)
project_search = project.search(task, 3) if not is_meta_query(task) else []

dedupe by turn.id → format → return combined markdown
```

**`record_completed_turn` 逻辑**:

```
1. 从 state 提取: task, mode, tool_history, plan
2. files_changed = write_file + edit_file 成功路径
3. 构造 TurnRecord(success=success, final_answer=final_answer, ...)
4. session.add_turn(turn)
5. project.add_turn(turn)  # 持久化
6. return turn
```

### 2.6 持久化 (`memory/store.py`)

重写为双文件：

| 文件 | 内容 |
|------|------|
| `.agent/memory/turns.jsonl` | 所有 TurnRecord（append-only，LRU 仅影响内存索引） |
| `.agent/memory/session.json` | 可选：当前 session_id + turn ids（进程重启可恢复，**可选实现**） |

提供 `TurnStore` 类：`append(turn)`, `load_all()`, `load_recent(n)`, `clear()`, `rewrite()`.

**旧文件处理**: 启动时若存在 `entries.jsonl`（V2 格式），**忽略不迁移**（或在 README 注明手动删除）。不要写复杂迁移逻辑。

---

## 3. 集成改动

### 3.1 AgentState (`agent/state.py`)

```python
# 重命名/替换 V2 字段：
session_context: str      # 替代 memory_context — build_context_for_task 的输出
# 保留 context_summary, messages_token_estimate（ContextManager 用）
# 删除 memory_context 或保留为 alias（兼容测试时二选一，优先干净删除）
```

新增可选字段（供 graph 传递，非必须持久化在 state）:

```python
session_id: str           # 当前 REPL session
turn_index: int           # 当前是第几轮 task（0-based，init 时设置）
```

### 3.2 ClaudeCodeMini (`agent/agent.py`)

```python
class ClaudeCodeMini:
    def __init__(
        self,
        ...
        memory_manager: MemoryManager | None = None,  # 外部注入（CLI 持有同一实例）
        memory_enabled: bool = True,
    ):
        # 若 memory_manager is None 且 enabled → 创建新 MemoryManager
        # 删除 _long_term_memory / LongTermMemoryKeyword 引用

    async def run(self, task: str) -> dict:
        initial_state = self._build_initial_state(task)
        # _build_initial_state 内:
        #   session_context = memory_manager.build_context_for_task(task)
        #   session_id = memory_manager.session_id
        #   turn_index = len(memory_manager.get_session_turns())

    async def stream(self, task: str):
        # 同上
```

**关键**: `run`/`stream` **不再**每次创建新 MemoryManager；CLI 注入共享实例以实现跨轮次。

### 3.3 Graph Nodes (`graph/nodes.py`)

#### init_node

```python
async def init_node(state, memory_manager=None):
    task = state["task"]
    session_context = ""
    if memory_manager:
        session_context = memory_manager.build_context_for_task(task)

    system = SYSTEM_PROMPT
    if session_context:
        system += "\n\n" + session_context

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=task),
    ]
    ...
    return {..., "session_context": session_context, ...}
```

#### finish_node — **修复 B3**

```python
async def finish_node(state, llm, memory_manager=None):
    # 1. 先生成 final_answer（现有 LLM 摘要逻辑不变）
    final_answer = ...

    # 2. 再写记忆（final_answer 已就绪）
    success = True  # 能到 finish_node 即 success；或根据 state 判断
    if memory_manager:
        await memory_manager.record_completed_turn(
            state, final_answer, success=success
        )

    # 3. 最后设置 phase
    return {"phase": "done", "final_answer": final_answer}
```

**禁止**在 `final_answer` 生成前调用 `record_completed_turn`。

#### execute_node

- 继续用 `ContextManager.prepare_messages` — **不改 ContextManager**
- 若 step_context 需要 session 信息，优先用 state 中已有的 `session_context`（init 已注入 system，通常不必重复）

### 3.4 Graph Builder (`graph/builder.py`)

```python
def build_graph(..., memory_manager=None, context_manager=None):
    async def _init_node(state):
        return await init_node(state, memory_manager=memory_manager)
    async def _finish_node(state):
        return await finish_node(state, llm, memory_manager=memory_manager)
    # 删除 long_term_memory 参数
```

### 3.5 CLI (`cli/app.py`)

**核心改动 — REPL Session 生命周期**:

```python
class AgentCLI:
    def __init__(self, workspace_path, mode, memory_enabled=True):
        # 创建唯一 MemoryManager（整个 REPL 共享）
        self._memory_manager = MemoryManager(
            workspace_root=...,
            enabled=memory_enabled,
        ) if memory_enabled else None

        self._agent = ClaudeCodeMini(
            ...,
            memory_manager=self._memory_manager,
            memory_enabled=memory_enabled,
        )

    async def run_task(self, task):
        # 每次 task 仍调用 agent.stream(task)
        # 但 agent 使用同一 memory_manager → 轮次累积
        async for state in self._agent.stream(task):
            ...
        # stream 结束后 memory 已在 finish_node 写入，无需 CLI 额外处理

    async def _handle_memory_command(self, task):
        # 重写：显示 SessionMemory turns + ProjectMemory count
        # /memory clear → memory_manager.clear_all()
```

**`/mode` 切换**: 重建 `ClaudeCodeMini` 但 **保留同一 `MemoryManager` 实例**（切换模式不应丢 session）。

### 3.6 Config (`config/settings.py`)

```python
# 重命名/新增
memory_session_turns: int = 10       # REPL 注入最近 N 轮
memory_project_recent: int = 5       # 跨 session 注入最近 N 轮
memory_max_turns: int = 200          # project 持久化上限
# 保留 memory_enabled
# 删除 memory_search_top_k / memory_max_entries（或映射到新字段）
```

### 3.7 Prompts (`prompts/system.py`)

在 SYSTEM_PROMPT 增加简短说明：

```
## Session Memory
When "Recent Session History" is provided in your context, treat it as
authoritative record of what the user asked in earlier turns THIS session.
For follow-up questions like "delete what I just did" or "what was the
previous task", use session history — do NOT claim you have no memory.
```

### 3.8 main.py

- `--no-memory` 传递给 CLI 和 Agent
- 可选新增 `--memory-session-turns N`（非必须，settings 默认即可）

---

## 4. 测试要求

### 4.1 重写 `tests/test_memory.py`

覆盖新 API：
- `TurnRecord` 序列化
- `TurnStore` append/load/clear
- `SessionMemory.add_turn` + `format_for_prompt`
- `ProjectMemory.search` + meta query 走 recent
- `MemoryManager.build_context_for_task` 含最近 turn
- `MemoryManager.record_completed_turn` 在 final_answer 非空时写入
- `memory_enabled=False` 不写不读
- `init_node` 注入 session_context
- `finish_node` 写入后 project 有记录

### 4.2 新增 `tests/test_session_memory.py`

**集成级场景（Mock LLM）**:

```python
async def test_repl_three_turn_recall():
    """模拟 REPL 三轮：建文件夹 → 删除刚刚任务 → 刚才做了什么。
    验证第三轮 build_context_for_task 包含第一轮的 user_task 和 files_changed。
    """
```

```python
async def test_mode_switch_preserves_session():
    """/mode react → plan 切换后 session turns 不丢失。"""
```

```python
def test_meta_query_includes_recent_turns():
    """task='刚刚我要求你完成了什么' → context 含最近 turn 而非空。"""
```

### 4.3 回归

```bash
python -m pytest tests/ -v --tb=short
```

221+ 测试全过。若 `test_memory.py` 旧用例引用 `LongTermMemoryKeyword`，全部更新为新 API。

---

## 5. 分阶段实现

### Phase 1: 数据层 + MemoryManager（~2h）

1. 重写 `memory/types.py`, `memory/store.py`
2. 实现 `session.py`, `project.py`, `manager.py`
3. 删除 `long_term.py`
4. 单元测试 `test_memory.py` 基础部分

### Phase 2: Graph + Agent 集成（~1.5h）

1. 改 `init_node`, `finish_node`, `builder.py`
2. 改 `agent/state.py`, `agent/agent.py`
3. 验证 finish_node 写入顺序正确

### Phase 3: CLI Session 生命周期（~1h）

1. 改 `cli/app.py` — 共享 MemoryManager
2. `/mode` 切换保留 memory
3. 重写 `/memory` 命令

### Phase 4: 测试 + 文档（~1h）

1. `test_session_memory.py`
2. 全量 pytest
3. 更新 `README.md` Memory 章节（简短）
4. 创建 `report/v2/V2.1_MEMORY_REPORT.md`（问题 → 方案 → 验收）

---

## 6. 约束

### 必须遵守

- **不修改** `memory/context_manager.py`（职责不同）
- **不破坏** Plan/React 双模式（221 测试）
- **不引入** 向量 DB / embedding 库（V3 再做）
- 新增依赖 **0 个**
- 记忆写入必须在 `final_answer` **之后**
- REPL 多轮共享 **同一个** `MemoryManager` 实例

### 非目标

- 不把完整 `messages` 历史跨 Task 传入 graph（token 爆炸；用 TurnRecord 摘要即可）
- 不做 `entries.jsonl` → `turns.jsonl` 自动迁移
- 不改 Benchmark runner（除非 import 报错）

---

## 7. 文件 Checklist

- [ ] `memory/types.py` — TurnRecord, SessionState
- [ ] `memory/store.py` — TurnStore, turns.jsonl
- [ ] `memory/session.py` — SessionMemory
- [ ] `memory/project.py` — ProjectMemory + meta query
- [ ] `memory/manager.py` — MemoryManager
- [ ] `memory/__init__.py` — 导出新 API
- [ ] `memory/long_term.py` — **删除**
- [ ] `agent/state.py` — session_context
- [ ] `agent/agent.py` — memory_manager 注入
- [ ] `graph/nodes.py` — init/finish 改造
- [ ] `graph/builder.py` — memory_manager 参数
- [ ] `cli/app.py` — 共享 MemoryManager + /memory
- [ ] `config/settings.py` — 新配置项
- [ ] `prompts/system.py` — session memory 说明
- [ ] `tests/test_memory.py` — 重写
- [ ] `tests/test_session_memory.py` — 新增
- [ ] `README.md` — 更新
- [ ] `report/v2/V2.1_MEMORY_REPORT.md` — 新增

---

## 8. 给 Claude Code 的执行指令（直接复制）

```
你是 Claude Code Mini 项目的记忆子系统重写工程师。工作目录是 owncode/。

请严格按照 CROSS_TURN_MEMORY_PROMPT.md 全文执行：

1. 阅读 §0 理解 ContextManager vs 记忆子系统的区别
2. 运行 pytest 确认 V2 基线 221 测试通过
3. 按 §5 四个 Phase 顺序实现
4. 重写 memory/（保留 context_manager.py 不动，删除 long_term.py）
5. 实现 SessionMemory + ProjectMemory + MemoryManager（§2）
6. CLI 层共享 MemoryManager 实现 REPL 跨轮次（§3.5）
7. 修复 finish_node 先 final_answer 后写记忆（§3.3）
8. 完成 §4 全部测试 + §7 checklist
9. 手动跑 §1.3 冒烟脚本验证三轮对话
10. 创建 report/v2/V2.1_MEMORY_REPORT.md

硬性要求：
- ContextManager 不要改
- REPL 三轮场景必须能回答「刚才完成了什么」
- record_completed_turn 必须在 final_answer 生成之后
- 221+ pytest 全过
- 新增依赖 0 个
- 不要 commit，除非我明确要求

开始执行 Phase 1。
```

---

> **文档版本**: CrossTurn-Memory-Prompt-1.0  
> **生成日期**: 2026-06-03  
> **前置版本**: V2（221 tests, dual mode, ContextManager）
