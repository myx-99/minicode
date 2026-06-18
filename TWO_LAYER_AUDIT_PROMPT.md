# Two-Layer Audit Implementation Prompt

> 可直接粘贴到 Claude Code 中使用。本项目已配置好所有依赖和环境变量。

## 任务

为 `ClaudeCodeMini` 项目的 Intent Auditor 添加双层审计架构：

```
          Goal (用户任务)
              │
              ▼
     ┌─ Embedding Filter (Layer 1) ─┐
     │   cosine_sim < 0.35  → 直接判定 contradiction (跳过LLM)      │
     │   cosine_sim > 0.82  → 直接判定 entailment (跳过LLM)        │
     │   其他                → 进入 Layer 2                           │
     └─────────────────────────────────┘
              │ (gray zone)
              ▼
     ┌─ NLI Judge (Layer 2) ─────────┐
     │   LLM-as-Judge 语义判断                                        │
     │   entailment / neutral / contradiction                         │
     └─────────────────────────────────┘
              │
              ▼
         最终判定 + 分数
```

## 环境配置 (`.env` 已配置)

```
# 嵌入模型 (阿里云 DashScope)
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=sk-bc18dc4533084729b21bc4e70970990c
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 向量库缓存 (Qdrant 云服务)
QDRANT_URL=https://63598e86-a74c-4162-96e9-b57d743b8a8e.sa-east-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTk4YjVhOWItOTE2MC00YzBhLTg0ZGUtZWQyN2U1MGZkOTk0In0.ifzMHCuTRYY7zldVO2w5qMYysdljKpPJNLITm6xjYB4
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_VECTOR_SIZE=384

# 双层审计阈值
AUDITOR_EMBED_LOW=0.35
AUDITOR_EMBED_HIGH=0.82
AUDITOR_TWO_LAYER=true
AUDITOR_QDRANT_CACHE=true

# LLM (DeepSeek, 用于 NLI Layer 2)
OPENAI_API_KEY=sk-c335800739c4442ebb49fbb673c4fd6e
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

## 已创建的文件

| 文件 | 说明 |
|------|------|
| `intent_auditor/embedding.py` | 嵌入提供者 (DashScope + 本地) + Qdrant 缓存 + Cosine相似度 |
| `intent_auditor/two_layer.py` | 双层审计编排器 (Embedding Filter → NLI Judge) |
| `intent_auditor/__init__.py` | 导出所有新符号 |
| `config/settings.py` | 新增 9 个配置字段 |
| `graph/nodes.py` | `audit_plan_node` + `execute_node` 已集成双层审计 |
| `tests/conftest.py` | 测试中禁用双层 (用单层 Mock LLM) |
| `tests/test_two_layer.py` | 17 个双层审计单元测试 |
| `benchmarks/eval/run_two_layer.py` | 离线对比评测脚本 |

## 架构关键设计决策

1. **双层用户视角**: 用户的 Goal Embedding 缓存在 Qdrant 中，Plan Step 不缓存。避免每个 goal-step pair 都重复计算 goal embedding。

2. **保守失败策略**: 
   - 嵌入层失败 → 回退到纯 NLI
   - NLI 失败 → 允许步骤执行 (safe by default)
   - 全部被 embedding 拒绝 → NLI 二次校验

3. **LLM 可注入**: `TwoLayerAuditor` 接受 `llm` 参数，Graph 节点传入自己的 LLM 实例，避免重复创建。

4. **测试隔离**: `conftest.py` 通过 `AUDITOR_TWO_LAYER=false` 环境变量在所有测试中禁用双层审计，避免 Mock LLM 对齐问题。

## 预期效果

- LLM 调用减少 **60-80%** (大部分 goal-step 对通过 embedding 快筛即可判定)
- Goal Deviation 召回率保持在 **80%** 以上
- 平均延迟从 1.8s (纯 NLI) 降至 **0.3-0.5s**
- 误报率降低 (embedding 相似度阻断了明显的 false positive)

## 验证步骤

```bash
# 1. 运行所有测试 (323 个)
pytest tests/ -v

# 2. 运行双层审计专项测试 (17 个)
pytest tests/test_two_layer.py -v

# 3. 运行离线对比评测 (需要网络)
python benchmarks/eval/run_two_layer.py
```
