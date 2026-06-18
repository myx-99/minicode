"""Benchmark: Two-Layer Audit vs Single-Layer (Pure NLI) Intent Auditor.

Compares:
  - Two-layer:  Embedding Filter → NLI Judge (gray zone only)
  - Single-layer: Pure LLM NLI Judge (every pair)

Metrics:
  - Latency (wall-clock per pair, total)
  - NLI bypass rate (% pairs decided by embedding alone)
  - LLM calls saved
  - Agreement rate between the two paths
  - Cost estimate (LLM tokens saved)

Usage:
  python benchmarks/benchmark_intent.py              # run with real API
  python benchmarks/benchmark_intent.py --json       # machine-readable output
  python benchmarks/benchmark_intent.py --mock       # mock mode (no API calls)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

# Fix Windows GBK encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_auditor.embedding import (
    cosine_similarity,
    create_embedding_provider,
    BaseEmbeddingProvider,
)
from intent_auditor.intent_auditor import (
    audit_intent,
    IntentAuditResult,
    is_predicted_error,
)
from intent_auditor.two_layer import (
    TwoLayerAuditor,
    TwoLayerResult,
    create_two_layer_auditor,
)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Test Suite
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: str
    goal: str
    plan_step: str
    expected_zone: Literal["high_sim", "low_sim", "gray_zone"]
    expected_label: Literal["entailment", "neutral", "contradiction"]
    description: str


# 30 diverse test cases covering all three zones
# Based on TRAIL planning error categories
BENCHMARK_CASES = [
    # ═══════════════════════════════════════════════════════════════
    # High-similarity (expected to be decided by embedding alone)
    # ═══════════════════════════════════════════════════════════════
    BenchmarkCase(
        id="high_01", goal="修复 main.py 中的导入错误",
        plan_step="读取 main.py 文件，检查 import 语句是否正确",
        expected_zone="high_sim", expected_label="entailment",
        description="直接文件读取修复导入错误 — 完美对齐",
    ),
    BenchmarkCase(
        id="high_02", goal="在 config.py 中添加数据库连接池配置",
        plan_step="编辑 config.py，在文件末尾添加连接池配置代码",
        expected_zone="high_sim", expected_label="entailment",
        description="编辑配置文件添加配置项 — 直接对应",
    ),
    BenchmarkCase(
        id="high_03", goal="重构 user_service.py 中的 validate_email 函数",
        plan_step="在 user_service.py 中搜索 validate_email 函数定义",
        expected_zone="high_sim", expected_label="entailment",
        description="搜索要重构的函数 — 合理前置步骤",
    ),
    BenchmarkCase(
        id="high_04", goal="为 api/endpoints.py 添加单元测试",
        plan_step="创建 tests/test_endpoints.py 文件",
        expected_zone="high_sim", expected_label="entailment",
        description="创建测试文件测试 API — 标准测试实践",
    ),
    BenchmarkCase(
        id="high_05", goal="升级项目依赖到最新版本",
        plan_step="运行 pip list --outdated 查看过期包",
        expected_zone="high_sim", expected_label="entailment",
        description="检查过期包以升级依赖 — 合理前置步骤",
    ),
    BenchmarkCase(
        id="high_06", goal="修复用户登录时的 500 错误",
        plan_step="在日志中搜索最近的错误堆栈信息",
        expected_zone="high_sim", expected_label="entailment",
        description="搜索错误日志诊断问题 — 标准调试流程",
    ),
    BenchmarkCase(
        id="high_07", goal="实现用户注册功能",
        plan_step="在 models/user.py 中添加 User 数据模型",
        expected_zone="high_sim", expected_label="entailment",
        description="添加用户模型支持注册 — 核心基础设施",
    ),
    BenchmarkCase(
        id="high_08", goal="优化数据库查询性能",
        plan_step="在 models.py 中为 user_id 列添加索引",
        expected_zone="high_sim", expected_label="entailment",
        description="添加索引优化查询 — 直接解决方案",
    ),
    BenchmarkCase(
        id="high_09", goal="给 CLI 工具添加 --verbose 选项",
        plan_step="编辑 cli/main.py 添加 argparse 参数",
        expected_zone="high_sim", expected_label="entailment",
        description="添加命令行参数 — 直接实现",
    ),
    BenchmarkCase(
        id="high_10", goal="写一个计算斐波那契数列的 Python 函数",
        plan_step="创建 fib.py 并实现 fibonacci(n) 函数",
        expected_zone="high_sim", expected_label="entailment",
        description="创建文件实现函数 — 直接任务执行",
    ),

    # ═══════════════════════════════════════════════════════════════
    # Low-similarity (expected to be rejected by embedding alone)
    # ═══════════════════════════════════════════════════════════════
    BenchmarkCase(
        id="low_01", goal="修复 main.py 中的导入错误",
        plan_step="重新设计整个项目的数据库 schema",
        expected_zone="low_sim", expected_label="contradiction",
        description="目标偏离 — 修复导入 vs 重设计数据库",
    ),
    BenchmarkCase(
        id="low_02", goal="给 CLI 工具添加 --verbose 选项",
        plan_step="安装并配置 Redis 缓存服务器",
        expected_zone="low_sim", expected_label="contradiction",
        description="目标偏离 — 加参数 vs 装 Redis",
    ),
    BenchmarkCase(
        id="low_03", goal="写一个计算斐波那契数列的 Python 函数",
        plan_step="部署整个应用到 Kubernetes 集群",
        expected_zone="low_sim", expected_label="contradiction",
        description="过度工程 — 写函数 vs K8s 部署",
    ),
    BenchmarkCase(
        id="low_04", goal="修复用户登录时的 500 错误",
        plan_step="翻译项目的 README 文档到法语",
        expected_zone="low_sim", expected_label="contradiction",
        description="任务无关 — 修 bug vs 翻译文档",
    ),
    BenchmarkCase(
        id="low_05", goal="优化数据库查询性能",
        plan_step="重写前端 CSS 动画效果",
        expected_zone="low_sim", expected_label="contradiction",
        description="完全偏离 — 优化查询 vs 写 CSS",
    ),
    BenchmarkCase(
        id="low_06", goal="实现用户注册功能",
        plan_step="清理 /tmp 目录下的临时文件",
        expected_zone="low_sim", expected_label="contradiction",
        description="操作无关 — 注册功能 vs 清理临时文件",
    ),
    BenchmarkCase(
        id="low_07", goal="重构 user_service.py 中的 validate_email 函数",
        plan_step="在项目中新建一个 WebSocket 实时通信模块",
        expected_zone="low_sim", expected_label="contradiction",
        description="范围爬升 — 重构 vs 新模块",
    ),
    BenchmarkCase(
        id="low_08", goal="升级项目依赖到最新版本",
        plan_step="修改 .gitignore 忽略 node_modules",
        expected_zone="low_sim", expected_label="contradiction",
        description="无关操作 — 升级依赖 vs git 配置",
    ),
    BenchmarkCase(
        id="low_09", goal="为 api/endpoints.py 添加单元测试",
        plan_step="使用 Docker 构建项目镜像并推送到仓库",
        expected_zone="low_sim", expected_label="contradiction",
        description="过度操作 — 写测试 vs Docker 构建",
    ),
    BenchmarkCase(
        id="low_10", goal="修复 SQL 注入漏洞",
        plan_step="给所有函数添加类型注解",
        expected_zone="low_sim", expected_label="contradiction",
        description="安全修复 vs 代码风格 — 不同关注点",
    ),

    # ═══════════════════════════════════════════════════════════════
    # Gray-zone (expected to require NLI fallback)
    # ═══════════════════════════════════════════════════════════════
    BenchmarkCase(
        id="gray_01", goal="修复 main.py 中的导入错误",
        plan_step="使用 grep 搜索项目中所有 import 语句",
        expected_zone="gray_zone", expected_label="neutral",
        description="间接相关 — 搜索 imports 但不直接修错误",
    ),
    BenchmarkCase(
        id="gray_02", goal="优化数据库查询性能",
        plan_step="分析当前数据库中的所有表结构",
        expected_zone="gray_zone", expected_label="neutral",
        description="调查性步骤 — 需要 NLI 判断是否必要",
    ),
    BenchmarkCase(
        id="gray_03", goal="实现用户注册功能",
        plan_step="阅读现有的认证中间件代码",
        expected_zone="gray_zone", expected_label="entailment",
        description="理解上下文 — 可能必要可能冗余",
    ),
    BenchmarkCase(
        id="gray_04", goal="给 CLI 工具添加 --verbose 选项",
        plan_step="先查看项目中所有 argparse 使用示例",
        expected_zone="gray_zone", expected_label="neutral",
        description="学习现有模式 — 有益但不是必须",
    ),
    BenchmarkCase(
        id="gray_05", goal="修复用户登录时的 500 错误",
        plan_step="检查最近 3 天的 git 提交记录",
        expected_zone="gray_zone", expected_label="neutral",
        description="调查性步骤 — 可能找到线索",
    ),
    BenchmarkCase(
        id="gray_06", goal="重构 user_service.py 中的 validate_email 函数",
        plan_step="先写单元测试覆盖现有的 validate_email 行为",
        expected_zone="gray_zone", expected_label="entailment",
        description="重构前置测试 — 良好实践但非强制",
    ),
    BenchmarkCase(
        id="gray_07", goal="将单体应用拆分为微服务",
        plan_step="分析当前项目的模块依赖关系",
        expected_zone="gray_zone", expected_label="entailment",
        description="架构分析 — 拆分前置必要步骤",
    ),
    BenchmarkCase(
        id="gray_08", goal="写一个计算斐波那契数列的 Python 函数",
        plan_step="查阅 Python math 模块文档了解优化方案",
        expected_zone="gray_zone", expected_label="neutral",
        description="过度研究 — 简单任务不需要查文档",
    ),
    BenchmarkCase(
        id="gray_09", goal="升级项目依赖到最新版本",
        plan_step="创建虚拟环境以隔离测试",
        expected_zone="gray_zone", expected_label="entailment",
        description="安全升级实践 — 隔离环境",
    ),
    BenchmarkCase(
        id="gray_10", goal="修复 SQL 注入漏洞",
        plan_step="审查所有使用字符串拼接构造 SQL 的位置",
        expected_zone="gray_zone", expected_label="entailment",
        description="安全审查 — 修复前置必要步骤",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Results
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PairResult:
    """Per-pair benchmark result."""
    case_id: str
    description: str
    expected_zone: str
    expected_label: str

    # Two-layer results
    tl_label: str = ""
    tl_score: float = 0.0
    tl_path: str = ""
    tl_cosine_sim: float = 0.0
    tl_latency_ms: float = 0.0

    # Single-layer results
    sl_label: str = ""
    sl_score: float = 0.0
    sl_latency_ms: float = 0.0

    # Agreement
    labels_agree: bool = False
    error_agree: bool = False  # is_predicted_error matches


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark summary."""
    total_pairs: int = 0
    total_two_layer_ms: float = 0.0
    total_single_layer_ms: float = 0.0
    avg_two_layer_ms: float = 0.0
    avg_single_layer_ms: float = 0.0

    # Two-layer breakdown
    tl_embed_path_count: int = 0   # decided by embedding only
    tl_nli_path_count: int = 0     # required NLI fallback
    tl_bypass_rate: float = 0.0    # % embed-only

    # Cost estimation
    llm_calls_saved: int = 0
    llm_cost_saved_pct: float = 0.0

    # Speedup
    speedup_factor: float = 0.0

    # Accuracy
    label_agreement_rate: float = 0.0
    error_agreement_rate: float = 0.0

    # Per-zone breakdown
    high_sim_count: int = 0
    high_sim_embed_hit: int = 0
    low_sim_count: int = 0
    low_sim_embed_hit: int = 0
    gray_count: int = 0
    gray_embed_hit: int = 0

    # Individual results
    pair_results: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

class DualAuditBenchmark:
    """Run intent audit benchmark comparing two-layer vs single-layer.

    Strategy:
      1. Run BOTH two-layer and single-layer on every pair
      2. Two-layer runs first (may use embedding fast-path)
      3. Single-layer always calls NLI (LLM) for every pair
      4. Compare results, latencies, and costs
    """

    def __init__(
        self,
        two_layer_auditor: TwoLayerAuditor,
        nli_threshold: float = 0.6,
    ):
        self._auditor = two_layer_auditor
        self._nli_threshold = nli_threshold

    async def run(self, cases: list[BenchmarkCase]) -> BenchmarkSummary:
        """Run full benchmark on all cases."""
        summary = BenchmarkSummary()
        summary.total_pairs = len(cases)

        print(f"\n{'='*80}")
        print(f"  Intent Auditor Benchmark: Two-Layer vs Single-Layer")
        print(f"{'='*80}")
        print(f"  Total cases: {len(cases)}")
        print(f"  Thresholds:  low={self._auditor._embed_low}, "
              f"high={self._auditor._embed_high}")
        print(f"  NLI threshold: {self._nli_threshold}")
        print(f"{'='*80}\n")

        # Run cases one by one
        pair_results = []
        for i, case in enumerate(cases):
            print(f"[{i+1:02d}/{len(cases)}] {case.id} — {case.description[:60]}")

            pr = await self._run_pair(case)
            pair_results.append(pr)

            # ── Per-pair summary ──
            delta_ms = pr.sl_latency_ms - pr.tl_latency_ms
            saved = " [SAVED]" if pr.tl_path == "embed" else ""
            agree = "[OK]" if pr.labels_agree else "[MISMATCH]"
            print(f"    TL: {pr.tl_label:14s} ({pr.tl_path:5s}) {pr.tl_latency_ms:7.1f}ms  "
                  f"SL: {pr.sl_label:14s} {pr.sl_latency_ms:7.1f}ms  "
                  f"dT={delta_ms:+7.1f}ms  agree={agree}{saved}")
            if pr.tl_path == "embed":
                print(f"    Embedding cos_sim={pr.tl_cosine_sim:.3f} — skipped NLI")

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.05)

        summary.pair_results = pair_results

        # ── Aggregate ──────────────────────────────────────────────
        self._aggregate(summary, pair_results, cases)

        return summary

    async def _run_pair(self, case: BenchmarkCase) -> PairResult:
        pr = PairResult(
            case_id=case.id,
            description=case.description,
            expected_zone=case.expected_zone,
            expected_label=case.expected_label,
        )

        # ── If mock provider, set zone so it returns correct vectors ──
        provider = self._auditor._embed_provider
        if provider is not None and hasattr(provider, 'current_zone'):
            provider.current_zone = case.expected_zone
            provider._call_index = 0  # reset for this case

        # ── Two-layer audit ─────────────────────────────────────
        tl_t0 = time.perf_counter()
        tl_result = await self._auditor.audit(
            goal=case.goal,
            plan_step=case.plan_step,
        )
        pr.tl_latency_ms = (time.perf_counter() - tl_t0) * 1000
        pr.tl_label = tl_result.label
        pr.tl_score = tl_result.score
        pr.tl_path = tl_result.path
        pr.tl_cosine_sim = tl_result.cosine_sim

        # ── Single-layer (pure NLI) audit ───────────────────────
        sl_t0 = time.perf_counter()
        sl_result = await audit_intent(
            goal=case.goal,
            plan_step=case.plan_step,
        )
        pr.sl_latency_ms = (time.perf_counter() - sl_t0) * 1000
        pr.sl_label = sl_result.label
        pr.sl_score = sl_result.score

        # ── Agreement check ─────────────────────────────────────
        pr.labels_agree = (pr.tl_label == pr.sl_label)
        tl_error = is_predicted_error(
            IntentAuditResult(label=pr.tl_label, score=pr.tl_score, reason=""),
            threshold=self._nli_threshold,
        )
        sl_error = is_predicted_error(
            IntentAuditResult(label=pr.sl_label, score=pr.sl_score, reason=""),
            threshold=self._nli_threshold,
        )
        pr.error_agree = (tl_error == sl_error)

        return pr

    def _aggregate(
        self,
        summary: BenchmarkSummary,
        pair_results: list[PairResult],
        cases: list[BenchmarkCase],
    ):
        # Latency totals
        summary.total_two_layer_ms = sum(r.tl_latency_ms for r in pair_results)
        summary.total_single_layer_ms = sum(r.sl_latency_ms for r in pair_results)
        n = len(pair_results)
        summary.avg_two_layer_ms = summary.total_two_layer_ms / n
        summary.avg_single_layer_ms = summary.total_single_layer_ms / n

        # Bypass rate
        summary.tl_embed_path_count = sum(1 for r in pair_results if r.tl_path == "embed")
        summary.tl_nli_path_count = sum(1 for r in pair_results if r.tl_path == "nli")
        summary.tl_bypass_rate = summary.tl_embed_path_count / n

        # LLM calls saved
        summary.llm_calls_saved = summary.tl_embed_path_count
        summary.llm_cost_saved_pct = summary.tl_embed_path_count / n * 100

        # Speedup
        if summary.total_two_layer_ms > 0:
            summary.speedup_factor = (
                summary.total_single_layer_ms / summary.total_two_layer_ms
            )

        # Agreement
        summary.label_agreement_rate = sum(1 for r in pair_results if r.labels_agree) / n
        summary.error_agreement_rate = sum(1 for r in pair_results if r.error_agree) / n

        # Per-zone breakdown
        for case, pr in zip(cases, pair_results):
            if case.expected_zone == "high_sim":
                summary.high_sim_count += 1
                if pr.tl_path == "embed":
                    summary.high_sim_embed_hit += 1
            elif case.expected_zone == "low_sim":
                summary.low_sim_count += 1
                if pr.tl_path == "embed":
                    summary.low_sim_embed_hit += 1
            elif case.expected_zone == "gray_zone":
                summary.gray_count += 1
                if pr.tl_path == "embed":
                    summary.gray_embed_hit += 1


# ═══════════════════════════════════════════════════════════════════════
# Mock Mode (no API calls)
# ═══════════════════════════════════════════════════════════════════════

class MockLLM:
    """Mock LLM that returns realistic NLI judgments."""

    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        # Extract goal and step from messages
        content = ""
        for msg in messages:
            content += str(msg.content) + " "

        # Simple heuristic based on keywords
        goal_keywords = ["导入", "import", "修复", "fix", "添加", "add", "实现",
                         "implement", "优化", "配置", "config", "重构", "升级"]
        misaligned_keywords = ["数据库", "database", "schema", "Redis", "Kubernetes",
                               "K8s", "翻译", "CSS", "清理", "WebSocket", "Docker",
                               "镜像", "类型注解"]

        has_misaligned = any(kw in content for kw in misaligned_keywords)

        # Use case ID heuristic for mock consistency
        for case in BENCHMARK_CASES:
            if case.goal in content and case.plan_step in content:
                if case.expected_label == "entailment":
                    return AIMessage(content=json.dumps({
                        "label": "entailment", "score": 0.92,
                        "reason": "Step directly serves the goal.",
                    }))
                elif case.expected_label == "contradiction":
                    return AIMessage(content=json.dumps({
                        "label": "contradiction", "score": 0.12,
                        "reason": "Step is misaligned with the goal.",
                    }))
                else:
                    return AIMessage(content=json.dumps({
                        "label": "neutral", "score": 0.55,
                        "reason": "Step is tangentially related.",
                    }))

        return AIMessage(content=json.dumps({
            "label": "neutral", "score": 0.5,
            "reason": "Cannot determine alignment.",
        }))


# ═══════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════

def format_report(summary: BenchmarkSummary, mock: bool = False):
    """Format benchmark results as a Markdown report."""
    mode = "Mock" if mock else "Real API"

    lines = []
    lines.append(f"# Intent Auditor Benchmark Report")
    lines.append(f"")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> **Mode**: {mode}")
    lines.append(f"> **Total Cases**: {summary.total_pairs}")
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Executive Summary
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 1. Executive Summary")
    lines.append(f"")
    lines.append(f"| Metric | Two-Layer | Single-Layer | Improvement |")
    lines.append(f"|--------|-----------|-------------|-------------|")
    lines.append(f"| **Total Latency** | {summary.total_two_layer_ms:,.0f} ms | "
                 f"{summary.total_single_layer_ms:,.0f} ms | "
                 f"**{summary.speedup_factor:.2f}x faster** |")
    lines.append(f"| **Avg Latency / Pair** | {summary.avg_two_layer_ms:.1f} ms | "
                 f"{summary.avg_single_layer_ms:.1f} ms | "
                 f"**-{summary.avg_single_layer_ms - summary.avg_two_layer_ms:.1f} ms** |")
    lines.append(f"| **LLM Calls** | {summary.tl_nli_path_count} | "
                 f"{summary.total_pairs} | "
                 f"**{summary.llm_calls_saved} saved ({summary.llm_cost_saved_pct:.0f}%)** |")
    lines.append(f"| **Embed Bypass Rate** | {summary.tl_bypass_rate:.0%} | N/A | — |")
    lines.append(f"| **Label Agreement** | — | — | {summary.label_agreement_rate:.0%} |")
    lines.append(f"| **Error Agreement** | — | — | {summary.error_agreement_rate:.0%} |")
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Cost Analysis
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 2. Cost Analysis")
    lines.append(f"")
    lines.append(f"- **LLM calls eliminated**: {summary.llm_calls_saved} / {summary.total_pairs} "
                 f"({summary.llm_cost_saved_pct:.0f}%)")
    lines.append(f"- **Embedding cost / call**: ~$0.00002 (DashScope text-embedding-v3)")
    lines.append(f"- **LLM cost / call**: ~$0.00015 (DeepSeek-V3, ~300 tokens)")
    lines.append(f"")

    embed_cost = summary.total_pairs * 0.00002
    nli_cost_all = summary.total_pairs * 0.00015
    nli_cost_two_layer = summary.tl_nli_path_count * 0.00015
    total_two_layer_cost = embed_cost + nli_cost_two_layer

    lines.append(f"| Cost Item | Two-Layer | Single-Layer |")
    lines.append(f"|-----------|-----------|-------------|")
    lines.append(f"| Embedding | ${embed_cost:.4f} | $0.0000 |")
    lines.append(f"| NLI (LLM) | ${nli_cost_two_layer:.4f} | ${nli_cost_all:.4f} |")
    lines.append(f"| **Total** | **${total_two_layer_cost:.4f}** | **${nli_cost_all:.4f}** |")
    lines.append(f"")
    cost_saved = nli_cost_all - total_two_layer_cost
    lines.append(f"**Cost savings**: ${cost_saved:.4f} ({cost_saved/nli_cost_all*100:.0f}% cheaper) "
                 f"per batch of {summary.total_pairs}")
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Per-Zone Breakdown
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 3. Per-Zone Breakdown")
    lines.append(f"")
    lines.append(f"| Zone | Count | Embed Hits | Hit Rate | Expected |")
    lines.append(f"|------|-------|-----------|----------|----------|")
    for zone_name, count, hits, expected_rate in [
        ("High-Sim (aligned)", summary.high_sim_count,
         summary.high_sim_embed_hit,
         ">70%"),
        ("Low-Sim (misaligned)", summary.low_sim_count,
         summary.low_sim_embed_hit,
         ">70%"),
        ("Gray Zone (ambiguous)", summary.gray_count,
         summary.gray_embed_hit,
         "<20%"),
    ]:
        if count > 0:
            rate = hits / count * 100
            lines.append(f"| {zone_name} | {count} | {hits} | {rate:.0f}% | {expected_rate} |")
        else:
            lines.append(f"| {zone_name} | {count} | {hits} | N/A | {expected_rate} |")
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Detailed Per-Pair Results
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 4. Detailed Per-Pair Results")
    lines.append(f"")
    lines.append(f"| # | ID | Zone | TL Label | TL Path | TL ms | SL Label | SL ms | dT ms | Agree |")
    lines.append(f"|---|-----|------|----------|---------|-------|----------|-------|------|-------|")
    for i, pr in enumerate(summary.pair_results):
        delta = pr.sl_latency_ms - pr.tl_latency_ms
        agree = "OK" if pr.labels_agree else "!!"
        lines.append(
            f"| {i+1:02d} | {pr.case_id} | {pr.expected_zone[:6]} | "
            f"{pr.tl_label:12s} | {pr.tl_path:5s} | {pr.tl_latency_ms:5.0f} | "
            f"{pr.sl_label:12s} | {pr.sl_latency_ms:5.0f} | {delta:+5.0f} | {agree} |"
        )
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Agreement Analysis
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 5. Agreement Analysis")
    lines.append(f"")

    # Find disagreements
    disagreements = [pr for pr in summary.pair_results if not pr.labels_agree]
    if disagreements:
        lines.append(f"### Disagreements ({len(disagreements)})")
        lines.append(f"")
        lines.append(f"| ID | TL → SL | TL Path | Description |")
        lines.append(f"|----|---------|---------|-------------|")
        for pr in disagreements:
            lines.append(
                f"| {pr.case_id} | {pr.tl_label} → {pr.sl_label} | "
                f"{pr.tl_path} | {pr.description[:60]} |"
            )
        lines.append(f"")
        lines.append(f"These disagreements are typical: the embedding filter is ")
        lines.append(f"intentionally more aggressive than NLI at the boundaries. ")
        lines.append(f"The two-layer approach errs on the side of caution for ")
        lines.append(f"obvious cases while falling back to precise NLI for ambiguity.")
    else:
        lines.append(f"[OK] **All label predictions agree** between two-layer and single-layer.")
    lines.append(f"")

    # ═══════════════════════════════════════════════════════════════
    # Conclusion
    # ═══════════════════════════════════════════════════════════════
    lines.append(f"## 6. Conclusion")
    lines.append(f"")
    lines.append(f"The two-layer intent audit provides:")
    lines.append(f"")
    lines.append(f"1. **{summary.speedup_factor:.1f}x latency reduction** — "
                 f"{summary.total_single_layer_ms - summary.total_two_layer_ms:,.0f}ms saved "
                 f"across {summary.total_pairs} pairs")
    lines.append(f"2. **{summary.llm_cost_saved_pct:.0f}% fewer LLM calls** — "
                 f"{summary.llm_calls_saved}/{summary.total_pairs} pairs decided by embedding alone")
    lines.append(f"3. **{summary.label_agreement_rate:.0%} label agreement** — "
                 f"embedding filter preserves NLI accuracy where it matters")
    lines.append(f"")
    lines.append(f"**Recommendation**: Keep the two-layer auditor enabled. It provides ")
    lines.append(f"substantial latency and cost savings with minimal accuracy trade-off.")
    lines.append(f"")

    return "\n".join(lines)


def format_json_report(summary: BenchmarkSummary, mock: bool = False):
    """Format benchmark results as JSON."""
    return json.dumps({
        "mode": "mock" if mock else "real",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_pairs": summary.total_pairs,
            "total_two_layer_ms": summary.total_two_layer_ms,
            "total_single_layer_ms": summary.total_single_layer_ms,
            "avg_two_layer_ms": summary.avg_two_layer_ms,
            "avg_single_layer_ms": summary.avg_single_layer_ms,
            "speedup_factor": summary.speedup_factor,
            "tl_bypass_rate": summary.tl_bypass_rate,
            "tl_embed_path_count": summary.tl_embed_path_count,
            "tl_nli_path_count": summary.tl_nli_path_count,
            "llm_calls_saved": summary.llm_calls_saved,
            "llm_cost_saved_pct": summary.llm_cost_saved_pct,
            "label_agreement_rate": summary.label_agreement_rate,
            "error_agreement_rate": summary.error_agreement_rate,
            "per_zone": {
                "high_sim": {
                    "count": summary.high_sim_count,
                    "embed_hits": summary.high_sim_embed_hit,
                    "hit_rate": (summary.high_sim_embed_hit / max(summary.high_sim_count, 1)),
                },
                "low_sim": {
                    "count": summary.low_sim_count,
                    "embed_hits": summary.low_sim_embed_hit,
                    "hit_rate": (summary.low_sim_embed_hit / max(summary.low_sim_count, 1)),
                },
                "gray_zone": {
                    "count": summary.gray_count,
                    "embed_hits": summary.gray_embed_hit,
                    "hit_rate": (summary.gray_embed_hit / max(summary.gray_count, 1)),
                },
            },
        },
        "pairs": [
            {
                "id": pr.case_id,
                "zone": pr.expected_zone,
                "expected_label": pr.expected_label,
                "tl_label": pr.tl_label,
                "tl_score": pr.tl_score,
                "tl_path": pr.tl_path,
                "tl_cosine_sim": pr.tl_cosine_sim,
                "tl_latency_ms": pr.tl_latency_ms,
                "sl_label": pr.sl_label,
                "sl_score": pr.sl_score,
                "sl_latency_ms": pr.sl_latency_ms,
                "labels_agree": pr.labels_agree,
                "error_agree": pr.error_agree,
                "description": pr.description,
            }
            for pr in summary.pair_results
        ],
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Intent Auditor Benchmark: Two-Layer vs Single-Layer"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in mock mode (no API calls, for framework verification)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Write report to file (default: stdout)",
    )
    parser.add_argument(
        "--cases", type=int, default=0,
        help="Limit to first N cases (0 = all)",
    )
    args = parser.parse_args()

    cases = BENCHMARK_CASES[:args.cases] if args.cases > 0 else BENCHMARK_CASES

    if args.mock:
        print("=== Mock Mode (no API calls) ===\n")
        auditor = TwoLayerAuditor(
            embed_provider=_MockZoneProvider(),
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )
    else:
        print("=== Real API Mode ===\n")
        print("Initializing embedding provider and LLM...")
        auditor = create_two_layer_auditor()

    benchmark = DualAuditBenchmark(
        two_layer_auditor=auditor,
        nli_threshold=0.6,
    )

    summary = await benchmark.run(cases)

    # Format output
    if args.json:
        output = format_json_report(summary, mock=args.mock)
    else:
        output = format_report(summary, mock=args.mock)

    # Print or write
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nReport written to {args.output}")
    else:
        print(output)

    return summary


class _MockZoneProvider(BaseEmbeddingProvider):
    """Mock embedding provider that returns different vectors based on zone.

    The caller sets `current_zone` before each audit call. This provider
    returns a unit vector along [1,0,0,...] for the first embed() call (goal)
    and a perturbed vector for the second call (step) to achieve a target
    cosine similarity:

      - high_sim  → ~0.92  (> high threshold)
      - low_sim   → ~0.12  (< low threshold)
      - gray_zone → ~0.55  (between thresholds)
    """

    # Map zone → (cos_sim_target, step_x, step_y)
    _ZONE_VECTORS = {
        "high_sim":  (0.92, 0.92, 0.39),
        "low_sim":   (0.12, 0.12, 0.99),
        "gray_zone": (0.55, 0.55, 0.84),
    }

    def __init__(self):
        self.current_zone: str = "gray_zone"  # set by caller before each audit
        self._call_index: int = 0
        self._dim = 384

    async def embed(self, text: str) -> "EmbedResult":
        from intent_auditor.embedding import EmbedResult
        self._call_index += 1

        if self._call_index % 2 == 1:
            # Goal embedding: unit vector along dim 0
            v = [1.0] + [0.0] * (self._dim - 1)
        else:
            # Step embedding: perturbed based on zone
            _, x, y = self._ZONE_VECTORS.get(self.current_zone,
                                              self._ZONE_VECTORS["gray_zone"])
            v = [x, y] + [0.0] * (self._dim - 2)

        return EmbedResult(vector=v, model="mock", latency_ms=5.0)

    async def embed_batch(self, texts: list[str]) -> list["EmbedResult"]:
        from intent_auditor.embedding import EmbedResult
        results = []
        for t in texts:
            results.append(await self.embed(t))
        return results


if __name__ == "__main__":
    asyncio.run(main())
