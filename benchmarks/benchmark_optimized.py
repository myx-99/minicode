"""Optimized benchmark comparing different audit configurations.

Configurations:
  A) Two-layer (DashScope embed) — no Qdrant cache
  B) Two-layer (local embed) — sentence-transformers, no network
  C) Single-layer (pure NLI)

Each configuration is run against the same 30 test pairs.
"""

import sys, io, asyncio, time, json, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dataclasses import dataclass, field
from typing import Literal, Optional

# Load .env before any other imports
from dotenv import load_dotenv
load_dotenv()

from intent_auditor.embedding import (
    BaseEmbeddingProvider, QdrantEmbeddingCache, cosine_similarity,
    create_embedding_provider, DashScopeEmbeddingProvider,
    LocalEmbeddingProvider,
)
from intent_auditor.intent_auditor import audit_intent, IntentAuditResult, is_predicted_error
from intent_auditor.two_layer import TwoLayerAuditor, TwoLayerResult

# Re-use the same benchmark cases
from benchmarks.benchmark_intent import (
    BENCHMARK_CASES, BenchmarkCase, PairResult,
    BenchmarkSummary, DualAuditBenchmark, format_report, format_json_report,
    _MockZoneProvider, MockLLM,
)

# ═══════════════════════════════════════════════════════════════════════
# Optimized Two-Layer Auditor (no Qdrant cache)
# ═══════════════════════════════════════════════════════════════════════

class OptimizedBenchmark:
    """Compare 3 configurations: DashScope embed, Local embed, Pure NLI."""

    def __init__(self, nli_threshold: float = 0.6):
        self._nli_threshold = nli_threshold
        self._llm = None  # shared LLM for fairness

    async def _get_llm(self):
        if self._llm is None:
            from config.llm import create_llm
            self._llm = create_llm()
        return self._llm

    async def run(self, cases: list[BenchmarkCase]):
        print(f"\n{'='*80}")
        print(f"  Intent Auditor Benchmark: Optimized Comparison")
        print(f"{'='*80}")
        print(f"  Config A: Two-Layer (DashScope embed, NO Qdrant cache)")
        print(f"  Config B: Two-Layer (Local embed, zero network)")
        print(f"  Config C: Single-Layer (Pure NLI)")
        print(f"{'='*80}\n")

        # Config A: DashScope two-layer (no cache)
        print("--- Config A: DashScope two-layer (no cache) ---")
        auditor_a = TwoLayerAuditor(
            embed_provider=DashScopeEmbeddingProvider(),
            embed_cache=None,  # NO Qdrant cache
            llm=await self._get_llm(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        # Config B: Local sentence-transformers two-layer (no network)
        print("--- Config B: Local sentence-transformers two-layer ---")
        try:
            local_provider = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2")
            # Force-load the model
            local_provider._load()
            print("  Local model loaded successfully")
            auditor_b = TwoLayerAuditor(
                embed_provider=local_provider,
                embed_cache=None,
                llm=await self._get_llm(),
                embed_low=0.35,
                embed_high=0.82,
                enabled=True,
            )
        except Exception as e:
            print(f"  Local model failed to load: {e}")
            print("  Skipping Config B")
            auditor_b = None

        # Config C: Pure NLI
        print("--- Config C: Pure NLI ---")
        llm = await self._get_llm()

        # Run all 3 configs per case
        results_a = []
        results_b = []
        results_c = []

        for i, case in enumerate(cases):
            print(f"\n[{i+1:02d}/{len(cases)}] {case.id} — {case.description[:60]}")

            # C: Single-layer NLI (always first as baseline)
            sl_result = await self._run_nli(case, llm)
            results_c.append(sl_result)

            # A: DashScope two-layer
            tl_a = await self._run_two_layer(case, auditor_a)
            tl_a.sl_label = sl_result.sl_label
            tl_a.sl_score = sl_result.sl_score
            tl_a.sl_latency_ms = sl_result.sl_latency_ms
            tl_a.labels_agree = (tl_a.tl_label == sl_result.sl_label)
            results_a.append(tl_a)

            # B: Local two-layer
            if auditor_b is not None:
                tl_b = await self._run_two_layer(case, auditor_b)
                tl_b.sl_label = sl_result.sl_label
                tl_b.sl_score = sl_result.sl_score
                tl_b.sl_latency_ms = sl_result.sl_latency_ms
                tl_b.labels_agree = (tl_b.tl_label == sl_result.sl_label)
                results_b.append(tl_b)

            # Print comparison
            path_a = tl_a.tl_path
            path_b = tl_b.tl_path if auditor_b else "N/A"
            print(f"    C(NLI-only): {sl_result.sl_label:14s} {sl_result.sl_latency_ms:6.0f}ms")
            print(f"    A(DashScope):  {tl_a.tl_label:14s} ({path_a:5s}) {tl_a.tl_latency_ms:6.0f}ms  "
                  f"cos_sim={tl_a.tl_cosine_sim:.3f}  dT={sl_result.sl_latency_ms - tl_a.tl_latency_ms:+6.0f}ms")
            if auditor_b:
                print(f"    B(Local):      {tl_b.tl_label:14s} ({path_b:5s}) {tl_b.tl_latency_ms:6.0f}ms  "
                      f"cos_sim={tl_b.tl_cosine_sim:.3f}  dT={sl_result.sl_latency_ms - tl_b.tl_latency_ms:+6.0f}ms")

            await asyncio.sleep(0.03)

        # Aggregate
        summary_a = self._compute_summary(results_a, cases, "DashScope Embed + NLI")
        summary_b = self._compute_summary(results_b, cases, "Local Embed + NLI") if auditor_b else None
        summary_c = self._compute_nli_only(results_c, "Pure NLI Only")

        return summary_a, summary_b, summary_c, results_a, results_b, results_c

    async def _run_nli(self, case: BenchmarkCase, llm) -> PairResult:
        pr = PairResult(
            case_id=case.id, description=case.description,
            expected_zone=case.expected_zone, expected_label=case.expected_label,
        )
        t0 = time.perf_counter()
        r = await audit_intent(goal=case.goal, plan_step=case.plan_step, llm=llm)
        pr.sl_latency_ms = (time.perf_counter() - t0) * 1000
        pr.sl_label = r.label
        pr.sl_score = r.score
        return pr

    async def _run_two_layer(self, case: BenchmarkCase, auditor: TwoLayerAuditor) -> PairResult:
        pr = PairResult(
            case_id=case.id, description=case.description,
            expected_zone=case.expected_zone, expected_label=case.expected_label,
        )
        t0 = time.perf_counter()
        r = await auditor.audit(goal=case.goal, plan_step=case.plan_step)
        pr.tl_latency_ms = (time.perf_counter() - t0) * 1000
        pr.tl_label = r.label
        pr.tl_score = r.score
        pr.tl_path = r.path
        pr.tl_cosine_sim = r.cosine_sim
        return pr

    def _compute_summary(self, results: list[PairResult], cases: list[BenchmarkCase],
                         label: str) -> BenchmarkSummary:
        s = BenchmarkSummary()
        s.total_pairs = len(results)
        s.total_two_layer_ms = sum(r.tl_latency_ms for r in results)
        s.total_single_layer_ms = sum(r.sl_latency_ms for r in results)
        n = len(results)
        s.avg_two_layer_ms = s.total_two_layer_ms / n
        s.avg_single_layer_ms = s.total_single_layer_ms / n
        s.tl_embed_path_count = sum(1 for r in results if r.tl_path == "embed")
        s.tl_nli_path_count = sum(1 for r in results if r.tl_path == "nli")
        s.tl_bypass_rate = s.tl_embed_path_count / n if n else 0
        s.llm_calls_saved = s.tl_embed_path_count
        s.llm_cost_saved_pct = s.tl_embed_path_count / n * 100 if n else 0
        if s.total_two_layer_ms > 0:
            s.speedup_factor = s.total_single_layer_ms / s.total_two_layer_ms
        s.label_agreement_rate = sum(1 for r in results if r.labels_agree) / n if n else 0
        s.error_agreement_rate = sum(1 for r in results if r.error_agree) / n if n else 0

        for case, pr in zip(cases, results):
            if case.expected_zone == "high_sim":
                s.high_sim_count += 1
                if pr.tl_path == "embed": s.high_sim_embed_hit += 1
            elif case.expected_zone == "low_sim":
                s.low_sim_count += 1
                if pr.tl_path == "embed": s.low_sim_embed_hit += 1
            elif case.expected_zone == "gray_zone":
                s.gray_count += 1
                if pr.tl_path == "embed": s.gray_embed_hit += 1

        s.pair_results = results
        return s

    def _compute_nli_only(self, results: list[PairResult], label: str) -> dict:
        total_ms = sum(r.sl_latency_ms for r in results)
        n = len(results)
        return {
            "label": label,
            "total_ms": total_ms,
            "avg_ms": total_ms / n if n else 0,
            "nli_calls": n,
        }


def format_comparison_report(
    summary_a: BenchmarkSummary,
    summary_b: Optional[BenchmarkSummary],
    nli_summary: dict,
) -> str:
    """Generate a comparison report across all configs."""
    lines = []
    lines.append("# Intent Auditor — Configuration Comparison Report")
    lines.append(f"")
    lines.append(f"> **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> **Mode**: Real API (optimized, no Qdrant cache)")
    lines.append(f"> **Total Cases**: {summary_a.total_pairs}")
    lines.append(f"")

    # ═════════════════════════════════════════════════════════════
    # Side-by-side comparison
    # ═════════════════════════════════════════════════════════════
    lines.append("## 1. Configuration Comparison")
    lines.append("")
    lines.append("| Metric | A: DashScope 2-Layer | B: Local 2-Layer | C: Pure NLI | Best |")
    lines.append("|--------|---------------------|-------------------|-------------|------|")

    # Row: Total latency
    a_total = summary_a.total_two_layer_ms
    b_total = summary_b.total_two_layer_ms if summary_b else float('inf')
    c_total = nli_summary["total_ms"]
    best_total = min(a_total, b_total, c_total)
    lines.append(
        f"| **Total Latency** | {a_total:,.0f} ms | {b_total:,.0f} ms | "
        f"{c_total:,.0f} ms | "
        f"{'A' if best_total == a_total else 'B' if best_total == b_total else 'C'} |"
    )

    # Row: Avg latency
    a_avg = summary_a.avg_two_layer_ms
    b_avg = summary_b.avg_two_layer_ms if summary_b else float('inf')
    c_avg = nli_summary["avg_ms"]
    lines.append(
        f"| **Avg / Pair** | {a_avg:.0f} ms | {b_avg:.0f} ms | "
        f"{c_avg:.0f} ms | |"
    )

    # Row: Embed bypass rate
    b_bypass_str = f"{summary_b.tl_bypass_rate:.0%} ({summary_b.tl_embed_path_count}/{summary_b.total_pairs})" if summary_b else "N/A"
    lines.append(
        f"| **Embed Bypass** | {summary_a.tl_bypass_rate:.0%} "
        f"({summary_a.tl_embed_path_count}/{summary_a.total_pairs}) | "
        f"{b_bypass_str} | N/A | |"
    )

    # Row: LLM calls
    lines.append(
        f"| **LLM Calls** | {summary_a.tl_nli_path_count} | "
        f"{summary_b.tl_nli_path_count if summary_b else 'N/A'} | "
        f"{nli_summary['nli_calls']} | |"
    )

    # Row: Label agreement
    b_agree_str = f"{summary_b.label_agreement_rate:.0%}" if summary_b else "N/A"
    lines.append(
        f"| **Label Agree** | {summary_a.label_agreement_rate:.0%} | "
        f"{b_agree_str} | "
        f"100% (baseline) | |"
    )

    # Row: Speedup vs NLI
    a_speedup = c_total / a_total if a_total else 0
    b_speedup = c_total / b_total if b_total else 0
    a_saved = c_total - a_total
    b_saved = c_total - b_total
    lines.append(f"| **vs NLI (speedup)** | {a_speedup:.2f}x "
                 f"({a_saved:+,.0f}ms) | "
                 f"{b_speedup:.2f}x ({b_saved:+,.0f}ms) | "
                 f"1.00x (baseline) | |")
    lines.append("")

    # ═════════════════════════════════════════════════════════════
    # Per-zone breakdown
    # ═════════════════════════════════════════════════════════════
    lines.append("## 2. Per-Zone Embed Bypass Rate")
    lines.append("")
    lines.append("| Zone | A: DashScope | B: Local | Expected |")
    lines.append("|------|-------------|----------|----------|")
    for zone_label, a_count, a_hits, b_count, b_hits, expected in [
        ("High-Sim", summary_a.high_sim_count, summary_a.high_sim_embed_hit,
         summary_b.high_sim_count if summary_b else 0,
         summary_b.high_sim_embed_hit if summary_b else 0, ">70%"),
        ("Low-Sim", summary_a.low_sim_count, summary_a.low_sim_embed_hit,
         summary_b.low_sim_count if summary_b else 0,
         summary_b.low_sim_embed_hit if summary_b else 0, ">70%"),
        ("Gray Zone", summary_a.gray_count, summary_a.gray_embed_hit,
         summary_b.gray_count if summary_b else 0,
         summary_b.gray_embed_hit if summary_b else 0, "<20%"),
    ]:
        a_rate = f"{a_hits/max(a_count,1)*100:.0f}%" if a_count else "N/A"
        b_rate = f"{b_hits/max(b_count,1)*100:.0f}%" if b_count else "N/A"
        lines.append(f"| {zone_label} | {a_rate} ({a_hits}/{a_count}) | "
                     f"{b_rate} ({b_hits}/{b_count}) | {expected} |")
    lines.append("")

    # ═════════════════════════════════════════════════════════════
    # Cost Analysis
    # ═════════════════════════════════════════════════════════════
    lines.append("## 3. Cost Analysis (per batch of 30)")
    lines.append("")
    lines.append("| Cost Item | A: DashScope | B: Local | C: Pure NLI |")
    lines.append("|-----------|-------------|----------|-------------|")
    n = summary_a.total_pairs

    # DashScope embedding cost
    a_embed_cost = n * 0.00002  # $0.00002 per DashScope embed
    a_nli_cost = summary_a.tl_nli_path_count * 0.00015
    a_total_cost = a_embed_cost + a_nli_cost

    # Local embedding cost (free)
    b_embed_cost = 0
    b_nli_cost = summary_b.tl_nli_path_count * 0.00015 if summary_b else 0
    b_total_cost = b_embed_cost + b_nli_cost

    # NLI only
    c_cost = n * 0.00015

    lines.append(f"| Embedding | ${a_embed_cost:.6f} | $0.000000 | $0.000000 |")
    lines.append(f"| NLI (LLM) | ${a_nli_cost:.6f} | ${b_nli_cost:.6f} | ${c_cost:.6f} |")
    lines.append(f"| **Total** | **${a_total_cost:.6f}** | **${b_total_cost:.6f}** | **${c_cost:.6f}** |")
    lines.append("")

    a_savings = c_cost - a_total_cost
    b_savings = c_cost - b_total_cost
    lines.append(f"- A vs C: save **${a_savings:.6f}** ({a_savings/c_cost*100:.0f}%)")
    lines.append(f"- B vs C: save **${b_savings:.6f}** ({b_savings/c_cost*100:.0f}%)")
    lines.append("")

    # ═════════════════════════════════════════════════════════════
    # Detailed Results
    # ═════════════════════════════════════════════════════════════
    lines.append("## 4. Detailed Per-Pair Results")
    lines.append("")
    lines.append("| # | ID | NLI ms | A ms | A path | A cos | B ms | B path | B cos | Win |")
    lines.append("|---|-----|--------|------|--------|-------|------|--------|-------|-----|")

    for i in range(len(summary_a.pair_results)):
        r_a = summary_a.pair_results[i]
        r_b = summary_b.pair_results[i] if summary_b else None
        sl_ms = r_a.sl_latency_ms

        a_ms = r_a.tl_latency_ms
        b_ms = r_b.tl_latency_ms if r_b else 0.0

        best = "C"
        if a_ms < sl_ms and a_ms < b_ms: best = "A"
        elif b_ms < sl_ms and b_ms < a_ms: best = "B"
        elif a_ms < sl_ms or b_ms < sl_ms: best = "A" if a_ms < b_ms else "B"

        lines.append(
            f"| {i+1:02d} | {r_a.case_id} | {sl_ms:.0f} | {a_ms:.0f} | "
            f"{r_a.tl_path:5s} | {r_a.tl_cosine_sim:.3f} | "
            f"{b_ms:.0f} | {r_b.tl_path if r_b else 'N/A':5s} | "
            f"{r_b.tl_cosine_sim if r_b else 0:.3f} | "
            f"{best} |"
        )
    lines.append("")

    # ═════════════════════════════════════════════════════════════
    # Conclusion
    # ═════════════════════════════════════════════════════════════
    lines.append("## 5. Recommendations")
    lines.append("")
    lines.append(f"1. **Disable Qdrant cloud cache** — cross-region latency (sa-east-1) "
                 f"adds 3-5s per cache operation, making the cache counterproductive")
    lines.append(f"2. **Use local embedding model** if sentence-transformers is installed — "
                 f"zero network latency, zero API cost, good discrimination")
    lines.append(f"3. **Tune thresholds per embedding model** — DashScope v3 needs "
                 f"lower high threshold (~0.75) to match its output distribution")
    lines.append(f"4. **Implement local LRU cache** instead of remote Qdrant — "
                 f"simple dict with TTL beats cross-region API calls")
    lines.append("")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Optimized Intent Audit Benchmark")
    parser.add_argument("--cases", type=int, default=30, help="Number of cases (default: 30)")
    parser.add_argument("--output", type=str, default="benchmarks/report_optimized_comparison.md")
    args = parser.parse_args()

    cases = BENCHMARK_CASES[:args.cases]

    bench = OptimizedBenchmark()
    summary_a, summary_b, nli_summary, results_a, results_b, results_c = await bench.run(cases)

    report = format_comparison_report(summary_a, summary_b, nli_summary)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n\nReport written to {args.output}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
