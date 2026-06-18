"""Intent Auditor offline evaluation on TRAIL-PAS.

Runs audit_intent() on every TRAIL-PAS sample and computes:
  - P/R/F1/AUROC vs gold labels
  - Per-category / per-impact / per-source breakdown
  - Cost estimation (token counts, latency)

Supports --limit (debug runs) and --resume (skip already-evaluated samples).

Usage:
    python benchmarks/eval/run_auditor.py \
      --input benchmarks/trail_pas/trail_pas.jsonl \
      --output benchmarks/results/auditor_after.json \
      --threshold 0.6
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from intent_auditor.intent_auditor import (
    audit_intent,
    is_predicted_error,
    IntentAuditResult,
)
from metrics import (
    load_samples,
    compute_metrics,
    compute_category_breakdown,
    compute_impact_breakdown,
    compute_source_breakdown,
)


# ── Token estimation (char / 4 heuristic) ───────────────────────────

def estimate_tokens(text: str) -> int:
    """Quick token estimate: char count / 4 (works across providers)."""
    return max(1, len(text) // 4)


# ── Evaluation runner ────────────────────────────────────────────────

async def evaluate_sample(sample: dict, threshold: float) -> dict:
    """Run audit_intent on one sample, return result dict."""
    goal = sample["user_goal"]
    plan_step = sample["plan_step"]

    result: IntentAuditResult = await audit_intent(goal=goal, plan_step=plan_step)
    predicted = is_predicted_error(result, threshold=threshold)

    return {
        "sample_id": sample["sample_id"],
        "trace_id": sample["trace_id"],
        "source": sample["source"],
        "gold_label": int(sample["alignment_error"]),
        "gold_category": sample.get("error_category", ""),
        "gold_impact": sample.get("impact", "LOW"),
        "predicted_error": int(predicted),
        "auditor_label": result.label,
        "auditor_score": result.score,
        "auditor_reason": result.reason,
        "latency_ms": result.latency_ms,
        "goal_len": len(goal),
        "step_len": len(plan_step),
        "prompt_tokens_est": estimate_tokens(goal) + estimate_tokens(plan_step),
    }


async def run_all(
    samples: list[dict],
    threshold: float = 0.6,
    limit: int | None = None,
    resume_ids: set | None = None,
) -> list[dict]:
    """Evaluate all samples, skipping those already in resume_ids.

    Returns list of result dicts.
    """
    results = []
    samples_to_run = samples[:limit] if limit else samples

    for sample in tqdm(samples_to_run, desc="Auditing", unit="sample"):
        sid = sample["sample_id"]
        if resume_ids and sid in resume_ids:
            continue

        try:
            result = await evaluate_sample(sample, threshold)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {sid}: {e}", file=sys.stderr)
            results.append({
                "sample_id": sid,
                "trace_id": sample["trace_id"],
                "source": sample["source"],
                "gold_label": int(sample["alignment_error"]),
                "gold_category": sample.get("error_category", ""),
                "gold_impact": sample.get("impact", "LOW"),
                "predicted_error": 0,
                "auditor_label": "error",
                "auditor_score": 0.5,
                "auditor_reason": f"Auditor error: {e}",
                "latency_ms": 0,
                "goal_len": len(sample["user_goal"]),
                "step_len": len(sample["plan_step"]),
                "prompt_tokens_est": 0,
            })

    return results


# ── Reporting helpers ───────────────────────────────────────────────

def find_cases(
    results: list[dict], label: str, max_n: int = 3
) -> list[dict]:
    """Find TP/FP/FN/TN cases for case study.

    label: "TP" | "FP" | "FN" | "TN"
    """
    cases = []
    for r in results:
        gold = r["gold_label"]
        pred = r["predicted_error"]
        if label == "TP" and gold == 1 and pred == 1:
            cases.append(r)
        elif label == "FP" and gold == 0 and pred == 1:
            cases.append(r)
        elif label == "FN" and gold == 1 and pred == 0:
            cases.append(r)
        elif label == "TN" and gold == 0 and pred == 0:
            cases.append(r)
        if len(cases) >= max_n:
            break
    return cases


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Intent Auditor on TRAIL-PAS")
    parser.add_argument("--input", required=True, help="Path to trail_pas.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--threshold", type=float, default=0.6, help="Decision threshold")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for debugging)")
    parser.add_argument("--resume", type=str, default=None, help="Path to previous results JSON for resume")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {input_path}...")
    samples = load_samples(str(input_path))
    print(f"  Loaded {len(samples)} samples")

    # Resume support
    resume_ids: set = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                prev_results = json.load(f)
                prev_predictions = prev_results.get("predictions", [])
                resume_ids = {p["sample_id"] for p in prev_predictions}
                print(f"  Resuming: {len(resume_ids)} already evaluated")

    effective_limit = min(args.limit or len(samples), len(samples))
    print(f"  Will evaluate {effective_limit - len(resume_ids)} samples")
    print(f"  Threshold: {args.threshold}")

    print("\nRunning Intent Auditor...")
    t_start = time.perf_counter()
    results = asyncio.run(run_all(
        samples, threshold=args.threshold,
        limit=effective_limit, resume_ids=resume_ids,
    ))
    total_elapsed_s = time.perf_counter() - t_start

    # Merge with previous if resuming
    if resume_ids:
        prev_predictions = prev_results.get("predictions", [])
        existing_ids = {p["sample_id"] for p in prev_predictions}
        for r in results:
            if r["sample_id"] not in existing_ids:
                prev_predictions.append(r)
        all_predictions = prev_predictions
    else:
        all_predictions = results

    # ── Compute metrics ──
    # Need to align predictions with gold samples
    gold_map = {s["sample_id"]: s for s in samples}
    y_true = [int(gold_map[p["sample_id"]]["alignment_error"]) for p in all_predictions]
    y_pred = [p["predicted_error"] for p in all_predictions]
    scores = [p["auditor_score"] for p in all_predictions]

    full_metrics = compute_metrics(y_true, y_pred, scores=scores, threshold=args.threshold)
    cat_breakdown = compute_category_breakdown(
        [gold_map[p["sample_id"]] for p in all_predictions], y_pred
    )
    impact_breakdown = compute_impact_breakdown(
        [gold_map[p["sample_id"]] for p in all_predictions], y_pred
    )
    source_breakdown = compute_source_breakdown(
        [gold_map[p["sample_id"]] for p in all_predictions], y_pred
    )

    # ── Cost estimation ──
    total_latency_ms = sum(p.get("latency_ms", 0) for p in all_predictions)
    total_prompt_tokens = sum(p.get("prompt_tokens_est", 0) for p in all_predictions)
    avg_latency_ms = total_latency_ms / len(all_predictions) if all_predictions else 0

    output = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "threshold": args.threshold,
        "total_samples": len(all_predictions),
        "positive_samples": sum(y_true),
        "negative_samples": len(y_true) - sum(y_true),
        "metrics": full_metrics,
        "category_breakdown": cat_breakdown,
        "impact_breakdown": impact_breakdown,
        "source_breakdown": source_breakdown,
        "cost": {
            "total_calls": len(all_predictions),
            "total_latency_ms": total_latency_ms,
            "avg_latency_ms": avg_latency_ms,
            "total_prompt_tokens_est": total_prompt_tokens,
            "total_elapsed_s": total_elapsed_s,
        },
        "case_studies": {
            "TP": find_cases(all_predictions, "TP", max_n=3),
            "FP": find_cases(all_predictions, "FP", max_n=3),
            "FN": find_cases(all_predictions, "FN", max_n=3),
            "TN": find_cases(all_predictions, "TN", max_n=3),
        },
        "predictions": all_predictions,
    }

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("Intent Auditor Results")
    print(f"{'='*60}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Total predictions: {len(all_predictions)}")
    print(f"  Total time: {total_elapsed_s:.1f}s")
    print(f"  Avg latency: {avg_latency_ms:.0f}ms")
    print()
    print(f"  Precision: {full_metrics['precision']:.4f}")
    print(f"  Recall:    {full_metrics['recall']:.4f}")
    print(f"  F1:        {full_metrics['f1']:.4f}")
    print(f"  Accuracy:  {full_metrics['accuracy']:.4f}")
    if full_metrics.get("auroc") is not None:
        print(f"  AUROC:     {full_metrics['auroc']:.4f}")
    print(f"  TP={full_metrics['tp']} FP={full_metrics['fp']} FN={full_metrics['fn']} TN={full_metrics['tn']}")
    print(f"  Est. prompt tokens: {total_prompt_tokens}")

    # ── Write output ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
