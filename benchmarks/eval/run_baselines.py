"""Baseline predictors for TRAIL-PAS evaluation.

Implements three baselines that do NOT call any LLM:
  - Always-Pass: predict alignment_error=False for all samples
  - Keyword-Heuristic: tool keyword → entailment, else → inconsistent
  - Random: random prediction at the positive class prevalence

Usage:
    python benchmarks/eval/run_baselines.py \
      --input benchmarks/trail_pas/trail_pas.jsonl \
      --output benchmarks/results/baselines_before_auditor.json \
      --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

from metrics import (
    load_samples,
    compute_metrics,
    compute_category_breakdown,
    compute_impact_breakdown,
    compute_source_breakdown,
)

# ── Baseline predictors ─────────────────────────────────────────────

CODING_KEYWORDS = (
    "edit", "write", "read", "glob", "search", "run", "shell",
    "pytest", "file", "tool", "code", "patch", "function", "class",
    "import", "gitingest", "execute", "test", "install",
)


def predict_always_pass(samples: list[dict]) -> list[int]:
    """All samples predicted as consistent (no error)."""
    return [0] * len(samples)


def predict_keyword_heuristic(samples: list[dict]) -> list[int]:
    """Keyword-based: has tool keyword → entailment(0), else → inconsistent(1)."""
    predictions = []
    for s in samples:
        step = s["plan_step"].lower()
        if any(k in step for k in CODING_KEYWORDS):
            predictions.append(0)  # entailment — step mentions tools
        else:
            predictions.append(1)  # inconsistent — conservative guess
    return predictions


def predict_random(samples: list[dict], seed: int = 42) -> list[int]:
    """Random prediction at positive class prevalence."""
    rng = random.Random(seed)
    pos_count = sum(1 for s in samples if s["alignment_error"])
    prevalence = pos_count / len(samples) if samples else 0.5
    return [1 if rng.random() < prevalence else 0 for _ in samples]


# ── Helpers ─────────────────────────────────────────────────────────

def find_case_studies(samples: list[dict], y_pred: list[int], category: str, n: int = 5):
    """Find false-negative cases (gold=1, pred=0) for a given category."""
    cases = []
    for s, pred in zip(samples, y_pred):
        if s["alignment_error"] and pred == 0 and s.get("error_category") == category:
            cases.append({
                "sample_id": s["sample_id"],
                "trace_id": s["trace_id"],
                "source": s["source"],
                "impact": s["impact"],
                "goal_preview": s["user_goal"][:200],
                "step_preview": s["plan_step"][:200],
                "description": s["error_description"][:200],
            })
        if len(cases) >= n:
            break
    return cases


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baselines on TRAIL-PAS")
    parser.add_argument("--input", required=True, help="Path to trail_pas.jsonl")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {input_path}...")
    samples = load_samples(str(input_path))
    print(f"  Loaded {len(samples)} samples")

    y_true = [int(s["alignment_error"]) for s in samples]

    baselines = {
        "always_pass": {
            "name": "Always-Pass",
            "predictions": predict_always_pass(samples),
            "scores": [1.0] * len(samples),  # Always predicts consistent
        },
        "keyword_heuristic": {
            "name": "Keyword-Heuristic",
            "predictions": predict_keyword_heuristic(samples),
            "scores": [0.0 if p == 1 else 1.0 for p in predict_keyword_heuristic(samples)],
        },
        "random": {
            "name": "Random",
            "predictions": predict_random(samples, seed=args.seed),
            "scores": [0.5] * len(samples),
        },
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "total_samples": len(samples),
        "positive_samples": sum(y_true),
        "negative_samples": len(samples) - sum(y_true),
        "baselines": {},
    }

    print("\n=== Baseline Results ===")
    for key, bl in baselines.items():
        y_pred = bl["predictions"]
        scores = bl["scores"]
        name = bl["name"]

        metrics = compute_metrics(y_true, y_pred, scores=scores, threshold=0.6)
        cat_breakdown = compute_category_breakdown(samples, y_pred)
        impact_breakdown = compute_impact_breakdown(samples, y_pred)
        source_breakdown = compute_source_breakdown(samples, y_pred)

        results["baselines"][key] = {
            "name": name,
            "metrics": metrics,
            "category_breakdown": cat_breakdown,
            "impact_breakdown": impact_breakdown,
            "source_breakdown": source_breakdown,
        }

        print(f"\n--- {name} ---")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if metrics.get("auroc") is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

    # ── Case studies: heuristic misses on Goal Deviation ──
    print("\n=== Keyword-Heuristic: Missed Goal Deviation Cases ===")
    kw_preds = baselines["keyword_heuristic"]["predictions"]
    missed_gd = find_case_studies(samples, kw_preds, "Goal Deviation", n=5)
    results["missed_goal_deviation_by_keyword"] = missed_gd
    for i, case in enumerate(missed_gd):
        print(f"\n  [{i+1}] {case['sample_id']}")
        print(f"      Impact: {case['impact']}")
        print(f"      Goal:   {case['goal_preview'][:150]}")
        print(f"      Step:   {case['step_preview'][:150]}")

    # ── Write output ──
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
