#!/usr/bin/env python3
"""Two-Layer Auditor vs Single-Layer NLI — TRAIL-PAS offline comparison.

Usage:  python benchmarks/eval/run_two_layer.py

Output: benchmarks/results/two_layer_comparison.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)


# ── Load data ────────────────────────────────────────────────────

def load_trail_pas(data_path: str = "benchmarks/trail_pas/trail_pas.jsonl"):
    path = Path(data_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            samples.append(s)
    return samples


# ── Metrics ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, scores):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = 0.5
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "accuracy": round(acc, 4),
    }


# ── Main ─────────────────────────────────────────────────────────

async def run_comparison():
    samples = load_trail_pas()
    n = len(samples)
    print(f"Loaded {n} samples.")

    from intent_auditor.intent_auditor import audit_intent, is_predicted_error
    from intent_auditor.two_layer import create_two_layer_auditor
    from config.settings import settings

    threshold = settings.intent_auditor_threshold
    two_layer = create_two_layer_auditor()

    print(f"Threshold: {threshold}")
    print(f"Two-Layer: enabled={two_layer._enabled}, "
          f"low={two_layer._embed_low}, high={two_layer._embed_high}")

    # ── Ground truth ─────────────────────────────────────────
    # TRAIL-PAS: alignment_error = "True"/"False"
    y_true = [1 if str(s.get("alignment_error", "")).lower() == "true" else 0
              for s in samples]

    # ── Run single-layer NLI ──────────────────────────────────
    results_single = []
    for i, s in enumerate(samples):
        goal = s.get("user_goal", "")
        step = s.get("plan_step", "")
        if i % 100 == 0:
            print(f"  Single-layer: {i}/{n}")

        try:
            r = await audit_intent(goal=goal, plan_step=step)
            pred = 1 if is_predicted_error(r, threshold=threshold) else 0
        except Exception:
            pred = 0  # fail-safe

        results_single.append({
            "pred": pred,
            "label": r.label if "r" in dir() else "neutral",
            "score": r.score if "r" in dir() else 0.5,
            "latency_ms": r.latency_ms if "r" in dir() else 0,
        })

    # ── Run two-layer audit ───────────────────────────────────
    results_two = []
    for i, s in enumerate(samples):
        goal = s.get("user_goal", "")
        step = s.get("plan_step", "")
        if i % 100 == 0:
            print(f"  Two-Layer: {i}/{n}")

        try:
            r = await two_layer.audit(goal=goal, plan_step=step)
            pred = 1 if (
                r.label == "contradiction" or r.score < threshold
            ) else 0
        except Exception:
            pred = 0
            r = None

        results_two.append({
            "pred": pred,
            "label": r.label if r else "neutral",
            "score": r.score if r else 0.5,
            "path": r.path if r else "error",
            "cosine_sim": r.cosine_sim if r else 0.0,
            "latency_ms": r.total_latency_ms if r else 0,
        })

    # ── Overall metrics ───────────────────────────────────────
    y_pred_s = [r["pred"] for r in results_single]
    scores_s = [1.0 - r["score"] for r in results_single]
    y_pred_t = [r["pred"] for r in results_two]
    scores_t = [1.0 - r["score"] for r in results_two]

    m_s = compute_metrics(y_true, y_pred_s, scores_s)
    m_t = compute_metrics(y_true, y_pred_t, scores_t)

    # ── Category breakdown ────────────────────────────────────
    cats = {}
    for i, s in enumerate(samples):
        c = s.get("error_category", "Unknown")
        cats.setdefault(c, {"yt": [], "ps": [], "pt": []})
        cats[c]["yt"].append(y_true[i])
        cats[c]["ps"].append(y_pred_s[i])
        cats[c]["pt"].append(y_pred_t[i])

    cat_metrics = {}
    for c, d in cats.items():
        u = sum(d["yt"])
        if u == 0 or u == len(d["yt"]):
            continue
        cat_metrics[c] = {
            "single": compute_metrics(d["yt"], d["ps"], [0.5]*len(d["yt"])),
            "two":   compute_metrics(d["yt"], d["pt"], [0.5]*len(d["yt"])),
            "support": len(d["yt"]),
        }

    # ── Impact breakdown ──────────────────────────────────────
    impacts = {}
    for i, s in enumerate(samples):
        imp = s.get("impact", "UNKNOWN")
        impacts.setdefault(imp, {"yt": [], "ps": [], "pt": []})
        impacts[imp]["yt"].append(y_true[i])
        impacts[imp]["ps"].append(y_pred_s[i])
        impacts[imp]["pt"].append(y_pred_t[i])

    imp_metrics = {}
    for imp, d in impacts.items():
        u = sum(d["yt"])
        if u == 0:
            continue
        imp_metrics[imp] = {
            "single_recall": round(sum(1 for i, p in enumerate(d["ps"])
                if p == 1 and d["yt"][i] == 1) / max(u, 1), 4),
            "two_recall": round(sum(1 for i, p in enumerate(d["pt"])
                if p == 1 and d["yt"][i] == 1) / max(u, 1), 4),
            "support": len(d["yt"]),
            "positives": u,
        }

    # ── Efficiency ────────────────────────────────────────────
    embed_n = sum(1 for r in results_two if r["path"] == "embed")
    nli_n = sum(1 for r in results_two if r["path"] == "nli")
    bypass = embed_n / n

    s_lats = [r["latency_ms"] for r in results_single]
    t_lats = [r["latency_ms"] for r in results_two]
    e_lats = [r["latency_ms"] for r in results_two if r["path"] == "embed"]
    n_lats = [r["latency_ms"] for r in results_two if r["path"] == "nli"]

    report = {
        "config": {
            "threshold": threshold,
            "embed_low": two_layer._embed_low,
            "embed_high": two_layer._embed_high,
            "total": n,
            "positive": sum(y_true),
            "negative": n - sum(y_true),
        },
        "overall": {
            "single_layer": m_s,
            "two_layer": m_t,
            "delta": {k: round(m_t[k] - m_s[k], 4) for k in m_s},
        },
        "efficiency": {
            "bypass_rate": round(bypass, 4),
            "embed_decisions": embed_n,
            "nli_decisions": nli_n,
            "single_avg_ms": round(sum(s_lats)/max(len(s_lats),1), 1),
            "two_avg_ms": round(sum(t_lats)/max(len(t_lats),1), 1),
            "embed_avg_ms": round(sum(e_lats)/max(len(e_lats),1), 1) if e_lats else 0,
            "nli_avg_ms": round(sum(n_lats)/max(len(n_lats),1), 1) if n_lats else 0,
        },
        "categories": cat_metrics,
        "impacts": imp_metrics,
        "stats": two_layer.stats,
    }

    # ── Print ─────────────────────────────────────────────────
    def delta(k):
        d = m_t[k] - m_s[k]
        return f"{d:+.4f}"

    print()
    print("=" * 64)
    print("  Two-Layer Auditor vs Single-Layer NLI — TRAIL-PAS (555)")
    print("=" * 64)
    print(f"  {'Metric':<14} {'Single':>8} {'Two-Layer':>10} {'Δ':>10}")
    print(f"  {'-'*42}")
    for k in ("precision", "recall", "f1", "auroc", "accuracy"):
        print(f"  {k:<14} {m_s[k]:>8.4f} {m_t[k]:>10.4f} {delta(k):>10}")
    print()
    print(f"  LLM calls saved: {embed_n}/{n} ({bypass*100:.1f}%)")
    print(f"  Avg latency: {report['efficiency']['single_avg_ms']:.0f}ms → "
          f"{report['efficiency']['two_avg_ms']:.0f}ms "
          f"({(1 - report['efficiency']['two_avg_ms']/max(report['efficiency']['single_avg_ms'], 1))*100:.1f}% faster)")
    print(f"  Embed path: {report['efficiency']['embed_avg_ms']:.0f}ms avg")
    print(f"  NLI path:   {report['efficiency']['nli_avg_ms']:.0f}ms avg")

    if cat_metrics:
        print(f"\n  {'Category':<30} {'Single F1':>10} {'Two F1':>10} {'Δ':>10}")
        print(f"  {'-'*62}")
        for c, m in sorted(cat_metrics.items()):
            d = m["two"]["f1"] - m["single"]["f1"]
            print(f"  {c:<30} {m['single']['f1']:>10.4f} {m['two']['f1']:>10.4f} {d:>+10.4f}")

    if imp_metrics:
        print(f"\n  {'Impact':<10} {'Single Rec':>12} {'Two Rec':>12} {'Δ':>10}")
        print(f"  {'-'*46}")
        for imp in ("HIGH", "MEDIUM", "LOW"):
            if imp in imp_metrics:
                m = imp_metrics[imp]
                d = m["two_recall"] - m["single_recall"]
                print(f"  {imp:<10} {m['single_recall']:>12.4f} {m['two_recall']:>12.4f} {d:>+10.4f}  (n={m['support']})")

    print()

    # ── Save ───────────────────────────────────────────────────
    out = Path("benchmarks/results/two_layer_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out}")

    return report


if __name__ == "__main__":
    asyncio.run(run_comparison())
