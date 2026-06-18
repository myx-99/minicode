"""TRAIL-PAS extractor: TRAIL annotations + traces → trail_pas.jsonl

Constructs the Planning-Alignment Subset from Patronus AI TRAIL dataset.
Reads local annotations and OpenTelemetry traces, produces labeled
(goal, plan_step, alignment_error) triples for Intent Auditor evaluation.

Usage:
    python benchmarks/trail_pas/extract_pas.py
    # → benchmarks/trail_pas/trail_pas.jsonl
    # → benchmarks/trail_pas/trail_pas_stats.json
"""

import json
import re
import sys
from collections import Counter
from itertools import chain
from pathlib import Path

from schema import (
    TrailPASSample,
    PLANNING_CATEGORIES,
    normalize_category,
    is_planning_category,
)

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIL_ROOT = PROJECT_ROOT / "benchmarks" / "data" / "trail"
OUTPUT_JSONL = Path(__file__).resolve().parent / "trail_pas.jsonl"
OUTPUT_STATS = Path(__file__).resolve().parent / "trail_pas_stats.json"

# ── Constants ───────────────────────────────────────────────────────
NEG_CAP_PER_TRACE = 2
MIN_STEP_TEXT_LEN = 40
MAX_USER_GOAL_LEN = 800
MAX_EVIDENCE_LEN = 2000

# Spans whose names indicate agent action (candidate for negative sampling)
AGENT_SPAN_NAMES = {"LiteLLMModel.__call__", "ActionStep", "smolagents"}


# ═════════════════════════════════════════════════════════════════════
# ── JSON helpers ────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

# Some annotation JSON files have trailing commas before ] or }
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def load_json_lenient(path: Path) -> dict:
    """Load JSON, fixing trailing commas that appear in some TRAIL annotations."""
    text = path.read_text(encoding="utf-8")
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return json.loads(text)


def load_trace(path: Path) -> dict:
    """Load a trace JSON (no lenient fix needed for traces)."""
    return json.loads(path.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════
# ── user_goal extraction ────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

def extract_user_goal(trace: dict) -> str:
    """Extract the user's original task from a trace.

    Strategy (in order):
      1. SWE-Bench: CodeAgent.run span input.value has {"task": "..."}
      2. GAIA: User messages with "Here is your task:" / "Here is the task:"
      3. Search output.value for "New task:" patterns
      4. Fallback: use the first assistant output content
    """
    # ── Strategy 0: SWE-Bench CodeAgent.run input.value.task ──
    for span in iter_spans(trace):
        if span.get("span_name") == "CodeAgent.run":
            attrs = span.get("span_attributes", {})
            in_val = attrs.get("input.value", "")
            if in_val:
                try:
                    in_data = json.loads(in_val)
                    task = in_data.get("task", "")
                    if task:
                        # SWE Bench: extract issue statement from <issue> tags
                        m = re.search(r"<issue>\s*(.+?)\s*</issue>", task, re.DOTALL)
                        if m:
                            issue_text = m.group(1).strip()
                            # Remove markdown code blocks for cleaner text
                            issue_text = re.sub(r"```.*?```", "[CODE]", issue_text, flags=re.DOTALL)
                            return issue_text[:MAX_USER_GOAL_LEN]
                        # Fallback: extract first meaningful part
                        return _extract_task_from_content(task)[:MAX_USER_GOAL_LEN]
                except (json.JSONDecodeError, TypeError):
                    pass
            break  # Only check the first CodeAgent.run

    # ── Strategy 1: GAIA user messages ──
    all_user_msgs = []
    for span in iter_spans(trace):
        attrs = span.get("span_attributes", {})
        role = attrs.get("llm.input_messages.0.message.role", "")
        content = attrs.get("llm.input_messages.0.message.content", "")
        if role == "user" and content:
            all_user_msgs.append(content)

    # Prefer messages that contain task markers
    for content in all_user_msgs:
        if "Here is your task:" in content or "New task:" in content:
            task = _extract_task_from_content(content)
            if task:
                return task[:MAX_USER_GOAL_LEN]

    # Fallback: try any user message
    for content in all_user_msgs:
        task = _extract_task_from_content(content)
        if task:
            return task[:MAX_USER_GOAL_LEN]

    # ── Strategy 2: Search output.value for task markers ──
    for span in iter_spans(trace):
        attrs = span.get("span_attributes", {})
        out_val = attrs.get("output.value", "")
        task = _extract_task_from_output(out_val)
        if task:
            return task[:MAX_USER_GOAL_LEN]

    # ── Strategy 3: Fallback — use first assistant content ──
    for span in iter_spans(trace):
        attrs = span.get("span_attributes", {})
        content = attrs.get("llm.output_messages.0.message.content", "")
        if content and len(content) >= 40:
            return content[:MAX_USER_GOAL_LEN]

    return "(no user goal found)"


def _extract_task_from_content(content: str) -> str:
    """Extract the actual task from a user message that may contain framework prompts.

    GAIA traces embed the real task inside multi-layered prompts:
      1. "Here is your task:" → Task: ```...actual task...```
      2. "New task:\\n" (in output.value)
      3. Innermost "Here is the task:" → actual question
      4. Direct task after "Task:" marker
    """
    # ── Pattern 0: Innermost "Here is the task:" — the actual user question ──
    # GAIA wraps: "Here is your task:" (orchestrator) → ... → "Here is the task:" (actual)
    last_here_is = content.rfind("Here is the task:")
    if last_here_is >= 0:
        task_text = content[last_here_is + len("Here is the task:"):]
        # Trim leading whitespace/newlines
        task_text = task_text.strip()
        # Cut at end markers
        for end_marker in ["\n<end_plan>", "\n```", "\n---", "\n\nFailure"]:
            end_idx = task_text.find(end_marker)
            if end_idx >= 0:
                task_text = task_text[:end_idx]
        if len(task_text.strip()) >= 20:
            return task_text.strip()

    # ── Pattern 1: "Here is your task:" followed by Task: code block ──
    m = re.search(
        r"Here is your task:.*?Task:\s*\n```\s*\n(.+?)\n```",
        content, re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # ── Pattern 2: "New task:" in output.value ──
    for marker in ["New task:\n"]:
        idx = content.find(marker)
        if idx >= 0:
            task_text = content[idx + len(marker):]
            end_m = re.search(r"\n---|\n<end_plan>|\n```", task_text)
            if end_m:
                task_text = task_text[:end_m.start()]
            return task_text.strip()

    # ── Pattern 3: "Task:" after dashes ──
    m = re.search(r"---\s*\nTask:\s*\n(.+)", content, re.DOTALL)
    if m:
        return m.group(1).strip()

    # ── Pattern 4: Content IS the task ──
    skip_prefixes = [
        "Below I will present you a task.",
        "You will now build a comprehensive",
        "You are a world expert",
    ]
    for prefix in skip_prefixes:
        if content.startswith(prefix):
            m = re.search(
                r"(?:Here is your task:.*?Task:|the task is|Your task is|task description):\s*\n?```?\s*(.+?)(?:\n```|\n\n###|\n---|\n<end_plan>)",
                content, re.DOTALL | re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()
            break

    # Return content as-is (truncation applied by caller)
    return content.strip()


def _extract_task_from_output(out_val: str) -> str | None:
    """Try to extract task description from output.value."""
    if not out_val:
        return None

    for marker in ["New task:\n", "Here is the task:"]:
        idx = out_val.find(marker)
        if idx >= 0:
            text = out_val[idx + len(marker):]
            # Cut at the next apparent section boundary
            end_idx = text.find("\n---")
            if end_idx > 0:
                text = text[:end_idx]
            return text.strip()

    # Check for Task: within the output
    m = re.search(r"Task:\s*\n(.+?)(?:\n\n---|\n\nYou're|\n\nYou have)", out_val, re.DOTALL)
    if m:
        return m.group(1).strip()

    return None


# ═════════════════════════════════════════════════════════════════════
# ── plan_step extraction (negative samples) ─────────────────────────
# ═════════════════════════════════════════════════════════════════════

def extract_thought_code(span: dict) -> str:
    """Extract Thought/Code content from a span for negative sampling.

    Returns the LLM output content if it looks like an agent action step
    (contains Thought: or code/tool references), otherwise empty string.
    """
    attrs = span.get("span_attributes", {})

    # Try llm.output_messages first (most common in TRAIL traces)
    content = attrs.get("llm.output_messages.0.message.content", "")
    if content:
        return content.strip()

    # Try output.value (may contain execution logs)
    out_val = attrs.get("output.value", "")
    if out_val and len(out_val) >= MIN_STEP_TEXT_LEN:
        return out_val.strip()

    return ""


def iter_spans(trace: dict):
    """Depth-first iterator over all spans in a trace (including child_spans)."""
    for span in trace.get("spans", []):
        yield from _iter_span_tree(span)


def _iter_span_tree(span: dict):
    """Recursively yield span and all descendants."""
    yield span
    for child in span.get("child_spans", []):
        yield from _iter_span_tree(child)


# ═════════════════════════════════════════════════════════════════════
# ── Main extraction ─────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

def extract_samples() -> list[TrailPASSample]:
    """Main extraction: TRAIL annotations + traces → TrailPASSample list."""
    samples: list[TrailPASSample] = []

    # Collect annotation paths
    ann_paths = sorted(chain(
        (TRAIL_ROOT / "processed_annotations_gaia").glob("*.json"),
        (TRAIL_ROOT / "processed_annotations_swe_bench").glob("*.json"),
    ))

    stats = {
        "total_annotations": len(ann_paths),
        "total_errors_in_annotations": 0,
        "planning_errors_found": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "traces_with_negatives": 0,
        "traces_without_trace_json": 0,
    }

    for ann_path in ann_paths:
        try:
            ann = load_json_lenient(ann_path)
        except Exception as e:
            print(f"  WARNING: Could not parse {ann_path.name}: {e}", file=sys.stderr)
            continue

        # ── Determine trace_id ──────────────────────────────
        trace_id = ann.get("trace_id")
        if not trace_id:
            # Some annotations lack trace_id field — use filename
            trace_id = ann_path.stem

        # ── Determine source ─────────────────────────────────
        source = "GAIA" if "gaia" in str(ann_path).lower() else "SWE-Bench"

        # ── Find trace JSON ──────────────────────────────────
        if source == "GAIA":
            trace_path = TRAIL_ROOT / "GAIA" / f"{trace_id}.json"
        else:
            trace_path = TRAIL_ROOT / "SWE Bench" / f"{trace_id}.json"

        if not trace_path.exists():
            stats["traces_without_trace_json"] += 1
            continue

        try:
            trace = load_trace(trace_path)
        except Exception as e:
            print(f"  WARNING: Could not parse trace {trace_path}: {e}", file=sys.stderr)
            continue

        # ── Extract user goal ────────────────────────────────
        user_goal = extract_user_goal(trace)

        # ── Build error location set ─────────────────────────
        errors = ann.get("errors", [])
        stats["total_errors_in_annotations"] += len(errors)
        error_locations = {e["location"] for e in errors}

        # ── Positive samples: planning errors ────────────────
        for err in errors:
            cat = normalize_category(err["category"])
            if cat not in PLANNING_CATEGORIES:
                continue

            stats["planning_errors_found"] += 1
            stats["positive_samples"] += 1

            evidence = err.get("evidence", "")[:MAX_EVIDENCE_LEN]
            location = err.get("location", "")
            sample_id = f"{trace_id}_{location}_{cat.replace(' ', '_')}"

            samples.append(TrailPASSample(
                sample_id=sample_id,
                trace_id=trace_id,
                source=source,
                user_goal=user_goal,
                plan_step=evidence,
                span_id=location,
                alignment_error=True,
                error_category=cat,
                error_description=err.get("description", ""),
                impact=err.get("impact", "LOW").upper(),
                evidence=evidence,
            ))

        # ── Negative samples: non-error agent spans ──────────
        neg_count_for_trace = 0
        for span in iter_spans(trace):
            span_id = span.get("span_id", "")
            if span_id in error_locations:
                continue

            span_name = span.get("span_name", "")
            # Accept spans with LLM output or relevant span names
            step_text = extract_thought_code(span)
            if not step_text or len(step_text) < MIN_STEP_TEXT_LEN:
                continue

            if neg_count_for_trace >= NEG_CAP_PER_TRACE:
                break

            neg_count_for_trace += 1
            stats["negative_samples"] += 1

            samples.append(TrailPASSample(
                sample_id=f"{trace_id}_{span_id}_neg{neg_count_for_trace}",
                trace_id=trace_id,
                source=source,
                user_goal=user_goal,
                plan_step=step_text[:MAX_EVIDENCE_LEN],
                span_id=span_id,
                alignment_error=False,
                error_category="",
                error_description="",
                impact="LOW",
                evidence="",
            ))

        if neg_count_for_trace > 0:
            stats["traces_with_negatives"] += 1

    return samples, stats


# ═════════════════════════════════════════════════════════════════════
# ── Statistics ──────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

def compute_stats(samples: list[TrailPASSample]) -> dict:
    """Compute category/impact/source distribution from samples."""
    categories = Counter()
    impacts = Counter()
    sources = Counter()
    pos_count = 0
    neg_count = 0

    # Track positive samples only for category/impact breakdown
    pos_by_category = Counter()
    pos_by_impact = Counter()
    pos_by_source = Counter()

    for s in samples:
        sources[s.source] += 1
        if s.alignment_error:
            pos_count += 1
            pos_by_category[s.error_category] += 1
            pos_by_impact[s.impact] += 1
            pos_by_source[s.source] += 1
        else:
            neg_count += 1

    return {
        "total_samples": len(samples),
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "positive_by_category": dict(pos_by_category.most_common()),
        "positive_by_impact": dict(pos_by_impact.most_common()),
        "by_source": dict(sources.most_common()),
        "positive_by_source": dict(pos_by_source.most_common()),
        "goal_length_stats": _compute_length_stats(
            [s.user_goal for s in samples]
        ),
        "step_length_stats": _compute_length_stats(
            [s.plan_step for s in samples]
        ),
    }


def _compute_length_stats(texts: list[str]) -> dict:
    """Compute length distribution stats."""
    lengths = [len(t) for t in texts]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    lengths.sort()
    n = len(lengths)
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
    }


# ═════════════════════════════════════════════════════════════════════
# ── Serialization ───────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

def sample_to_dict(s: TrailPASSample) -> dict:
    """Convert TrailPASSample to JSON-serializable dict."""
    return {
        "sample_id": s.sample_id,
        "trace_id": s.trace_id,
        "source": s.source,
        "user_goal": s.user_goal,
        "plan_step": s.plan_step,
        "span_id": s.span_id,
        "alignment_error": s.alignment_error,
        "error_category": s.error_category,
        "error_description": s.error_description,
        "impact": s.impact,
        "evidence": s.evidence,
    }


def write_jsonl(samples: list[TrailPASSample], path: Path):
    """Write samples as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n")


# ═════════════════════════════════════════════════════════════════════
# ── Main ────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════

def main():
    print("TRAIL-PAS Extractor")
    print(f"  Source: {TRAIL_ROOT}")
    print(f"  Output: {OUTPUT_JSONL}")
    print()

    print("Extracting samples...")
    samples, extract_stats = extract_samples()

    print()
    print("=== Extraction Stats ===")
    print(f"  Annotations processed: {extract_stats['total_annotations']}")
    print(f"  Total errors in annotations: {extract_stats['total_errors_in_annotations']}")
    print(f"  Planning errors found: {extract_stats['planning_errors_found']}")
    print(f"  Positive samples: {extract_stats['positive_samples']}")
    print(f"  Negative samples: {extract_stats['negative_samples']}")
    print(f"  Traces with negatives: {extract_stats['traces_with_negatives']}")
    print(f"  Traces without trace JSON: {extract_stats['traces_without_trace_json']}")
    print(f"  TOTAL samples: {len(samples)}")

    if len(samples) < 160:
        print(f"  WARNING: Total samples ({len(samples)}) < 160 target!")
    else:
        print(f"  OK: Total samples ({len(samples)}) >= 160 target")

    # Compute detailed stats
    stats = compute_stats(samples)
    print()
    print("=== Sample Distribution ===")
    print(f"  Positive / Negative: {stats['positive_samples']} / {stats['negative_samples']}")
    print(f"  By source: {stats['by_source']}")
    print()
    print("=== Positive by Category ===")
    for cat, count in stats["positive_by_category"].items():
        print(f"  {cat}: {count}")
    print()
    print("=== Positive by Impact ===")
    for imp, count in stats["positive_by_impact"].items():
        print(f"  {imp}: {count}")
    print()
    print("=== Length Stats ===")
    gs = stats["goal_length_stats"]
    print(f"  user_goal: min={gs['min']} max={gs['max']} mean={gs['mean']:.0f} median={gs['median']}")
    ss = stats["step_length_stats"]
    print(f"  plan_step: min={ss['min']} max={ss['max']} mean={ss['mean']:.0f} median={ss['median']}")

    # Write outputs
    write_jsonl(samples, OUTPUT_JSONL)
    print(f"\nWrote {len(samples)} samples to {OUTPUT_JSONL}")

    # Write stats JSON
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump({**extract_stats, **stats}, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats to {OUTPUT_STATS}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
