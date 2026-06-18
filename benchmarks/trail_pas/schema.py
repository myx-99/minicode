"""TRAIL-PAS (Planning-Alignment Subset) schema.

Extracted from Patronus AI TRAIL dataset annotations and OpenTelemetry traces.
Each sample is a (goal, plan_step, alignment_error) triple for Intent Auditor evaluation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class TrailPASSample:
    """A single TRAIL-PAS evaluation sample.

    Attributes:
        sample_id: Unique ID "{trace_id}_{span_id}_{category_or_neg}"
        trace_id: TRAIL trace identifier
        source: "GAIA" or "SWE-Bench"
        user_goal: Extracted from trace's first user message (≤800 chars)
        plan_step: Agent action text — evidence for positives, Thought/Code span for negatives
        span_id: TRAIL annotation location or trace span_id
        alignment_error: Gold label — True if this step is a planning error
        error_category: TRAIL error category (empty for negatives)
        error_description: Expert annotation description
        impact: LOW / MEDIUM / HIGH
        evidence: Expert annotation evidence text
    """
    sample_id: str
    trace_id: str
    source: Literal["GAIA", "SWE-Bench"]
    user_goal: str
    plan_step: str
    span_id: str
    alignment_error: bool
    error_category: str = ""
    error_description: str = ""
    impact: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    evidence: str = ""


# ── Planning error categories (canonical forms) ─────────────────────

PLANNING_CATEGORIES = {
    "Goal Deviation",
    "Task Orchestration",
    "Resource Abuse",
    "Context Handling Failures",
    "Poor Information Retrieval",
}

# ── Category normalization map ──────────────────────────────────────
# Real-world annotations contain spelling/capitalization variants.
# This map normalizes them to the canonical form.

CATEGORY_NORMALIZE = {
    "goal deviation": "Goal Deviation",
    "goal deviation ": "Goal Deviation",
    "context handling failure": "Context Handling Failures",
    "task orchestration errors": "Task Orchestration",
    "task orchestration error": "Task Orchestration",
    "poor information retrieval": "Poor Information Retrieval",
    "resource abuse": "Resource Abuse",
    # Canonical forms map to themselves
    "goal deviation": "Goal Deviation",
    "context handling failures": "Context Handling Failures",
    "task orchestration": "Task Orchestration",
    "poor information retrieval": "Poor Information Retrieval",
}


def normalize_category(category: str) -> str:
    """Normalize a TRAIL error category string to canonical form.

    Handles observed variants:
      - "Goal deviation" → "Goal Deviation"
      - "Context Handling Failure" → "Context Handling Failures"
      - "Task Orchestration Errors" → "Task Orchestration"
      - "Poor Information retrieval" → "Poor Information Retrieval"
    """
    stripped = category.strip()
    # Try exact string match first (preserves canonical forms)
    if stripped in PLANNING_CATEGORIES:
        return stripped
    # Try lowercase lookup
    normalized = CATEGORY_NORMALIZE.get(stripped.lower())
    if normalized:
        return normalized
    # Fallback: return as-is
    return stripped


def is_planning_category(category: str) -> bool:
    """Check if a (normalized) category is a planning error category."""
    return normalize_category(category) in PLANNING_CATEGORIES
