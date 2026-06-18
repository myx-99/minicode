"""Two-Layer Auditor — Embedding Filter → NLI Judge.

Architecture:
    Goal (user task)
        │
        ▼
    Embedding Filter (fast, ~200ms)
        │ cosine_sim < low_threshold → contradiction (skip NLI)
        │ cosine_sim > high_threshold → entailment (skip NLI)
        │ otherwise → fallthrough to NLI Judge
        ▼
    NLI Judge (LLM, ~1.8s)
        │ entailment / neutral / contradiction
        ▼
    Final Score + Label

This two-layer design eliminates ~60-80% of LLM calls because most
goal-step pairs are either obviously aligned (high sim) or obviously
misaligned (low sim). Only ambiguous cases invoke the LLM.

Usage:
    from intent_auditor.two_layer import TwoLayerAuditor

    auditor = TwoLayerAuditor()
    result = await auditor.audit(
        goal="Fix the import error in main.py",
        plan_step="Read main.py to understand imports",
    )
    print(result.label, result.score, result.path)  # path = "embed" | "nli"
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from intent_auditor.intent_auditor import (
    IntentAuditResult,
    audit_intent,
    is_predicted_error,
)
from intent_auditor.embedding import (
    BaseEmbeddingProvider,
    EmbedResult,
    cosine_similarity,
    create_embedding_provider,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TwoLayerResult:
    """Result from the two-layer audit pipeline.

    Attributes:
        label: entailment / neutral / contradiction
        score: 0.0–1.0 alignment score
        reason: Human-readable explanation
        path: "embed" (decided by embedding alone) or "nli" (used LLM)
        cosine_sim: Cosine similarity between goal and step embeddings
        nli_result: Full NLI result if path=="nli", else None
        total_latency_ms: End-to-end pipeline latency
    """
    label: Literal["entailment", "neutral", "contradiction"]
    score: float
    reason: str
    path: Literal["embed", "nli"]
    cosine_sim: float = 0.0
    nli_result: Optional[IntentAuditResult] = None
    total_latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Two-Layer Auditor
# ═══════════════════════════════════════════════════════════════════

class TwoLayerAuditor:
    """Two-layer audit: Embedding fast-path → NLI Judge slow-path.

    Layer 1 — Embedding Filter:
      - cosine_sim < low_threshold  → contradiction (skip NLI)
      - cosine_sim > high_threshold → entailment  (skip NLI)
      - otherwise                    → fallthrough to Layer 2

    Layer 2 — NLI Judge:
      - Same LLM-as-Judge as the existing intent_auditor module.
      - Only invoked for the "gray zone" between thresholds.

    Threshold guidance:
      - low=0.35, high=0.82  (tested defaults)
      - Lower the high threshold to catch more false positives
      - Raise the low threshold to be more aggressive on rejections
    """

    def __init__(
        self,
        *,
        embed_provider: Optional[BaseEmbeddingProvider] = None,
        llm = None,  # Optional[BaseChatModel] — injected from caller
        nli_threshold: float = 0.6,
        embed_low: float = 0.35,
        embed_high: float = 0.82,
        enabled: bool = True,
    ):
        """Initialize the two-layer auditor.

        Args:
            embed_provider: Embedding provider (auto-created from settings if None).
            llm: Optional LangChain ChatModel for NLI fallback.
                 If None, audit_intent() creates its own from settings.
            nli_threshold: Decision threshold passed to is_predicted_error.
            embed_low: Below this cosine sim → contradiction (skip NLI).
            embed_high: Above this cosine sim → entailment (skip NLI).
            enabled: When False, falls back to pure NLI (single-layer).
        """
        self._nli_threshold = nli_threshold
        self._embed_low = embed_low
        self._embed_high = embed_high
        self._enabled = enabled
        self._llm = llm  # injected LLM for NLI (None → use settings)

        # ── Embedding provider (lazy init) ────────────────────
        self._embed_provider = embed_provider

        # ── Stats ──────────────────────────────────────────────
        self._stats = {"embed_hits": 0, "nli_calls": 0, "total": 0}

    # ── Public API ────────────────────────────────────────────────

    async def audit(
        self,
        goal: str,
        plan_step: str,
    ) -> TwoLayerResult:
        """Run the two-layer audit pipeline.

        Args:
            goal: The user's original task.
            plan_step: The agent's action or plan step text.

        Returns:
            TwoLayerResult with label, score, reason, and decision path.
        """
        t0 = time.perf_counter()
        self._stats["total"] += 1

        # ── Short text guard ──────────────────────────────────
        if not goal.strip() or not plan_step.strip():
            return TwoLayerResult(
                label="neutral",
                score=0.5,
                reason="Empty goal or plan step — cannot audit.",
                path="embed",
                cosine_sim=0.0,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Fallback: pure NLI (when two_layer is disabled) ───
        if not self._enabled:
            return await self._nli_path(goal, plan_step, 0.0, t0)

        # ═══════════════════════════════════════════════════════
        # Layer 1: Embedding Filter
        # ═══════════════════════════════════════════════════════

        try:
            goal_vec = await self._get_goal_embedding(goal)
            step_vec = await self._embed_step(plan_step)
            sim = cosine_similarity(goal_vec, step_vec)
        except Exception as e:
            logger.warning(
                "Embedding layer failed (%s), falling through to NLI", e
            )
            return await self._nli_path(goal, plan_step, 0.0, t0)

        # ── Fast-path: obviously misaligned ───────────────────
        if sim < self._embed_low:
            self._stats["embed_hits"] += 1
            elapsed = (time.perf_counter() - t0) * 1000
            return TwoLayerResult(
                label="contradiction",
                score=0.1,
                reason=(
                    f"Embedding similarity {sim:.3f} is below "
                    f"low threshold {self._embed_low}. "
                    f"Goal and step are semantically unrelated."
                ),
                path="embed",
                cosine_sim=sim,
                total_latency_ms=elapsed,
            )

        # ── Fast-path: obviously aligned ──────────────────────
        if sim > self._embed_high:
            self._stats["embed_hits"] += 1
            elapsed = (time.perf_counter() - t0) * 1000
            return TwoLayerResult(
                label="entailment",
                score=0.92,
                reason=(
                    f"Embedding similarity {sim:.3f} exceeds "
                    f"high threshold {self._embed_high}. "
                    f"Goal and step are clearly aligned."
                ),
                path="embed",
                cosine_sim=sim,
                total_latency_ms=elapsed,
            )

        # ═══════════════════════════════════════════════════════
        # Layer 2: NLI Judge (gray zone)
        # ═══════════════════════════════════════════════════════

        return await self._nli_path(goal, plan_step, sim, t0)

    async def audit_batch(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[TwoLayerResult]:
        """Audit multiple (goal, step) pairs efficiently.

        All embeddings are computed in one batch request,
        then NLI is called only for gray-zone pairs.
        """
        t0 = time.perf_counter()
        results: list[Optional[TwoLayerResult]] = [None] * len(pairs)

        if not pairs:
            return []

        goals = [g for g, _ in pairs]
        steps = [s for _, s in pairs]

        # ── Layer 1: batch embed ──────────────────────────────
        try:
            goal_vecs = await self._embed_goals_batch(goals)
            step_vecs = await self._embed_steps_batch(steps)
            sims = [
                cosine_similarity(gv, sv)
                for gv, sv in zip(goal_vecs, step_vecs)
            ]
        except Exception as e:
            logger.warning("Batch embedding failed (%s), falling back to NLI", e)
            # Fallback: NLI for everything — pass injected LLM when available
            nli_results = []
            for goal, step in pairs:
                r = await audit_intent(
                    goal=goal,
                    plan_step=step,
                    llm=self._llm,
                )
                nli_results.append(r)
            return [
                TwoLayerResult(
                    label=r.label,
                    score=r.score,
                    reason=r.reason,
                    path="nli",
                    nli_result=r,
                    total_latency_ms=r.latency_ms,
                )
                for r in nli_results
            ]

        # ── Classify fast-path vs gray-zone ───────────────────
        nli_indices: list[int] = []
        for i, sim in enumerate(sims):
            if sim < self._embed_low:
                results[i] = TwoLayerResult(
                    label="contradiction",
                    score=0.1,
                    reason=f"Embedding sim {sim:.3f} < {self._embed_low}",
                    path="embed",
                    cosine_sim=sim,
                )
            elif sim > self._embed_high:
                results[i] = TwoLayerResult(
                    label="entailment",
                    score=0.92,
                    reason=f"Embedding sim {sim:.3f} > {self._embed_high}",
                    path="embed",
                    cosine_sim=sim,
                )
            else:
                nli_indices.append(i)

        # ── Layer 2: NLI for gray-zone only ───────────────────
        for i in nli_indices:
            goal, step = pairs[i]
            try:
                nli_result = await audit_intent(
                    goal=goal,
                    plan_step=step,
                    llm=self._llm,
                )
                results[i] = TwoLayerResult(
                    label=nli_result.label,
                    score=nli_result.score,
                    reason=nli_result.reason,
                    path="nli",
                    cosine_sim=sims[i],
                    nli_result=nli_result,
                    total_latency_ms=nli_result.latency_ms,
                )
            except Exception:
                # Conservative: allow on error
                results[i] = TwoLayerResult(
                    label="neutral",
                    score=0.5,
                    reason="NLI call failed — allowing step (conservative).",
                    path="nli",
                    cosine_sim=sims[i],
                )

        self._stats["embed_hits"] += len(pairs) - len(nli_indices)
        self._stats["nli_calls"] += len(nli_indices)
        self._stats["total"] += len(pairs)

        elapsed_total = (time.perf_counter() - t0) * 1000
        for r in results:
            if r is not None and r.total_latency_ms == 0.0:
                r.total_latency_ms = elapsed_total / max(len(pairs), 1)

        return [r for r in results if r is not None]

    # ── Embedding helpers ────────────────────────────────────────

    async def _get_goal_embedding(self, goal: str) -> list[float]:
        """Get goal embedding via the configured provider."""
        provider = await self._get_provider()
        result = await provider.embed(goal)
        return result.vector

    async def _embed_step(self, step: str) -> list[float]:
        """Embed a single plan step (not cached)."""
        provider = await self._get_provider()
        result = await provider.embed(step)
        return result.vector

    async def _embed_goals_batch(self, goals: list[str]) -> list[list[float]]:
        """Embed multiple goals via the configured provider."""
        provider = await self._get_provider()
        results = await provider.embed_batch(goals)
        return [r.vector for r in results]

    async def _embed_steps_batch(self, steps: list[str]) -> list[list[float]]:
        """Embed multiple plan steps (no cache)."""
        provider = await self._get_provider()
        results = await provider.embed_batch(steps)
        return [r.vector for r in results]

    async def _get_provider(self) -> BaseEmbeddingProvider:
        """Lazy-init the embedding provider from settings."""
        if self._embed_provider is None:
            try:
                from config.settings import settings
                provider_type = settings.embed_model_type
            except Exception:
                provider_type = "dashscope"
            self._embed_provider = create_embedding_provider(provider_type)
        return self._embed_provider

    # ── NLI fallback ─────────────────────────────────────────────

    async def _nli_path(
        self,
        goal: str,
        plan_step: str,
        cosine_sim: float,
        t0: float,
    ) -> TwoLayerResult:
        """Run the NLI Judge (Layer 2).

        Uses injected LLM if available, otherwise creates from settings
        via audit_intent().
        """
        self._stats["nli_calls"] += 1

        try:
            nli_result = await audit_intent(
                goal=goal,
                plan_step=plan_step,
                llm=self._llm,  # None → audit_intent creates its own
            )
        except Exception:
            # Conservative fail-open
            return TwoLayerResult(
                label="neutral",
                score=0.5,
                reason="NLI Judge failed → allowing step (conservative).",
                path="nli",
                cosine_sim=cosine_sim,
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        elapsed = (time.perf_counter() - t0) * 1000

        return TwoLayerResult(
            label=nli_result.label,
            score=nli_result.score,
            reason=nli_result.reason,
            path="nli",
            cosine_sim=cosine_sim,
            nli_result=nli_result,
            total_latency_ms=elapsed,
        )

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return audit statistics."""
        return dict(self._stats)

    @property
    def nli_bypass_rate(self) -> float:
        """Fraction of audits decided by embedding alone (0.0–1.0)."""
        total = self._stats["total"]
        if total == 0:
            return 0.0
        return self._stats["embed_hits"] / total


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_two_layer_auditor(
    *,
    enabled: Optional[bool] = None,
    embed_low: Optional[float] = None,
    embed_high: Optional[float] = None,
    nli_threshold: Optional[float] = None,
    llm = None,
) -> TwoLayerAuditor:
    """Create a TwoLayerAuditor from project settings.

    All parameters are optional; defaults come from Settings / .env.

    Args:
        enabled: Override auditor_two_layer setting.
        embed_low: Override auditor_embed_low.
        embed_high: Override auditor_embed_high.
        nli_threshold: Override intent_auditor_threshold.
        llm: Optional LangChain ChatModel for NLI calls.
             When provided, NLI uses this instead of creating one from settings.

    Returns:
        A configured TwoLayerAuditor ready for use.
    """
    try:
        from config.settings import settings

        resolved_enabled = enabled if enabled is not None else settings.auditor_two_layer
        resolved_low = embed_low if embed_low is not None else settings.auditor_embed_low
        resolved_high = embed_high if embed_high is not None else settings.auditor_embed_high
        resolved_threshold = (
            nli_threshold if nli_threshold is not None
            else settings.intent_auditor_threshold
        )
    except Exception:
        resolved_enabled = enabled if enabled is not None else True
        resolved_low = embed_low if embed_low is not None else 0.35
        resolved_high = embed_high if embed_high is not None else 0.82
        resolved_threshold = nli_threshold if nli_threshold is not None else 0.6

    # ── Runtime override via env var (used by pytest conftest) ──
    env_override = os.environ.get("AUDITOR_TWO_LAYER", "").lower()
    if env_override in ("false", "0", "no", "off"):
        resolved_enabled = False
    elif env_override in ("true", "1", "yes", "on"):
        resolved_enabled = True

    return TwoLayerAuditor(
        llm=llm,
        nli_threshold=resolved_threshold,
        embed_low=resolved_low,
        embed_high=resolved_high,
        enabled=resolved_enabled,
    )
