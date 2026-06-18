"""Tests for TwoLayerAuditor (Embedding Filter + NLI Judge).

Uses mock embedding provider to avoid network calls.
"""

import json
import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from intent_auditor.embedding import (
    EmbedResult,
    BaseEmbeddingProvider,
    cosine_similarity,
)
from intent_auditor.intent_auditor import IntentAuditResult, is_predicted_error
from intent_auditor.two_layer import (
    TwoLayerAuditor,
    TwoLayerResult,
    create_two_layer_auditor,
)


# ═══════════════════════════════════════════════════════════════════
# Mock Embedding Provider
# ═══════════════════════════════════════════════════════════════════

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Fake embedding provider — returns pre-configured vectors.

    Each call returns the next vector from `vectors` or a fixed default.
    """

    def __init__(self, vectors=None, default_vector=None):
        self.vectors = vectors or []
        self.call_count = 0
        self.default_vector = default_vector or [1.0, 0.0, 0.0]  # normalized

    async def embed(self, text: str) -> EmbedResult:
        v = self._next_vector()
        return EmbedResult(vector=v, model="mock", latency_ms=1.0)

    async def embed_batch(self, texts):
        return [EmbedResult(vector=self._next_vector(), model="mock", latency_ms=0.5)
                for _ in texts]

    def _next_vector(self):
        if self.call_count < len(self.vectors):
            v = self.vectors[self.call_count]
            self.call_count += 1
            return v
        self.call_count += 1
        return self.default_vector


def mock_nli_result(label="entailment", score=0.9, reason="Mock NLI"):
    """Create a mock IntentAuditResult."""
    return IntentAuditResult(label=label, score=score, reason=reason)


# ═══════════════════════════════════════════════════════════════════
# Cosine similarity unit tests
# ═══════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        sim = cosine_similarity(v, v)
        assert math.isclose(sim, 1.0, abs_tol=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = cosine_similarity(a, b)
        assert math.isclose(sim, 0.0, abs_tol=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        sim = cosine_similarity(a, b)
        assert math.isclose(sim, -1.0, abs_tol=1e-6)

    def test_partial_similarity(self):
        a = [1.0, 0.0]
        b = [0.7071, 0.7071]  # ~45 degrees
        sim = cosine_similarity(a, b)
        assert 0.70 < sim < 0.71

    def test_zero_vector(self):
        sim = cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert sim == 0.0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════
# TwoLayerAuditor: Embedding fast-path tests
# ═══════════════════════════════════════════════════════════════════

class TestTwoLayerEmbedPath:
    """Verify embedding-only decisions (no NLI call needed)."""

    @pytest.mark.asyncio
    async def test_high_similarity_returns_entailment_via_embed(self):
        """cosine_sim > high threshold → entailment from embed layer only."""
        goal_vec = [1.0, 0.0, 0.0]
        step_vec = [0.99, 0.01, 0.0]    # ~0.99 cosine sim (very high)
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit(
            goal="Fix import error in main.py",
            plan_step="Read main.py to understand imports",
        )

        assert result.path == "embed"
        assert result.label == "entailment"
        assert result.score > 0.9
        assert result.cosine_sim > 0.9
        assert result.nli_result is None

    @pytest.mark.asyncio
    async def test_low_similarity_returns_contradiction_via_embed(self):
        """cosine_sim < low threshold → contradiction from embed layer only."""
        goal_vec = [1.0, 0.0, 0.0]
        step_vec = [0.0, 1.0, 0.0]      # 0.0 cosine sim (orthogonal)
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit(
            goal="Fix import error in main.py",
            plan_step="Update the database schema",
        )

        assert result.path == "embed"
        assert result.label == "contradiction"
        assert result.score == 0.1
        assert result.cosine_sim < 0.35
        assert result.nli_result is None

    @pytest.mark.asyncio
    async def test_disabled_falls_back_to_nli(self):
        """When enabled=False, always use NLI path (embed layer skipped)."""
        provider = MockEmbeddingProvider(
            vectors=[[1.0, 0.0], [0.5, 0.5]]
        )

        # Pre-compute the NLI result (simulate mock LLM)
        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "entailment", "score": 0.95,
                    "reason": "Step aligned with goal via NLI.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            llm=MockLLM(),
            enabled=False,  # DISABLED → pure NLI
        )

        result = await auditor.audit(
            goal="Fix import error",
            plan_step="Read main.py",
        )

        assert result.path == "nli"
        assert result.label == "entailment"
        assert result.nli_result is not None


# ═══════════════════════════════════════════════════════════════════
# TwoLayerAuditor: Gray zone → NLI fallback
# ═══════════════════════════════════════════════════════════════════

class TestTwoLayerNLIFallback:
    """Verify gray-zone steps invoke the NLI judge."""

    @pytest.mark.asyncio
    async def test_gray_zone_calls_nli(self):
        """cosine_sim between thresholds → NLI is called."""
        goal_vec = [1.0, 0.0, 0.0]
        step_vec = [0.6, 0.8, 0.0]      # ~0.6 cosine sim (gray zone)
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "entailment", "score": 0.78,
                    "reason": "Step is reasonably aligned.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit(
            goal="Fix bug",
            plan_step="Check the code for issues",
        )

        assert result.path == "nli"
        assert result.label == "entailment"
        assert result.nli_result is not None
        assert 0.35 <= result.cosine_sim <= 0.82

    @pytest.mark.asyncio
    async def test_nli_contradiction_preserved(self):
        """Gray zone + NLI says contradiction → contradiction propagated."""
        goal_vec = [1.0, 0.0, 0.0]
        step_vec = [0.6, 0.8, 0.0]
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "contradiction", "score": 0.15,
                    "reason": "Step is completely misaligned with goal.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit(
            goal="Fix import error",
            plan_step="Redesign the entire database schema",
        )

        assert result.path == "nli"
        assert result.label == "contradiction"
        assert result.nli_result is not None


# ═══════════════════════════════════════════════════════════════════
# TwoLayerAuditor: Batch audit
# ═══════════════════════════════════════════════════════════════════

class TestTwoLayerBatch:
    """Verify batch audit correctly splits embed vs NLI paths."""

    @pytest.mark.asyncio
    async def test_batch_mixed_paths(self):
        """3 pairs: high-sim (embed), low-sim (embed), mid-sim (NLI).

        Uses a text-aware mock that returns vectors based on prefix
        (goal vs step) to ensure correct alignment in batch calls.
        """
        # Text-aware mock: "goal_X" → vector, "step_X" → different vector
        class TextAwareProvider(BaseEmbeddingProvider):
            async def embed(self, text: str) -> EmbedResult:
                if text.startswith("goal_high"):
                    return EmbedResult(vector=[1.0, 0.0, 0.0])
                elif text.startswith("step_aligned"):
                    return EmbedResult(vector=[0.99, 0.01, 0.0])
                elif text.startswith("goal_low"):
                    return EmbedResult(vector=[1.0, 0.0, 0.0])
                elif text.startswith("step_ortho"):
                    return EmbedResult(vector=[0.0, 1.0, 0.0])
                elif text.startswith("goal_mid"):
                    return EmbedResult(vector=[1.0, 0.0, 0.0])
                elif text.startswith("step_mid"):
                    return EmbedResult(vector=[0.5, 0.87, 0.0])  # ~0.5 sim
                return EmbedResult(vector=[0.5, 0.5, 0.0])

            async def embed_batch(self, texts):
                results = []
                for t in texts:
                    results.append(await self.embed(t))
                return results

        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "neutral", "score": 0.55,
                    "reason": "Borderline case.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=TextAwareProvider(),
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        results = await auditor.audit_batch([
            ("goal_high", "step_aligned"),  # ~0.99 → embed entailment
            ("goal_low", "step_ortho"),     # ~0.0  → embed contradiction
            ("goal_mid", "step_mid"),       # ~0.5  → NLI (gray zone)
        ])

        assert len(results) == 3
        # pair 0: high sim → embed entailment
        assert results[0].path == "embed"
        assert results[0].label == "entailment"
        # pair 1: low sim → embed contradiction
        assert results[1].path == "embed"
        assert results[1].label == "contradiction"
        # pair 2: mid sim → NLI
        assert results[2].path == "nli"
        assert results[2].nli_result is not None

    @pytest.mark.asyncio
    async def test_batch_empty_input(self):
        """Empty pairs → empty results."""
        auditor = TwoLayerAuditor(enabled=True)
        results = await auditor.audit_batch([])
        assert results == []


# ═══════════════════════════════════════════════════════════════════
# TwoLayerAuditor: Stats tracking
# ═══════════════════════════════════════════════════════════════════

class TestTwoLayerStats:
    """Verify audit statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_track_embed_vs_nli(self):
        """Embed-only decisions count toward embed_hits, NLI toward nli_calls."""
        goal_vec = [1.0, 0.0, 0.0]
        step_vec = [0.99, 0.01, 0.0]      # ~0.99 → embed path
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        await auditor.audit("goal", "step")
        await auditor.audit("goal", "step")

        stats = auditor.stats
        assert stats["total"] == 2
        assert stats["embed_hits"] == 2
        assert stats["nli_calls"] == 0
        assert auditor.nli_bypass_rate == 1.0

    @pytest.mark.asyncio
    async def test_empty_goal_returns_neutral(self):
        """Empty goal or step → neutral, no embedding call."""
        auditor = TwoLayerAuditor(enabled=True)
        result = await auditor.audit("", "step")
        assert result.label == "neutral"
        assert result.path == "embed"
        assert auditor.stats["total"] == 1


# ═══════════════════════════════════════════════════════════════════
# TwoLayerAuditor: Threshold edge cases
# ═══════════════════════════════════════════════════════════════════

class TestTwoLayerThresholds:
    """Verify boundary behavior at threshold edges."""

    @pytest.mark.asyncio
    async def test_exactly_at_low_threshold_goes_to_nli(self):
        """cosine_sim == low threshold → NOT < low, goes to gray zone → NLI."""
        goal_vec = [1.0, 0.0, 0.0]
        # Vector producing exactly 0.35 sim: need a·b = 0.35 with |a|=1, |b|=1
        # One solution: b = [0.35, sqrt(1-0.35²), 0] = [0.35, ~0.9367, 0]
        y = math.sqrt(1 - 0.35**2)
        step_vec = [0.35, y, 0.0]
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "entailment", "score": 0.7,
                    "reason": "Aligned.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit("goal", "step")

        # sim == 0.35 is NOT < 0.35 → goes to gray zone → NLI
        assert result.path == "nli"
        assert math.isclose(result.cosine_sim, 0.35, abs_tol=0.01)

    @pytest.mark.asyncio
    async def test_exactly_at_high_threshold_goes_to_nli(self):
        """cosine_sim == high threshold → NOT > high, goes to gray zone → NLI."""
        goal_vec = [1.0, 0.0, 0.0]
        y = math.sqrt(1 - 0.82**2)
        step_vec = [0.82, y, 0.0]
        provider = MockEmbeddingProvider(vectors=[goal_vec, step_vec])

        class MockLLM:
            async def ainvoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content=json.dumps({
                    "label": "entailment", "score": 0.8,
                    "reason": "Aligned.",
                }))

        auditor = TwoLayerAuditor(
            embed_provider=provider,
            llm=MockLLM(),
            embed_low=0.35,
            embed_high=0.82,
            enabled=True,
        )

        result = await auditor.audit("goal", "step")

        # sim == 0.82 is NOT > 0.82 → goes to gray zone → NLI
        assert result.path == "nli"
        assert math.isclose(result.cosine_sim, 0.82, abs_tol=0.01)
