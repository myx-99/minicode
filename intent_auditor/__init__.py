"""Intent Auditor — LLM-as-Judge Goal-Plan alignment checker.

Evaluates whether an agent's plan step directly serves the user's goal.
Uses TRAIL-style planning error intuition for contradiction detection.

V4: Two-layer audit (Embedding Filter → NLI Judge) for lower latency
    and reduced LLM costs.
"""

from intent_auditor.intent_auditor import (
    IntentAuditResult,
    audit_intent,
    audit_intent_sync,
    is_predicted_error,
    AUDITOR_SYSTEM_PROMPT,
)
from intent_auditor.embedding import (
    EmbedResult,
    BaseEmbeddingProvider,
    DashScopeEmbeddingProvider,
    LocalEmbeddingProvider,
    cosine_similarity,
    create_embedding_provider,
)
from intent_auditor.two_layer import (
    TwoLayerResult,
    TwoLayerAuditor,
    create_two_layer_auditor,
)

__all__ = [
    # Core NLI
    "IntentAuditResult",
    "audit_intent",
    "audit_intent_sync",
    "is_predicted_error",
    "AUDITOR_SYSTEM_PROMPT",
    # Embedding
    "EmbedResult",
    "BaseEmbeddingProvider",
    "DashScopeEmbeddingProvider",
    "LocalEmbeddingProvider",
    "cosine_similarity",
    "create_embedding_provider",
    # Two-Layer
    "TwoLayerResult",
    "TwoLayerAuditor",
    "create_two_layer_auditor",
]
