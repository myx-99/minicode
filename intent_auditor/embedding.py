"""Embedding Provider — unified interface for DashScope and local models.

Supports:
  - DashScope text-embedding-v3 (cloud, via OpenAI-compatible API)
  - sentence-transformers (local, no API key needed)

All backends expose the same async callable interface so the two-layer
auditor can swap them without code changes.

Usage:
    from intent_auditor.embedding import create_embedding_provider

    provider = create_embedding_provider()
    vec = await provider.embed("修复登录 Bug")
    sim = cosine_similarity(vec1, vec2)  # 0.0–1.0
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urljoin


# ═══════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EmbedResult:
    """Result of a single embed() call."""
    vector: List[float]
    model: str = ""
    latency_ms: float = 0.0
    cached: bool = False


# ═══════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════

class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider — all backends inherit from this."""

    @abstractmethod
    async def embed(self, text: str) -> EmbedResult:
        """Embed a single text → vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        """Embed multiple texts in one request (where supported)."""
        ...


# ═══════════════════════════════════════════════════════════════════
# Cosine similarity
# ═══════════════════════════════════════════════════════════════════

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0–1.0 (vectors are L2-normalised first for safety).
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════
# DashScope (cloud — OpenAI-compatible /v1/embeddings)
# ═══════════════════════════════════════════════════════════════════

class DashScopeEmbeddingProvider(BaseEmbeddingProvider):
    """DashScope text-embedding-v3 via OpenAI-compatible API.

    Uses the same endpoint format as OpenAI /v1/embeddings.  Works with
    dashscope.aliyuncs.com/compatible-mode/v1 and similar backends.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v3",
        timeout: float = 10.0,
    ):
        self._api_key = api_key or os.environ.get("EMBED_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

        # Lazy import — only pay for aiohttp when this provider is used
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def embed(self, text: str) -> EmbedResult:
        t0 = time.perf_counter()
        await self._ensure_session()

        url = urljoin(self._base_url + "/", "embeddings")
        payload = {
            "model": self._model,
            "input": text,
            "encoding_format": "float",
        }

        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Embedding API error {resp.status}: {body[:500]}"
                )
            data = await resp.json()

        vector = data["data"][0]["embedding"]
        elapsed = (time.perf_counter() - t0) * 1000

        return EmbedResult(
            vector=vector,
            model=self._model,
            latency_ms=elapsed,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        """DashScope supports batch input natively."""
        t0 = time.perf_counter()
        await self._ensure_session()

        url = urljoin(self._base_url + "/", "embeddings")
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }

        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Embedding API error {resp.status}: {body[:500]}"
                )
            data = await resp.json()

        elapsed_total = (time.perf_counter() - t0) * 1000
        per_item = elapsed_total / max(len(data["data"]), 1)

        results = []
        for item in data["data"]:
            results.append(EmbedResult(
                vector=item["embedding"],
                model=self._model,
                latency_ms=per_item,
            ))
        return results

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None


# ═══════════════════════════════════════════════════════════════════
# Local (sentence-transformers — zero-cost, no network)
# ═══════════════════════════════════════════════════════════════════

class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local sentence-transformers model.

    Default: all-MiniLM-L6-v2 (384 dims, ~80 MB download, fast on CPU).
    Falls back gracefully if the library is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self._loaded = True

    async def embed(self, text: str) -> EmbedResult:
        import asyncio
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        vec = await loop.run_in_executor(None, self._embed_sync, text)
        elapsed = (time.perf_counter() - t0) * 1000

        return EmbedResult(
            vector=vec,
            model=self._model_name,
            latency_ms=elapsed,
        )

    def _embed_sync(self, text: str) -> List[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    async def embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        import asyncio
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()

        vecs = await loop.run_in_executor(None, self._embed_batch_sync, texts)
        elapsed_total = (time.perf_counter() - t0) * 1000
        per_item = elapsed_total / max(len(vecs), 1)

        return [
            EmbedResult(vector=v, model=self._model_name, latency_ms=per_item)
            for v in vecs
        ]

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return vecs.tolist()


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════

def create_embedding_provider(
    provider_type: str = "dashscope",
) -> BaseEmbeddingProvider:
    """Create an embedding provider from config or defaults.

    Args:
        provider_type: "dashscope" or "local"

    Returns:
        A BaseEmbeddingProvider instance ready for use.

    Raises:
        ValueError: If provider_type is unknown.
    """
    if provider_type == "dashscope":
        try:
            from config.settings import settings
            return DashScopeEmbeddingProvider(
                api_key=settings.embed_api_key,
                base_url=settings.embed_base_url,
                model=settings.embed_model_name,
            )
        except Exception:
            # Fall back to env vars if settings not available
            return DashScopeEmbeddingProvider()
    elif provider_type == "local":
        try:
            from config.settings import settings
            model = settings.embed_model_name or "all-MiniLM-L6-v2"
        except Exception:
            model = "all-MiniLM-L6-v2"
        return LocalEmbeddingProvider(model_name=model)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            f"Use 'dashscope' or 'local'."
        )

