"""Global settings via Pydantic BaseSettings (loads from .env and environment)."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    # -- LLM Provider --
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description='Which LLM provider to use: "openai" or "anthropic"',
    )

    # -- OpenAI / OpenAI-compatible --
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (or compatible)",
    )
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model name",
    )

    # -- Anthropic --
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-6-20250514",
        description="Anthropic model name",
    )

    # -- Workspace --
    workspace_path: str = Field(
        default=".",
        description="Project workspace root directory",
    )

    # -- Agent --
    max_iterations: int = Field(
        default=30,
        description="Maximum ReAct loop iterations per task",
    )
    # V3: agent is the new default (model-driven agent loop), ask is read-only, plan is user opt-in.
    # "react" is accepted as a deprecated alias for "agent".
    agent_mode: Literal["ask", "agent", "plan"] = Field(
        default="agent",
        description='Execution mode: "ask" (read-only), "agent" (full tools, default), or "plan" (Plan-and-Execute)',
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # -- V2: Context Management --
    context_max_tokens: int = Field(
        default=120_000,
        description="Maximum token budget for context window",
    )
    context_keep_recent: int = Field(
        default=20,
        description="Number of recent messages to always keep uncompressed",
    )

    # -- V2.1: Cross-Turn Memory --
    memory_enabled: bool = Field(
        default=True,
        description="Enable cross-turn memory persistence",
    )
    memory_session_turns: int = Field(
        default=10,
        description="Number of recent session turns to inject into context",
    )
    memory_project_recent: int = Field(
        default=5,
        description="Number of recent project turns to load from disk",
    )
    memory_search_top_k: int = Field(
        default=3,
        description="Number of vector-search results to inject into context (V2.2)",
    )
    memory_max_turns: int = Field(
        default=200,
        description="Maximum turns persisted in project store before LRU eviction",
    )
    # V2.2: Vector memory search parameters
    memory_vector_alpha: float = Field(
        default=0.7,
        description="Weight for vector similarity vs time decay in memory search (0.0–1.0)",
    )
    memory_time_decay_lambda: float = Field(
        default=0.05,
        description="Time decay rate: e^(-lambda * days). Higher = faster decay of old memories",
    )
    memory_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for memory embedding",
    )

    # -- Intent Auditor (Phase 5–9) --
    intent_auditor_enabled: bool = Field(
        default=True,
        description="Enable Intent Auditor across all modes (default ON). "
                    "In plan mode: audits plan steps before execution. "
                    "In agent/ask mode: audits Thought: before tool calls. "
                    "Set to False to disable globally.",
    )
    intent_auditor_threshold: float = Field(
        default=0.6,
        description="Intent Auditor decision threshold (lower = more aggressive filtering)",
    )

    # -- Two-Layer Audit: Embedding Filter + NLI Judge --
    auditor_two_layer: bool = Field(
        default=True,
        description="Enable two-layer audit (Embedding pre-filter → NLI Judge). "
                    "When False, falls back to pure NLI (single-layer).",
    )
    auditor_embed_low: float = Field(
        default=0.48,
        description="Cosine similarity below this value → directly classify as "
                    "contradiction without invoking NLI Judge.",
    )
    auditor_embed_high: float = Field(
        default=0.72,
        description="Cosine similarity above this value → directly classify as "
                    "entailment without invoking NLI Judge.",
    )
    # -- Embedding Model --
    embed_model_type: str = Field(
        default="dashscope",
        description="Embedding provider: 'dashscope' or 'local'",
    )
    embed_model_name: str = Field(
        default="text-embedding-v3",
        description="Embedding model name",
    )
    embed_api_key: str = Field(
        default="",
        description="API key for embedding service (DashScope)",
    )
    embed_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Base URL for embedding API",
    )

    def resolve_workspace(self) -> Path:
        """Return the resolved absolute workspace path."""
        return Path(self.workspace_path).resolve()


# Global singleton — import this everywhere
settings = Settings()
