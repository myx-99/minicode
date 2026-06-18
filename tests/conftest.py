"""Pytest configuration — shared fixtures and environment setup for all tests.

Key behavior:
  - Tests that use mock LLMs assume single-layer NLI audit (one LLM call per step).
    Two-layer auditing (embedding filter) bypasses mock LLMs and makes real API calls,
    breaking mock alignment. So two-layer is DISABLED by default in tests.

  - Tests that explicitly test the two-layer path can enable it via the
    `enable_two_layer` fixture or by directly constructing a TwoLayerAuditor
    with a mock embedding provider.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def _disable_two_layer_in_tests(monkeypatch):
    """Disable two-layer auditing for all tests by default.

    Tests that explicitly test the two-layer path should re-enable it
    or construct a TwoLayerAuditor directly with a mock embedding provider.
    """
    monkeypatch.setenv("AUDITOR_TWO_LAYER", "false")
