"""Tests for shell platform detection and prompt guidance."""

import platform

from runtime.shell_platform import get_shell_environment


def test_get_shell_environment_matches_platform():
    env = get_shell_environment()
    assert env.system == platform.system()
    if platform.system() == "Windows":
        assert env.is_windows
        assert "cmd" in env.shell_name.lower()
        assert "<<<" in env.tool_description_suffix
        assert "Windows" in env.agent_prompt_section
    else:
        assert not env.is_windows
        assert "bash" in env.shell_name.lower()
