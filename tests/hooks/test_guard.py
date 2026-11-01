"""The guard hook is the only thing stopping bad pushes and leaked keys; test its verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).parents[2] / ".claude" / "hooks" / "guard.py"


def verdict(tool: str, **tool_input: str) -> int:
    """Run guard.py as Claude Code would and return its exit code (2 = blocked)."""
    event = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": ""})
    return subprocess.run(
        [sys.executable, str(GUARD)], input=event, capture_output=True, text=True
    ).returncode


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin feat/x",
        "git push origin main",
        "git reset --hard HEAD~1",
        "git add .env",
        "cat .env",
        'git commit -m "x" -m "Co-Authored-By: Claude <noreply@anthropic.com>"',
    ],
)
def test_bash_blocked(command: str) -> None:
    assert verdict("Bash", command=command) == 2


@pytest.mark.parametrize("command", ["git status", "git push origin feat/x", 'git commit -m "x"'])
def test_bash_allowed(command: str) -> None:
    assert verdict("Bash", command=command) == 0


def test_env_edit_blocked() -> None:
    assert verdict("Write", file_path=".env", content="X=1") == 2


def test_key_in_code_blocked() -> None:
    fake_key = "sk-or-v1-" + "a" * 24  # built at runtime so this file holds no key-like text
    assert verdict("Write", file_path="src/x.py", content=f"K = '{fake_key}'") == 2


def test_normal_edit_allowed() -> None:
    assert verdict("Write", file_path="src/x.py", content="x = 1\n") == 0
