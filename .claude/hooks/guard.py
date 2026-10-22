#!/usr/bin/env python3
"""PreToolUse guard: block edits/commands that would break shared repo rules.

Exit 2 = block the tool call; stderr is shown to Claude so it can adjust.
Exit 0 = allow. Never blocks on internal errors (fails open, but warns).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import PurePosixPath

PROTECTED_PATHS = [
    (re.compile(r"(^|/)data/raw/"), "data/raw/ is read-only source data"),
    (re.compile(r"(^|/)\.env(\..*)?$"), ".env files hold secrets; edit them by hand"),
    (re.compile(r"(^|/)runs/"), "runs/ is produced by scripts, not edited by hand"),
]

SECRET_PATTERNS = [
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"), "OpenRouter API key"),
    (re.compile(r"sk-[A-Za-z0-9_-]{32,}"), "API key"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (
        re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
        "hard-coded credential",
    ),
]

DANGEROUS_BASH = [
    (
        re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\.|\*|data\b|src\b)(\s|$)"),
        "recursive delete of a broad path",
    ),
    (re.compile(r"\bgit\s+push\b.*(--force|-f\b)"), "force push"),
    (re.compile(r"\bgit\s+push\b.*\bmain\b"), "direct push to main (use a branch)"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard discards work"),
    (re.compile(r"\bgit\s+add\b.*(\s|/)(data|runs)(/|\s|$)"), "adding data/ or runs/ to git"),
    (re.compile(r"\bgit\s+add\b.*\.env\b"), "adding .env to git"),
    (re.compile(r"\b(cat|less|head|tail|more)\s+\S*\.env\b"), "printing .env exposes secrets"),
]


CREDIT_PATTERN = re.compile(
    r"co-authored-by:\s*claude|generated with \[?claude|noreply@anthropic\.com", re.I
)
CREDIT_CMD = re.compile(r"\bgit\s+(commit|push|tag)\b|\bgh\s+pr\s+(create|edit|comment)\b")


def _unpushed_messages(cwd: str) -> str:
    """Messages of commits not yet on origin/main; empty if git can't tell."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%B", "origin/main..HEAD"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout if out.returncode == 0 else ""
    except Exception:  # fail open
        return ""


def _block(msg: str) -> None:
    print(f"Blocked by .claude/hooks/guard.py: {msg}", file=sys.stderr)
    sys.exit(2)


def _rel(path: str, root: str) -> str:
    p = path.replace("\\", "/")
    root = root.replace("\\", "/").rstrip("/")
    if root and p.startswith(root + "/"):
        p = p[len(root) + 1 :]
    return str(PurePosixPath(p))


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception as exc:  # fail open
        print(f"guard.py: could not parse hook input ({exc})", file=sys.stderr)
        return

    tool = event.get("tool_name", "")
    inp = event.get("tool_input", {}) or {}
    root = event.get("cwd", "")

    if tool in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
        path = _rel(inp.get("file_path") or inp.get("notebook_path") or "", root)
        for pattern, why in PROTECTED_PATHS:
            if pattern.search(path):
                _block(f"{path}: {why}.")
        text = " ".join(str(inp.get(k, "")) for k in ("content", "new_string", "new_source"))
        for e in inp.get("edits", []) or []:
            text += " " + str(e.get("new_string", ""))
        for pattern, what in SECRET_PATTERNS:
            if pattern.search(text):
                _block(f"looks like a {what} in {path}. Read it from an env var instead.")

    elif tool == "Bash":
        cmd = inp.get("command", "")
        for pattern, why in DANGEROUS_BASH:
            if pattern.search(cmd):
                _block(f"{why}. Ask the user to run this themselves if it is intended.")
        if CREDIT_CMD.search(cmd):
            text = cmd + (
                "\n" + _unpushed_messages(root) if re.search(r"\bgit\s+push\b", cmd) else ""
            )
            if CREDIT_PATTERN.search(text):
                _block(
                    "commits and PRs must not credit Claude (no Co-Authored-By or "
                    "'Generated with Claude Code' lines). Remove the credit and retry."
                )


if __name__ == "__main__":
    main()
