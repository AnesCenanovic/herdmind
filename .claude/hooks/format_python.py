#!/usr/bin/env python3
"""PostToolUse: run ruff format + ruff check --fix on edited Python files.

Remaining lint errors are reported back to Claude (exit 2) so it fixes them
in the same turn. Skips quietly if ruff is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return
    path = (event.get("tool_input") or {}).get("file_path", "")
    if not path.endswith((".py", ".pyi")):
        return

    ruff = shutil.which("ruff")
    cmd = [ruff] if ruff else None
    if cmd is None and shutil.which("uv"):
        cmd = ["uv", "run", "--quiet", "ruff"]
    if cmd is None:
        print("format_python.py: ruff not found, skipping format/lint", file=sys.stderr)
        return

    subprocess.run([*cmd, "format", "--quiet", path], capture_output=True)
    res = subprocess.run(
        [*cmd, "check", "--fix", "--quiet", "--output-format=concise", path],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 and res.stdout.strip():
        print(f"ruff found issues it could not auto-fix:\n{res.stdout.strip()}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
