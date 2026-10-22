#!/usr/bin/env python3
"""Stop hook: if Python files changed, run the fast test suite before Claude
finishes. On failure, exit 2 so Claude keeps working on the fix.

Runs at most once per stop (respects stop_hook_active to avoid loops).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

TIMEOUT_S = 120


def _changed_python_files() -> bool:
    res = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return any(line.rstrip().endswith(".py") for line in res.stdout.splitlines())


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return
    if event.get("stop_hook_active"):
        return
    if not _changed_python_files():
        return

    if shutil.which("uv"):
        cmd = ["uv", "run", "--quiet", "pytest"]
    elif shutil.which("pytest"):
        cmd = ["pytest"]
    else:
        return
    cmd += ["-m", "not slow", "-q", "-x", "--no-header"]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"Fast tests exceeded {TIMEOUT_S}s; mark slow tests @pytest.mark.slow.",
              file=sys.stderr)
        sys.exit(2)

    # 5 = no tests collected: fine early in the project.
    if res.returncode not in (0, 5):
        tail = "\n".join(res.stdout.strip().splitlines()[-30:])
        print(f"Fast tests are failing — fix before finishing:\n{tail}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
