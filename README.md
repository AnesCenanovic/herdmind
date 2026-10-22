# herdmind
Emergent behaviour in ecosystem simulation via a distilled decision model (MPVI semester project)

## Development

```
uv sync            # install
uv run pytest -m "not slow" -q
uv run ruff check . && uv run ruff format --check .
```

Work on a branch and open a PR; `main` is protected and PRs are squash-merged. See `.claude/CLAUDE.md` for the project rules.
