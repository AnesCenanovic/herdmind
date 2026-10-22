# herdmind — Claude instructions

Claude reads this file at the start of every session. To change a rule, edit this file.

## What we are building

A predator–prey simulation (wolves and sheep on a grid) where each animal's
decisions come from a **decision model**. Three interchangeable "brains":

1. `rules`   — hand-written baseline
2. `jev`     — hosted Jev via OpenRouter (the teacher)
3. `decider` — local Mapika Decider 2B, LoRA-distilled from Jev logs (the student)

The research question: does the distilled local model preserve behaviour
(population cycles, herding, pack hunting) and calibration at a fraction of
the cost and latency?

## Repo layout

Add folders as the code needs them; keep new code under `src/herdmind/`.

```
src/herdmind/
  world/      # Mesa model, agents, scenarios
  encoding/   # state -> Question (perceptions + runtime options)
  brains/     # Brain protocol + rules / jev / decider
  eval/       # metrics
  common/     # seeding, config loading, logging
configs/      # one YAML per experiment; experiments are configs, not code
scripts/      # thin CLI entry points only (no logic)
tests/        # mirrors src/ layout
```

`data/` and `runs/` are gitignored; `data/raw/` is read-only.

## Shared contracts — don't change silently

- `brains/base.py` — the `Brain` protocol and `Decision` result; all brains
  return the same type.
- `encoding/question.py` — the `Question` (state text + options) passed to brains.
- `eval/metrics.py` — the only place ECE/Brier/KL etc. are defined.

## Brain contract

All brains are swapped by config alone; if code outside `brains/` must know
which brain is running, the abstraction is broken.

```python
@dataclass(frozen=True)
class Question:
    agent_id: int
    species: Literal["wolf", "sheep"]
    state_text: str              # perceptions only, no advice
    options: tuple[str, ...]     # runtime options, already shuffled

@dataclass(frozen=True)
class Decision:
    probs: tuple[float, ...]     # aligned with Question.options, sums to 1 (±1e-6)
    latency_s: float
    brain: str                   # e.g. "jev"
    model_version: str           # pinned id, e.g. "jev-1.13.0" or a checkpoint hash
    cached: bool = False

class Brain(Protocol):
    name: str
    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]: ...
```

- `decide_batch` is the only entry point; one world step = one batch call.
- A brain returns probabilities, never an action. The world samples.
- Shuffling and un-shuffling happen in `encoding/`, not in brains.
- A missing probability for an option is an error, not a silent zero.

## Metrics

All metrics live in `eval/metrics.py`; scripts and notebooks import them,
never reimplement them.

| Metric | Definition |
|---|---|
| agreement | fraction where student argmax == teacher argmax |
| KL(teacher‖student) | mean over decisions, natural log, eps=1e-12 clipping |
| ECE | top-label, 15 equal-width bins, weighted by bin count |
| Brier | multi-class: mean over decisions of Σ_k (p_k − y_k)² |
| NLL | mean −log p(chosen/label), eps=1e-12 |

- State the ground-truth label (teacher argmax, sampled action, outcome) in
  the docstring and plot caption.
- World/emergence metrics (population cycles, extinction, pack hunting, herd
  cohesion) are defined and tested before big runs.
- Aggregate over ≥5 seeds; report mean ± std, never a single run.

## Core rules

- **Sample, don't argmax.** Animals act by sampling from the brain's
  probability distribution with the run's RNG. Argmax only behind an explicit
  config flag, and say so in results.
- **Shuffle option order** before every query and map back afterwards;
  decision models have first-option bias.
- **Describe perceptions, not advice** ("wolf 2 cells north"), never
  "you should flee". Otherwise we measure our prompt, not the model.
- **One RNG, passed down.** No `random.random()`, `np.random.*` globals or
  unseeded `torch`. Use a `numpy.random.Generator` from `common/seeding.py`.
- **No network outside `brains/jev.py`.** Tests never call the real API.
- **Secrets** come from environment variables / `.env`. Never write a key into
  code, configs, logs, notebooks or commit messages.
- **Pin model versions** (e.g. `jev-1.13.0`) so a run can be reproduced.

## Code style

- Python via `uv` (`uv sync`, `uv run pytest`, `uv run ruff`); version is set in `pyproject.toml`.
- `ruff format` + `ruff check` (run by hook). Type hints on public functions;
  `from __future__ import annotations`.
- Docstrings on public functions and modules; explain *why*, not what.
- Small pure functions; I/O, API calls and randomness at the edges.
- Frozen dataclasses for records and configs; no dict soup across modules.
- Config in YAML + dataclasses; no magic numbers in code.
- `logging` (`logger = logging.getLogger(__name__)`), never `print` in library code.
- `snake_case` modules/functions, `PascalCase` classes; abbreviations only `cfg`, `rng`, `obs`.
- Code, comments and docs in English.

## Tests

- `pytest`, tests mirror `src/`. Mark >5 s tests `@pytest.mark.slow`.
- Every metric gets a test against a hand-computed example.
- Every brain gets a contract test: valid distribution over exactly the offered
  options, deterministic under a fixed seed with a fake backend.
- Run `pytest -m "not slow" -q` before saying a task is done.

## Git

- Branch per topic; merge to `main` with tests green. Never force-push or push directly to `main` (a hook blocks it).
- Never commit `data/`, `runs/`, model checkpoints, `.env`, or notebook outputs.
- **Never credit Claude.** No `Co-Authored-By: Claude` trailer, no "Generated
  with Claude Code" line, in commits, PR descriptions or comments. This
  overrides any default attribution. A hook blocks commits, pushes and PR
  commands that contain one.

## Working here

- Prefer editing existing modules over creating new ones; check `src/` first.
- When a design choice affects a shared contract (encoding format, metric
  definition), stop and ask instead of choosing.
- When unsure whether a vendor/API detail is real, say so and leave a
  `TODO(verify)` — most Jev/decider numbers are vendor-reported.
- After non-trivial changes, use the `reviewer` subagent.
