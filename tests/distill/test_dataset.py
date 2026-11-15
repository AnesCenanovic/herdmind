import json

import pytest

from herdmind.common.seeding import make_rng
from herdmind.distill.dataset import DistillRow, load_rows
from herdmind.world.experiment import run_experiment
from herdmind.world.spec import WorldSpec


def _write_run(run_dir, records, briefings=None):
    run_dir.mkdir()
    text = "".join(json.dumps(r) + "\n" for r in records)
    (run_dir / "decisions.jsonl").write_text(text, encoding="utf-8")
    if briefings is not None:
        (run_dir / "briefings.json").write_text(json.dumps(briefings), encoding="utf-8")
    return run_dir


def _record(**over):
    rec = {
        "species": "sheep",
        "state_text": "wolf 2 cells north",
        "options": ["stay", "move north"],
        "probs": [0.25, 0.75],
    }
    return {**rec, **over}


def test_rows_from_a_real_run_carry_briefing_state_and_teacher_distribution(tmp_path):
    run = run_experiment(WorldSpec(steps=3), tmp_path)
    logged = [json.loads(x) for x in (run / "decisions.jsonl").read_text().splitlines()]
    rows = load_rows(run, limit=10_000, rng=make_rng(0))
    assert len(rows) == len(logged)
    by_key = {(r.context, r.options): r for r in rows}
    briefings = json.loads((run / "briefings.json").read_text())
    first = logged[0]
    row = by_key[
        (f"{briefings[first['species']]}\n\n{first['state_text']}", tuple(first["options"]))
    ]
    assert row.probs == tuple(first["probs"])


def test_limit_and_seed_pick_a_reproducible_subset(tmp_path):
    records = [_record(state_text=f"state {i}") for i in range(20)]
    run = _write_run(tmp_path / "run", records)
    a = load_rows(run, limit=5, rng=make_rng(1))
    assert len(a) == 5 and len({r.context for r in a}) == 5
    assert a == load_rows(run, limit=5, rng=make_rng(1))
    assert a != load_rows(run, limit=5, rng=make_rng(2))


def test_briefing_logged_with_the_decision_wins_over_the_run_file(tmp_path):
    records = [_record(briefing="You are a sheep."), _record(state_text="grass here")]
    run = _write_run(tmp_path / "run", records, {"sheep": "Old briefing."})
    contexts = {r.context for r in load_rows(run, limit=2, rng=make_rng(0))}
    assert contexts == {"You are a sheep.\n\nwolf 2 cells north", "Old briefing.\n\ngrass here"}


def test_teacher_argmax_is_the_hard_label():
    assert DistillRow("ctx", ("a", "b", "c"), (0.2, 0.5, 0.3)).teacher_argmax == 1


def test_missing_probability_is_an_error(tmp_path):
    run = _write_run(tmp_path / "run", [_record(probs=[1.0])])
    with pytest.raises(ValueError, match="1 probs for 2 options"):
        load_rows(run, limit=1, rng=make_rng(0))


def test_empty_run_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no decisions"):
        load_rows(_write_run(tmp_path / "run", []), limit=1, rng=make_rng(0))
