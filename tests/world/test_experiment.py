import json

import pyarrow.parquet as pq

from herdmind.world.experiment import run_experiment
from herdmind.world.ledger import read_ledger
from herdmind.world.log import DECISION_SCHEMA
from herdmind.world.spec import WorldSpec


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_folder_holds_everything_needed_to_learn_from_it(tmp_path):
    run = run_experiment(WorldSpec(steps=5), tmp_path)
    ledger = read_ledger(run)
    assert len(ledger) == 6
    meta = json.loads((run / "meta.json").read_text())
    assert meta["brains"]["wolf"] == {"brain": "rules", "model_version": "rules-0.2.0"}
    assert "always kills" in (run / "world.txt").read_text()
    assert "You are a wolf" in json.loads((run / "briefings.json").read_text())["wolf"]

    decisions = _lines(run / "decisions.jsonl")
    assert decisions and {"state_text", "options", "probs", "chosen"} <= decisions[0].keys()
    assert all(d["chosen"] in d["options"] for d in decisions)
    events = _lines(run / "events.jsonl")
    deaths = sum(e["kind"] == "death" for e in events)
    births = sum(e["kind"] == "birth" for e in events)
    assert births - deaths == len(ledger[-1].animals)  # initial animals are step-0 births
    # Every queried animal is logged, including sheep eaten later in the same step.
    alive_at_start = sum(len(b.animals) for b in ledger[:-1])
    assert len(decisions) == alive_at_start


def test_run_exports_decisions_to_parquet_with_run_ids(tmp_path):
    spec = WorldSpec(steps=3, seed=4, world_seed=9)
    run = run_experiment(spec, tmp_path)
    table = pq.read_table(run / "decisions.parquet")
    assert table.schema.equals(DECISION_SCHEMA)
    ids = {"world": spec.name, "world_seed": 9, "seed": 4, "run": run.name}
    assert table.to_pylist() == [ids | rec for rec in _lines(run / "decisions.jsonl")]
