import json

from herdmind.world.experiment import run_experiment
from herdmind.world.spec import WorldSpec


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_folder_holds_everything_needed_to_learn_from_it(tmp_path):
    run = run_experiment(WorldSpec(steps=5), tmp_path)
    pops = json.loads((run / "populations.json").read_text())
    assert len(pops) == 6
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
    end = pops[-1]
    assert births - deaths == end["sheep"] + end["wolf"]  # initial animals are step-0 births
    # Every queried animal is logged, including sheep eaten later in the same step.
    alive_at_start = sum(p["sheep"] + p["wolf"] for p in pops[:-1])
    assert len(decisions) == alive_at_start
