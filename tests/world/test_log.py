import dataclasses
import json

import pyarrow.parquet as pq
import pytest

from herdmind.world.log import (
    DECISION_SCHEMA,
    DECISIONS_FILE,
    DecisionRecord,
    RunLog,
    export_decisions,
)

IDS = {"world": "micro", "world_seed": 1, "seed": 2}


def _record(step: int) -> DecisionRecord:
    return DecisionRecord(
        step=step,
        agent_id=7,
        species="sheep",
        brain="rules",
        model_version="rules-0.2.0",
        state_text="wolf 2 cells north",
        options=("stay", "move north"),
        probs=(0.25, 0.75),
        chosen="move north",
        latency_s=0.5,
        cached=False,
    )


def _write_log(run_dir, steps: int) -> None:
    with RunLog(run_dir) as log:
        for step in range(steps):
            log.decision(_record(step))


def test_export_keeps_every_field_and_adds_run_ids(tmp_path):
    _write_log(tmp_path, 1)
    table = pq.read_table(export_decisions(tmp_path, **IDS))
    assert table.schema.equals(DECISION_SCHEMA)
    assert table.to_pylist() == [
        {
            "world": "micro",
            "world_seed": 1,
            "seed": 2,
            "run": tmp_path.name,
            "step": 0,
            "agent_id": 7,
            "species": "sheep",
            "brain": "rules",
            "model_version": "rules-0.2.0",
            "state_text": "wolf 2 cells north",
            "options": ["stay", "move north"],
            "probs": [0.25, 0.75],
            "chosen": "move north",
            "latency_s": 0.5,
            "cached": False,
        }
    ]


def test_export_writes_in_batches_without_losing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("herdmind.world.log.PARQUET_BATCH_ROWS", 2)
    _write_log(tmp_path, 5)
    table = pq.read_table(export_decisions(tmp_path, **IDS))
    assert table.column("step").to_pylist() == [0, 1, 2, 3, 4]


def test_export_of_an_empty_log_is_an_empty_table_with_the_schema(tmp_path):
    _write_log(tmp_path, 0)
    table = pq.read_table(export_decisions(tmp_path, **IDS))
    assert table.num_rows == 0 and table.schema.equals(DECISION_SCHEMA)


def test_export_drops_the_cut_off_last_line_of_a_killed_run(tmp_path):
    _write_log(tmp_path, 2)
    with (tmp_path / DECISIONS_FILE).open("a", encoding="utf-8") as f:
        f.write('{"step": 2, "agent_')
    table = pq.read_table(export_decisions(tmp_path, **IDS))
    assert table.column("step").to_pylist() == [0, 1]


def test_schema_is_the_run_ids_plus_every_record_field():
    fields = [f.name for f in dataclasses.fields(DecisionRecord)]
    assert DECISION_SCHEMA.names == ["world", "world_seed", "seed", "run", *fields]


def test_export_fails_on_a_damaged_line_in_the_middle(tmp_path):
    _write_log(tmp_path, 1)
    with (tmp_path / DECISIONS_FILE).open("a", encoding="utf-8") as f:
        f.write("not json\n" + json.dumps({"step": 1}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        export_decisions(tmp_path, **IDS)
    # No half-written dataset is left behind.
    assert sorted(p.name for p in tmp_path.glob("*.parquet*")) == []


def test_export_fails_on_a_record_missing_a_column(tmp_path):
    _write_log(tmp_path, 0)
    (tmp_path / DECISIONS_FILE).write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError):
        export_decisions(tmp_path, **IDS)
