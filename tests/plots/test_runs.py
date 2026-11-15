from herdmind.plots.runs import find_runs, group_by_name, load_run


def test_find_runs_loads_name_seed_and_boards(runs_dir):
    runs = find_runs(runs_dir)
    assert sorted((r.name, r.seed) for r in runs) == [("a", 0), ("a", 1), ("b", 0)]
    a0 = next(r for r in runs if (r.name, r.seed) == ("a", 0))
    assert [b.count("sheep") for b in a0.boards] == [4, 6, 2]
    assert a0.commit and isinstance(a0.dirty, bool)


def test_find_runs_filters_by_name_and_skips_non_runs(runs_dir):
    (runs_dir / "figures").mkdir()  # an output folder next to runs is not a run
    assert {r.name for r in find_runs(runs_dir, ["b"])} == {"b"}
    assert len(find_runs(runs_dir)) == 3


def test_group_by_name(runs_dir):
    groups = group_by_name(find_runs(runs_dir))
    assert {k: len(v) for k, v in groups.items()} == {"a": 2, "b": 1}


def test_load_run_single_folder(runs_dir):
    path = next(p for p in runs_dir.iterdir() if p.name.startswith("b-"))
    assert load_run(path).name == "b"
