"""Tests for world and emergence metrics on synthetic trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from herdmind.eval.world_metrics import (
    alarm_call_survival_effect,
    evaluate_run_dir,
    extinction_info,
    herd_cohesion,
    lotka_volterra_fit,
    pack_hunting_rate,
    population_cycle_metrics,
    population_series,
)
from herdmind.world.ledger import AnimalState, BoardState
from herdmind.world.log import EventRecord


def _board(step: int, animals: list[AnimalState], size: int = 10) -> BoardState:
    grass = tuple("#" * size for _ in range(size))
    regrow = tuple(tuple(0 for _ in range(size)) for _ in range(size))
    return BoardState(step=step, grass=grass, regrow=regrow, animals=tuple(animals))


def test_population_series_and_extinction() -> None:
    b0 = _board(
        0,
        [AnimalState(1, "sheep", 1, 1, 10), AnimalState(2, "wolf", 5, 5, 20)],
    )
    b1 = _board(1, [AnimalState(2, "wolf", 1, 1, 25)])
    b2 = _board(2, [])

    sheep, wolves = population_series([b0, b1, b2])
    assert tuple(sheep) == (1, 0, 0)
    assert tuple(wolves) == (1, 1, 0)
    assert extinction_info([b0]) == (False, None, None)
    assert extinction_info([b0, b1, b2]) == (True, 1, "sheep")


def test_lotka_volterra_fit_on_exact_discrete_trajectory() -> None:
    s = [40]
    w = [10]
    for _ in range(25):
        s_next = max(1, round(s[-1] * (1.0 + 0.20 - 0.01 * w[-1])))
        w_next = max(1, round(w[-1] * (1.0 + 0.005 * s[-1] - 0.15)))
        s.append(s_next)
        w.append(w_next)

    r2 = lotka_volterra_fit(s, w)
    assert r2 > 0.95


def test_population_cycle_metrics_detects_period_and_lag() -> None:
    steps = 120
    period_true = 20.0
    boards: list[BoardState] = []
    for t in range(steps):
        n_sheep = int(round(30 + 10 * math.sin(2.0 * math.pi * t / period_true)))
        n_wolves = int(round(12 + 4 * math.sin(2.0 * math.pi * (t - 4) / period_true)))
        animals = [AnimalState(i, "sheep", i % 10, 0, 10) for i in range(n_sheep)] + [
            AnimalState(1000 + i, "wolf", i % 10, 5, 20) for i in range(n_wolves)
        ]
        boards.append(_board(t, animals))

    period, purity, lag = population_cycle_metrics(boards, burn_in_frac=0.0)
    assert period == pytest.approx(period_true, abs=0.5)
    assert purity > 0.90
    assert lag == 4


def test_pack_hunting_rate_solo_vs_pack_and_torus_wrap() -> None:
    b1 = _board(
        1,
        [
            AnimalState(10, "wolf", 0, 0, 25),
            AnimalState(11, "wolf", 9, 9, 20),
            AnimalState(12, "wolf", 5, 5, 20),
        ],
    )
    b2 = _board(
        2,
        [
            AnimalState(10, "wolf", 0, 0, 24),
            AnimalState(11, "wolf", 9, 9, 19),
            AnimalState(12, "wolf", 5, 5, 25),
        ],
    )
    events = [
        EventRecord(step=1, kind="death", agent_id=1, species="sheep", cause="eaten", other_id=10),
        EventRecord(step=2, kind="death", agent_id=2, species="sheep", cause="eaten", other_id=12),
    ]

    rate, mean_pack = pack_hunting_rate([b1, b2], events, radius=1, pack_size=2)
    assert rate == pytest.approx(0.5)
    assert mean_pack == pytest.approx(1.5)
    no_kill_rate, _ = pack_hunting_rate([b1, b2], [])
    assert math.isnan(no_kill_rate)


def test_herd_cohesion_aligned_vs_opposing_motion() -> None:
    b0 = _board(0, [AnimalState(1, "sheep", 2, 3, 10), AnimalState(2, "sheep", 9, 3, 10)])
    b1 = _board(1, [AnimalState(1, "sheep", 3, 3, 10), AnimalState(2, "sheep", 0, 3, 10)])
    order_aligned, nnd = herd_cohesion([b0, b1])
    assert order_aligned == pytest.approx(1.0)
    assert nnd > 0.0

    b2 = _board(2, [AnimalState(1, "sheep", 4, 3, 10), AnimalState(2, "sheep", 9, 3, 10)])
    order_opposing, _ = herd_cohesion([b1, b2])
    assert order_opposing == pytest.approx(0.0)


def test_alarm_call_survival_effect() -> None:
    b0 = _board(
        0,
        [
            AnimalState(1, "sheep", 5, 7, 10),
            AnimalState(2, "sheep", 5, 3, 10),
            AnimalState(3, "sheep", 5, 9, 10),
            AnimalState(9, "wolf", 5, 5, 20),
        ],
    )
    b1 = _board(
        1,
        [
            AnimalState(1, "sheep", 5, 8, 10),
            AnimalState(3, "sheep", 5, 9, 10),
            AnimalState(9, "wolf", 5, 3, 25),
        ],
    )
    b2 = _board(
        2,
        [
            AnimalState(1, "sheep", 5, 8, 10),
            AnimalState(3, "sheep", 5, 9, 10),
            AnimalState(9, "wolf", 5, 2, 24),
        ],
    )
    events = [EventRecord(step=0, kind="call", agent_id=3, species="sheep")]

    lift = alarm_call_survival_effect(
        [b0, b1, b2],
        events,
        threat_radius=3,
        call_radius=2,
        lookback_window=1,
        horizon=2,
    )
    assert lift == pytest.approx(1.0)


def test_evaluate_run_dir_round_trip(tmp_path: Path) -> None:
    b0 = _board(
        0,
        [
            AnimalState(1, "sheep", 2, 2, 10),
            AnimalState(2, "sheep", 3, 2, 10),
            AnimalState(9, "wolf", 8, 8, 20),
        ],
    )
    b1 = _board(
        1,
        [
            AnimalState(1, "sheep", 2, 3, 10),
            AnimalState(2, "sheep", 3, 3, 10),
            AnimalState(9, "wolf", 8, 7, 19),
        ],
    )
    (tmp_path / "ledger.jsonl").write_text(
        "\n".join([b0.to_json(), b1.to_json()]) + "\n", encoding="utf-8"
    )
    ev = EventRecord(step=1, kind="birth", agent_id=3, species="sheep", other_id=1)
    (tmp_path / "events.jsonl").write_text(json.dumps(ev.__dict__) + "\n", encoding="utf-8")

    summary = evaluate_run_dir(tmp_path, burn_in_frac=0.0)
    assert summary.population.sheep == (2, 2)
    assert not summary.population.extinct
    assert summary.herd_cohesion == pytest.approx(1.0)
    assert math.isnan(summary.pack_hunting_rate)
