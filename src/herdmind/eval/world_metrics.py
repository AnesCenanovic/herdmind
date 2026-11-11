"""World-level and emergence metrics computed from run ledgers and event logs.

Implements the metrics defined in ``docs/research/emergence-metrics.md``
(backlog item 16): population series, extinction tracking, Lotka-Volterra fit
diagnostics, pack-hunting rate, herd cohesion (Vicsek order parameter + NND),
alarm-call survival effect, and dominant population cycle period.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from herdmind.world.ledger import BoardState, read_ledger
from herdmind.world.log import EventRecord
from herdmind.world.perception import wrap_delta


@dataclass(frozen=True)
class PopulationSummary:
    """Summary of sheep and wolf population trajectories across a run."""

    sheep: tuple[int, ...]
    wolves: tuple[int, ...]
    extinct: bool
    extinction_step: int | None
    extinct_species: str | None
    cycle_period: float
    spectral_purity: float
    predator_prey_lag: int | None
    lv_r2: float


@dataclass(frozen=True)
class EmergenceSummary:
    """Emergent behavioural metrics for a completed simulation run."""

    pack_hunting_rate: float
    mean_pack_size: float
    herd_cohesion: float
    normalized_nnd: float
    alarm_call_survival_effect: float
    population: PopulationSummary


def read_events(run_dir: str | Path) -> list[EventRecord]:
    """Read ``events.jsonl`` from a run directory into typed ``EventRecord`` items."""
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EventRecord(**json.loads(line)) for line in lines if line.strip()]


def population_series(boards: Sequence[BoardState]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(sheep_counts, wolf_counts)`` aligned with ``boards``."""
    if not boards:
        raise ValueError("boards must not be empty")
    sheep = np.array([b.count("sheep") for b in boards], dtype=np.int64)
    wolves = np.array([b.count("wolf") for b in boards], dtype=np.int64)
    return sheep, wolves


def extinction_info(boards: Sequence[BoardState]) -> tuple[bool, int | None, str | None]:
    """Return ``(extinct, first_extinction_step, extinct_species)``."""
    for b in boards:
        s_cnt = b.count("sheep")
        w_cnt = b.count("wolf")
        if s_cnt == 0 and w_cnt == 0:
            return True, b.step, "both"
        if s_cnt == 0:
            return True, b.step, "sheep"
        if w_cnt == 0:
            return True, b.step, "wolf"
    return False, None, None


def lotka_volterra_fit(
    sheep: NDArray[np.int64] | Sequence[int],
    wolves: NDArray[np.int64] | Sequence[int],
) -> float:
    """Return mean R^2 of discrete Lotka-Volterra one-step growth regression."""
    s = np.asarray(sheep, dtype=np.float64)
    w = np.asarray(wolves, dtype=np.float64)
    if s.shape != w.shape or s.ndim != 1 or s.size < 4:
        return float("nan")

    valid = (s[:-1] > 0.0) & (w[:-1] > 0.0)
    if int(np.sum(valid)) < 3:
        return float("nan")

    s0, w0 = s[:-1][valid], w[:-1][valid]
    s_rate = (s[1:][valid] - s0) / s0
    w_rate = (w[1:][valid] - w0) / w0

    def _r2(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 1e-12:
            return float("nan")
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        return 1.0 - (ss_res / ss_tot)

    r2_s = _r2(w0, s_rate)
    r2_w = _r2(s0, w_rate)
    if math.isnan(r2_s) or math.isnan(r2_w):
        return float("nan")
    return 0.5 * (r2_s + r2_w)


def population_cycle_metrics(
    boards: Sequence[BoardState],
    *,
    burn_in_frac: float = 0.1,
    min_purity: float = 0.15,
) -> tuple[float, float, int | None]:
    """Return ``(cycle_period, spectral_purity, predator_prey_lag)`` after burn-in."""
    sheep, wolves = population_series(boards)
    t_total = sheep.size
    t0 = int(math.floor(burn_in_frac * t_total))
    s = sheep[t0:].astype(np.float64)
    w = wolves[t0:].astype(np.float64)
    m = s.size
    if m < 4:
        return float("nan"), float("nan"), None

    s_centered = s - np.mean(s)
    if float(np.std(s_centered)) <= 1e-12:
        return float("nan"), 0.0, None

    fft_vals = np.fft.rfft(s_centered)
    power = (np.abs(fft_vals[1:]) ** 2) / m
    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        return float("nan"), 0.0, None

    k_star = int(np.argmax(power)) + 1
    purity = float(power[k_star - 1] / total_power)
    period = float(m / k_star) if purity >= min_purity else float("nan")

    w_centered = w - np.mean(w)
    lag: int | None = None
    if float(np.std(w_centered)) > 1e-12:
        max_lag = min(m // 2, int(round(period)) if math.isfinite(period) else m // 2)
        best_corr = -float("inf")
        for l_step in range(1, max(2, max_lag + 1)):
            if l_step >= m:
                break
            s_seg = s_centered[: m - l_step]
            w_seg = w_centered[l_step:]
            denom = float(np.linalg.norm(s_seg) * np.linalg.norm(w_seg))
            if denom > 1e-12:
                corr = float(np.dot(s_seg, w_seg) / denom)
                if corr > best_corr:
                    best_corr = corr
                    lag = l_step
    return period, purity, lag


def pack_hunting_rate(
    boards: Sequence[BoardState],
    events: Sequence[EventRecord],
    *,
    radius: int = 1,
    pack_size: int = 2,
    burn_in_steps: int = 0,
) -> tuple[float, float]:
    """Return ``(pack_hunting_rate, mean_wolves_at_kill)`` from predation events."""
    if not boards:
        raise ValueError("boards must not be empty")
    first_board = boards[0]
    height = len(first_board.grass)
    width = len(first_board.grass[0]) if height > 0 else 0
    board_by_step = {b.step: b for b in boards}

    kills = [
        e for e in events if e.kind == "death" and e.cause == "eaten" and e.step >= burn_in_steps
    ]
    if not kills:
        return float("nan"), float("nan")

    wolf_counts: list[int] = []
    for ev in kills:
        board = board_by_step.get(ev.step)
        if board is None:
            continue
        wolves = [a for a in board.animals if a.species == "wolf"]
        killer = next((w for w in wolves if w.agent_id == ev.other_id), None)
        if killer is None:
            prev = board_by_step.get(ev.step - 1)
            prey = (
                next((a for a in prev.animals if a.agent_id == ev.agent_id), None)
                if prev is not None
                else None
            )
            if prey is None:
                continue
            kx, ky = prey.x, prey.y
        else:
            kx, ky = killer.x, killer.y

        near = sum(
            1
            for w in wolves
            if max(abs(wrap_delta(kx, w.x, width)), abs(wrap_delta(ky, w.y, height))) <= radius
        )
        wolf_counts.append(max(1, near))

    if not wolf_counts:
        return float("nan"), float("nan")
    arr = np.asarray(wolf_counts, dtype=np.float64)
    rate = float(np.mean(arr >= pack_size))
    return rate, float(np.mean(arr))


def herd_cohesion(
    boards: Sequence[BoardState],
    *,
    burn_in_steps: int = 0,
) -> tuple[float, float]:
    """Return ``(mean_vicsek_order, mean_normalized_nnd)`` for sheep after burn-in."""
    if len(boards) < 2:
        return float("nan"), float("nan")

    first_board = boards[0]
    height = len(first_board.grass)
    width = len(first_board.grass[0]) if height > 0 else 0
    area = float(width * height)

    orders: list[float] = []
    nnds: list[float] = []

    for prev, curr in zip(boards[:-1], boards[1:], strict=True):
        if curr.step < burn_in_steps:
            continue
        prev_sheep = {a.agent_id: a for a in prev.animals if a.species == "sheep"}
        curr_sheep = [a for a in curr.animals if a.species == "sheep"]
        if len(curr_sheep) < 2:
            continue

        headings: list[tuple[float, float]] = []
        for s in curr_sheep:
            p = prev_sheep.get(s.agent_id)
            if p is None:
                continue
            dx = float(wrap_delta(p.x, s.x, width))
            dy = float(wrap_delta(p.y, s.y, height))
            norm = math.hypot(dx, dy)
            if norm > 0.0:
                headings.append((dx / norm, dy / norm))
            else:
                headings.append((0.0, 0.0))

        if len(headings) >= 2:
            sum_x = sum(hx for hx, _ in headings)
            sum_y = sum(hy for _, hy in headings)
            orders.append(math.hypot(sum_x, sum_y) / len(headings))

        n_s = len(curr_sheep)
        expected_random = 0.5 * math.sqrt(area / n_s)
        if expected_random > 0.0:
            min_dists: list[float] = []
            for i, a in enumerate(curr_sheep):
                best = float("inf")
                for j, b in enumerate(curr_sheep):
                    if i == j:
                        continue
                    dx = float(wrap_delta(a.x, b.x, width))
                    dy = float(wrap_delta(a.y, b.y, height))
                    d = math.hypot(dx, dy)
                    if d < best:
                        best = d
                min_dists.append(best / expected_random)
            nnds.append(float(np.mean(min_dists)))

    mean_order = float(np.mean(orders)) if orders else float("nan")
    mean_nnd = float(np.mean(nnds)) if nnds else float("nan")
    return mean_order, mean_nnd


def alarm_call_survival_effect(
    boards: Sequence[BoardState],
    events: Sequence[EventRecord],
    *,
    threat_radius: int = 4,
    call_radius: int = 5,
    lookback_window: int = 3,
    horizon: int = 5,
    burn_in_steps: int = 0,
) -> float:
    """Return survival lift ``Delta S_call`` for threatened sheep hearing an alarm call."""
    if len(boards) <= horizon:
        return float("nan")

    first_board = boards[0]
    height = len(first_board.grass)
    width = len(first_board.grass[0]) if height > 0 else 0
    board_by_step = {b.step: b for b in boards}
    max_step = max(board_by_step)

    calls_by_step: dict[int, list[tuple[int, int, int]]] = {}
    for ev in events:
        if ev.kind != "call":
            continue
        b = board_by_step.get(ev.step)
        if b is None:
            continue
        caller = next((a for a in b.animals if a.agent_id == ev.agent_id), None)
        if caller is not None:
            calls_by_step.setdefault(ev.step, []).append((caller.agent_id, caller.x, caller.y))

    if not calls_by_step:
        return float("nan")

    exposed_survived: list[float] = []
    unexposed_survived: list[float] = []

    for b in boards:
        if b.step < burn_in_steps or b.step + horizon > max_step:
            continue
        future_board = board_by_step.get(b.step + horizon)
        if future_board is None:
            continue
        future_alive = {a.agent_id for a in future_board.animals if a.species == "sheep"}

        wolves = [a for a in b.animals if a.species == "wolf"]
        sheep = [a for a in b.animals if a.species == "sheep"]
        if not wolves or not sheep:
            continue

        recent_calls: list[tuple[int, int, int]] = []
        for t_prev in range(max(0, b.step - lookback_window), b.step + 1):
            recent_calls.extend(calls_by_step.get(t_prev, ()))

        for s in sheep:
            threatened = any(
                abs(wrap_delta(s.x, w.x, width)) + abs(wrap_delta(s.y, w.y, height))
                <= threat_radius
                for w in wolves
            )
            if not threatened:
                continue

            heard = any(
                cid != s.agent_id
                and abs(wrap_delta(s.x, cx, width)) + abs(wrap_delta(s.y, cy, height))
                <= call_radius
                for cid, cx, cy in recent_calls
            )
            survived = 1.0 if s.agent_id in future_alive else 0.0
            if heard:
                exposed_survived.append(survived)
            else:
                unexposed_survived.append(survived)

    if not exposed_survived or not unexposed_survived:
        return float("nan")
    return float(np.mean(exposed_survived) - np.mean(unexposed_survived))


def summarize_run(
    boards: Sequence[BoardState],
    events: Sequence[EventRecord] = (),
    *,
    pack_radius: int = 1,
    pack_size: int = 2,
    burn_in_frac: float = 0.1,
) -> EmergenceSummary:
    """Compute all world and emergence metrics for a trajectory."""
    sheep, wolves = population_series(boards)
    extinct, ext_step, ext_species = extinction_info(boards)
    period, purity, lag = population_cycle_metrics(boards, burn_in_frac=burn_in_frac)
    lv_r2 = lotka_volterra_fit(sheep, wolves)

    burn_in_steps = int(math.floor(burn_in_frac * len(boards)))
    pack_rate, mean_pack = pack_hunting_rate(
        boards,
        events,
        radius=pack_radius,
        pack_size=pack_size,
        burn_in_steps=burn_in_steps,
    )
    cohesion, nnd = herd_cohesion(boards, burn_in_steps=burn_in_steps)
    call_effect = alarm_call_survival_effect(boards, events, burn_in_steps=burn_in_steps)

    pop_summary = PopulationSummary(
        sheep=tuple(int(x) for x in sheep),
        wolves=tuple(int(x) for x in wolves),
        extinct=extinct,
        extinction_step=ext_step,
        extinct_species=ext_species,
        cycle_period=period,
        spectral_purity=purity,
        predator_prey_lag=lag,
        lv_r2=lv_r2,
    )
    return EmergenceSummary(
        pack_hunting_rate=pack_rate,
        mean_pack_size=mean_pack,
        herd_cohesion=cohesion,
        normalized_nnd=nnd,
        alarm_call_survival_effect=call_effect,
        population=pop_summary,
    )


def evaluate_run_dir(
    run_dir: str | Path,
    *,
    pack_radius: int = 1,
    pack_size: int | None = None,
    burn_in_frac: float = 0.1,
) -> EmergenceSummary:
    """Load ``ledger.jsonl`` and ``events.jsonl`` from ``run_dir`` and compute metrics."""
    run_path = Path(run_dir)
    if pack_size is None:
        spec_path = run_path / "spec.json"
        if spec_path.exists():
            spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
            pack_size = int(spec_data.get("laws", {}).get("pack_size", 2))
        else:
            pack_size = 2
    boards = read_ledger(run_path)
    events = read_events(run_path)
    return summarize_run(
        boards,
        events,
        pack_radius=pack_radius,
        pack_size=pack_size,
        burn_in_frac=burn_in_frac,
    )
