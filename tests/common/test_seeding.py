from __future__ import annotations

from herdmind.common.seeding import make_rng


def test_same_seed_same_stream() -> None:
    assert make_rng(7).random(5).tolist() == make_rng(7).random(5).tolist()


def test_different_seed_different_stream() -> None:
    assert make_rng(7).random() != make_rng(8).random()
