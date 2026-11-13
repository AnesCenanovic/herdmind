"""Tests for decision and calibration metrics against hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from herdmind.common.seeding import make_rng
from herdmind.eval.metrics import (
    agreement,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    kl_divergence,
    negative_log_likelihood,
)


def test_agreement_hand_computed() -> None:
    teacher = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1],
    ]
    student = [
        [0.5, 0.4, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.2, 0.7],
        [0.4, 0.2, 0.4],
    ]
    assert agreement(teacher, student) == pytest.approx(0.75)


def test_kl_divergence_hand_computed() -> None:
    teacher = [[0.8, 0.2], [0.5, 0.5]]
    student = [[0.4, 0.6], [0.5, 0.5]]
    expected = 0.5 * (0.8 * math.log(2.0) + 0.2 * math.log(1.0 / 3.0))
    assert kl_divergence(teacher, student) == pytest.approx(expected)


def test_kl_divergence_clips_zero_student_prob() -> None:
    teacher = [[1.0, 0.0]]
    student = [[0.0, 1.0]]
    eps = 1e-12
    expected = 1.0 * math.log(1.0 / eps) + eps * math.log(eps / 1.0)
    assert kl_divergence(teacher, student, eps=eps) == pytest.approx(expected)


def test_expected_calibration_error_hand_computed() -> None:
    probs = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.6, 0.4],
    ]
    labels = [0, 1, 0]
    expected = (2.0 / 3.0) * 0.35 + (1.0 / 3.0) * 0.40
    assert expected_calibration_error(probs, labels, num_bins=3) == pytest.approx(expected)


def test_brier_score_hand_computed() -> None:
    probs = [
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
    ]
    labels = [0, 2]
    assert brier_score(probs, labels) == pytest.approx(0.20)


def test_negative_log_likelihood_hand_computed() -> None:
    probs = [
        [0.8, 0.2],
        [0.25, 0.75],
    ]
    labels = [0, 1]
    expected = 0.5 * (-math.log(0.8) - math.log(0.75))
    assert negative_log_likelihood(probs, labels) == pytest.approx(expected)


def test_bootstrap_ci_reproducible_and_bounded() -> None:
    values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ci_a = bootstrap_ci(values, make_rng(42), num_resamples=500, confidence=0.90)
    ci_b = bootstrap_ci(values, make_rng(42), num_resamples=500, confidence=0.90)
    assert ci_a == ci_b
    assert 0.2 <= ci_a[0] <= float(np.mean(values)) <= ci_a[1] <= 1.0


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        agreement([[0.5, 0.6]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="shape mismatch"):
        kl_divergence([[0.5, 0.5]], [[0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="labels must be in"):
        brier_score([[0.5, 0.5]], [2])
