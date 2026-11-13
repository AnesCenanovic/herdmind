"""Decision and calibration metrics comparing student and teacher distributions.

All metrics live here; scripts and notebooks import them rather than
reimplementing formulas. Follows the definitions in ``.claude/CLAUDE.md``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

EPS = 1e-12
DEFAULT_ECE_BINS = 15


def _as_prob_matrix(probs: ArrayLike, name: str = "probs") -> NDArray[np.float64]:
    """Validate and convert input probabilities to a 2D float64 array ``(N, K)``."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"each row in {name} must sum to 1, got sums {row_sums}")
    return arr


def _as_label_indices(labels: ArrayLike, num_samples: int, num_classes: int) -> NDArray[np.intp]:
    """Validate integer class indices ``(N,)`` in ``[0, K)``."""
    idx = np.asarray(labels, dtype=np.intp)
    if idx.ndim != 1 or idx.shape[0] != num_samples:
        raise ValueError(f"labels must have shape ({num_samples},), got {idx.shape}")
    if np.any(idx < 0) or np.any(idx >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes}), got {idx}")
    return idx


def agreement(teacher_probs: ArrayLike, student_probs: ArrayLike) -> float:
    """Return fraction of decisions where ``argmax(student) == argmax(teacher)``.

    Args:
        teacher_probs: Reference probability distributions of shape ``(N, K)``.
        student_probs: Student probability distributions of shape ``(N, K)``.

    Returns:
        Fraction in ``[0.0, 1.0]`` of matching top-1 predictions.
    """
    p_teacher = _as_prob_matrix(teacher_probs, "teacher_probs")
    p_student = _as_prob_matrix(student_probs, "student_probs")
    if p_teacher.shape != p_student.shape:
        raise ValueError(f"shape mismatch: teacher {p_teacher.shape} vs student {p_student.shape}")
    matches = np.argmax(p_teacher, axis=1) == np.argmax(p_student, axis=1)
    return float(np.mean(matches))


def kl_divergence(teacher_probs: ArrayLike, student_probs: ArrayLike, *, eps: float = EPS) -> float:
    """Return mean ``KL(teacher || student)`` using natural log and ``eps`` clipping.

    Args:
        teacher_probs: Reference probability distributions of shape ``(N, K)``.
        student_probs: Student probability distributions of shape ``(N, K)``.
        eps: Lower clipping bound to avoid ``log(0)``.

    Returns:
        Mean KL divergence across the ``N`` decisions.
    """
    p = np.clip(_as_prob_matrix(teacher_probs, "teacher_probs"), eps, 1.0)
    q = np.clip(_as_prob_matrix(student_probs, "student_probs"), eps, 1.0)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: teacher {p.shape} vs student {q.shape}")
    per_decision = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    return float(np.mean(per_decision))


def expected_calibration_error(
    probs: ArrayLike,
    labels: ArrayLike,
    *,
    num_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Return top-label Expected Calibration Error (ECE) over equal-width bins.

    Args:
        probs: Predicted probability distributions of shape ``(N, K)``.
        labels: Ground-truth integer class indices of shape ``(N,)`` (e.g.
            teacher argmax, sampled action, or outcome).
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        Sample-weighted average absolute gap between bin accuracy and confidence.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    p = _as_prob_matrix(probs, "probs")
    y = _as_label_indices(labels, p.shape[0], p.shape[1])

    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = p.shape[0]
    ece = 0.0
    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(correct[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    """Return multi-class Brier score: mean over decisions of ``sum_k (p_k - y_k)^2``.

    Args:
        probs: Predicted probability distributions of shape ``(N, K)``.
        labels: Ground-truth integer class indices of shape ``(N,)``.

    Returns:
        Mean squared probability error in ``[0.0, 2.0]``.
    """
    p = _as_prob_matrix(probs, "probs")
    y = _as_label_indices(labels, p.shape[0], p.shape[1])
    one_hot = np.zeros_like(p)
    one_hot[np.arange(p.shape[0]), y] = 1.0
    per_decision = np.sum((p - one_hot) ** 2, axis=1)
    return float(np.mean(per_decision))


def negative_log_likelihood(probs: ArrayLike, labels: ArrayLike, *, eps: float = EPS) -> float:
    """Return mean negative log-likelihood ``-log p(label)`` with ``eps`` clipping.

    Args:
        probs: Predicted probability distributions of shape ``(N, K)``.
        labels: Ground-truth integer class indices of shape ``(N,)`` (e.g.
            teacher argmax, sampled action, or outcome).
        eps: Lower clipping bound to avoid ``log(0)``.

    Returns:
        Mean NLL across decisions.
    """
    p = _as_prob_matrix(probs, "probs")
    y = _as_label_indices(labels, p.shape[0], p.shape[1])
    chosen_probs = np.clip(p[np.arange(p.shape[0]), y], eps, 1.0)
    return float(np.mean(-np.log(chosen_probs)))


def bootstrap_ci(
    values: ArrayLike,
    rng: np.random.Generator,
    *,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    num_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return percentile bootstrap confidence interval ``(low, high)``.

    Args:
        values: 1D sequence of per-decision or per-seed metric values.
        rng: Seeded ``numpy.random.Generator`` from ``common.seeding.make_rng``.
        statistic: Summary function applied to each bootstrap resample.
        num_resamples: Number of bootstrap resamples.
        confidence: Confidence level in ``(0, 1)``, e.g. ``0.95`` for 95% CI.

    Returns:
        Tuple ``(lower_bound, upper_bound)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError(f"values must be a non-empty 1D array, got shape {arr.shape}")
    if num_resamples <= 0:
        raise ValueError(f"num_resamples must be positive, got {num_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n = arr.shape[0]
    indices = rng.integers(0, n, size=(num_resamples, n))
    samples = arr[indices]
    stats = np.array([statistic(row) for row in samples], dtype=np.float64)

    alpha = 1.0 - confidence
    low = float(np.quantile(stats, alpha / 2.0))
    high = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return low, high
