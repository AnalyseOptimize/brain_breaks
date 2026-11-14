"""Shared validation, result type, and length-aware PELT for break detection.

Every breakpoint list ends with n. Internal b is the first index of the
right segment. No routines download data or change input arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BreakResult:
    method: str
    breakpoints: list[int]
    fit: np.ndarray
    objective: float
    penalty: float
    elapsed_seconds: float
    params: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def change_points(self):
        return self.breakpoints[:-1]


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def nonnegative(value, name):
    try:
        valid = np.isscalar(value) and np.isfinite(value) and value >= 0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def prepare_signal(signal, min_size=3, jump=1, times=None):
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("signal must be a finite 1-D array with >=3 samples")
    min_size = integer(min_size, "min_size", 2)
    jump = integer(jump, "jump")
    if min_size > len(y):
        raise ValueError("min_size cannot exceed signal length")
    if times is not None:
        t = np.asarray(times, dtype=float)
        if t.shape != y.shape or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("times must be finite, increasing, and match signal")
        if not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-5, atol=1e-12):
            raise ValueError("These implementations require uniform sampling")
    magnitude = float(np.max(np.abs(y))) or 1.0
    scaled = y / magnitude
    center = float(scaled.mean())
    spread = float(scaled.std())
    offset = center * magnitude
    scale = spread * magnitude if spread else magnitude
    z = (scaled - center) / spread if spread else np.zeros_like(y)
    return y, z, offset, scale, min_size, jump


def endpoint_grid(n, jump):
    return np.r_[np.arange(0, n, jump, dtype=int), n]


def pelt_search(cost, n, penalty, min_size, jump=1):
    """PELT with K=0; cost.error_many(starts, end) returns segment costs.

    Cost must satisfy C(a,b)+C(b,c)<=C(a,c) for admissible segments. A
    dominated start only expires when its replacement has min_size samples.
    With jump>1 this is exact over the explicitly restricted endpoint grid.
    """
    points = endpoint_grid(n, jump)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.full(n + 1, -1, dtype=int)
    counts = np.zeros(n + 1, dtype=int)
    expires = np.full(n + 1, n + min_size + 1, dtype=int)
    active = np.empty(0, dtype=int)
    cursor, evaluations, max_candidates = 0, 0, 0
    for end in points[1:]:
        active = active[expires[active] > end]
        while cursor < len(points) and points[cursor] <= end - min_size:
            start = points[cursor]
            if np.isfinite(best[start]):
                active = np.append(active, start)
            cursor += 1
        if not len(active):
            continue
        unpenalized = best[active] + cost.error_many(active, int(end))
        values = unpenalized + penalty
        winner = np.lexsort((active, counts[active], values))[0]
        start = active[winner]
        best[end], previous[end], counts[end] = values[winner], start, counts[start] + 1
        tolerance = 1e-9 * max(1.0, abs(best[end]))
        dominated = active[unpenalized > best[end] + tolerance]
        expires[dominated] = np.minimum(expires[dominated], end + min_size)
        evaluations += len(active)
        max_candidates = max(max_candidates, len(active))
    if not np.isfinite(best[n]):
        raise ValueError("No feasible segmentation for the chosen grid/min_size")
    result, end = [], n
    while end:
        result.append(int(end))
        end = previous[end]
    return result[::-1], float(best[n]), dict(
        cost_evaluations=evaluations, max_candidates=max_candidates,
        exact_on_grid=True, full_sample_grid=jump == 1, jump=jump,
    )


def match_breakpoints(predicted, truth, tolerance):
    """Maximum one-to-one matches, then minimum total absolute timing error.

    Inputs are internal indices or times in the SAME units. Empty/empty
    receives F1=1; no-match localization error is NaN, not zero.
    """
    predicted, truth = np.asarray(sorted(predicted)), np.asarray(sorted(truth))
    tolerance = nonnegative(tolerance, "tolerance")
    p, q = len(predicted), len(truth)
    counts = np.zeros((p + 1, q + 1), dtype=int)
    errors = np.zeros((p + 1, q + 1))
    actions = np.zeros((p + 1, q + 1), dtype=np.int8)
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            options = [(counts[i-1, j], -errors[i-1, j], 1),
                       (counts[i, j-1], -errors[i, j-1], 2)]
            distance = abs(predicted[i-1] - truth[j-1])
            roundoff = 8*np.finfo(float).eps*max(1., abs(predicted[i-1]), abs(truth[j-1]))
            if distance <= tolerance+roundoff:
                options.append((counts[i-1, j-1] + 1, -errors[i-1, j-1] - distance, 3))
            count, negative_error, action = max(options)
            counts[i, j], errors[i, j], actions[i, j] = count, -negative_error, action
    matches, i, j = [], p, q
    while i and j:
        action = actions[i, j]
        if action == 3:
            matches.append((float(predicted[i-1]), float(truth[j-1])))
            i, j = i-1, j-1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    tp = int(counts[p, q])
    precision = tp / p if p else (1.0 if not q else 0.0)
    recall = tp / q if q else (1.0 if not p else 0.0)
    f1 = 2 * tp / (p + q) if p + q else 1.0
    return dict(tp=tp, fp=p-tp, fn=q-tp, precision=precision, recall=recall,
                f1=f1, timing_mae=errors[p, q] / tp if tp else np.nan, matches=matches[::-1])
