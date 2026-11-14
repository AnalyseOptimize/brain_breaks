"""Independent exactness and integration checks; run with unittest discovery."""

import itertools
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rectangle_plots import plot_comparison, plot_rectangles
from src.rectangle_segmentation import (
    RectangleCost,
    segment_genetic, segment_pelt,
)


def direct_objective(y, points, penalty):
    """Oracle without normalization tables or production cost functions."""
    span = np.ptp(y)
    area = 0.0
    start = 0
    for end in points:
        area += (end - start) * np.ptp(y[start:end])
        start = end
    return (area / (len(y) * span) if span else 0.0) + penalty * (len(points) - 1)


def unpruned_dp(y, penalty, min_size):
    """Quadratic Bellman recurrence, direct extrema, no pruning."""
    n = len(y)
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    span = np.ptp(y)
    for end in range(min_size, n + 1):
        for start in range(end - min_size + 1):
            area = ((end - start) * np.ptp(y[start:end]) / (n * span)) if span else 0
            best[end] = min(best[end], best[start] + area + penalty)
    return best[-1]


class RectangleSegmentationTests(unittest.TestCase):
    def test_extrema_queries_match_direct_cost(self):
        y = np.random.default_rng(42).normal(size=41)
        cost = RectangleCost(y)
        for start in range(len(y)):
            for end in range(start + 1, len(y) + 1):
                expected = (end - start) * np.ptp(y[start:end]) / (len(y) * np.ptp(y))
                self.assertAlmostEqual(cost.error(start, end), expected)

    def test_pelt_matches_every_partition(self):
        rng = np.random.default_rng(81)
        for n in range(1, 10):
            y = rng.normal(size=n)
            for min_size in range(1, n + 1):
                for penalty in [0, 0.03, 0.3, 2]:
                    candidates = []
                    for flags in itertools.product([False, True], repeat=n - 1):
                        points = [i + 1 for i, flag in enumerate(flags) if flag] + [n]
                        if min(np.diff([0, *points])) >= min_size:
                            candidates.append(direct_objective(y, points, penalty))
                    result = segment_pelt(y, penalty, min_size)
                    self.assertAlmostEqual(result.objective, min(candidates), places=12)

    def test_delayed_pruning_matches_unpruned_dp(self):
        rng = np.random.default_rng(17)
        for _ in range(100):
            n = int(rng.integers(8, 65))
            min_size = int(rng.integers(2, n + 1))
            penalty = float(rng.choice([0, 0.001, 0.02, 0.2, 1]))
            y = rng.integers(-4, 5, size=n).astype(float)
            result = segment_pelt(y, penalty, min_size)
            self.assertAlmostEqual(result.objective, unpruned_dp(y, penalty, min_size), places=12)

    def test_known_breaks_and_scale_invariance(self):
        y = np.repeat([0.0, 5.0, -2.0], [15, 20, 17])
        expected = [15, 35, 52]
        for transformed in [y, y * 1e-13, y * -3 + 12]:
            result = segment_pelt(transformed, 0.03, 4)
            self.assertEqual(result.breakpoints, expected)
            self.assertAlmostEqual(result.objective, 0.06)

    def test_constant_and_single_sample(self):
        for n in [1, 20]:
            for penalty in [0, 0.03]:
                for fn in [segment_pelt, segment_genetic]:
                    kwargs = {} if fn == segment_pelt else dict(n_restarts=2, max_iterations=10)
                    result = fn(np.full(n, 1e-13), penalty, 1, **kwargs)
                    self.assertEqual(result.breakpoints, [n])
                    self.assertEqual(result.objective, 0)

    def test_ga_reproducibility_constraints_and_exact_bound(self):
        y = np.random.default_rng(4).normal(size=80)
        kwargs = dict(penalty=0.03, min_size=7, n_restarts=3, max_iterations=250, patience=100, seed=55)
        a, b = segment_genetic(y, **kwargs), segment_genetic(y, **kwargs)
        self.assertEqual(a.breakpoints, b.breakpoints)
        np.testing.assert_array_equal(a.history, b.history)
        self.assertTrue(np.all(np.diff(a.history) <= 0))
        self.assertTrue(np.all(np.diff([0, *a.breakpoints]) >= 7))
        self.assertAlmostEqual(a.objective, direct_objective(y, a.breakpoints, 0.03))
        self.assertGreaterEqual(a.objective + 1e-12, segment_pelt(y, 0.03, 7).objective)
        for r in a.rectangles:
            self.assertTrue(np.all(y[r.start:r.end] >= r.lower))
            self.assertTrue(np.all(y[r.start:r.end] <= r.upper))
        self.assertEqual(len(a.diagnostics['restart_objectives']), 3)
        self.assertEqual(len(a.diagnostics['iterations']), 4)  # champions phase

    def test_validation(self):
        for signal in [[], [[1, 2]], [1, np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                segment_pelt(signal, min_size=1)
        for kwargs in [dict(penalty=-1), dict(penalty=np.nan), dict(min_size=0),
                       dict(min_size=5), dict(min_size=1.5)]:
            with self.assertRaises(ValueError):
                segment_pelt([1, 2, 3], **kwargs)
        with self.assertRaises(ValueError):
            segment_genetic([1, 2, 3], min_size=1, p_uniform=0.8, p_one_point=0.8)
        cost = RectangleCost([1, 2, 3])
        for points in [[2], [0, 3], [3, 2, 3], [1.5, 3]]:
            with self.assertRaises(ValueError):
                cost.evaluate(points)

    def test_plot_boundaries_and_last_sample_cell(self):
        y = np.repeat([0.0, 3.0], 10)
        times = np.arange(20) / 1000
        series = dict(signal=y, times=times, label="test")
        result = segment_pelt(y, 0.02, 2)
        fig, axes = plot_comparison(series, [result], true_breakpoints=[10])
        # Signal plus one detected and one true internal break; no line at n.
        self.assertEqual(len(axes[0, 0].lines), 3)
        self.assertEqual(len(axes[0, 1].patches), 2)
        last = axes[0, 1].patches[-1]
        self.assertAlmostEqual(last.get_x() + last.get_width(), 0.020)
        plt.close(fig)
        fig, _ = plot_rectangles(dict(signal=[1.0], times=[0.0]), segment_pelt([1], min_size=1))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
