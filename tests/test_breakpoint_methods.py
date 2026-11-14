"""Independent optimality checks for CPOP, polynomial/RBF PELT and L1 TF."""
from itertools import combinations, product
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.breakpoint_common import match_breakpoints
from src.cpop_segmentation import segment_cpop, _Quadratic, _lower_envelope
from src.pelt_regression import segment_regression, PolynomialCost
from src.pelt_rbf import segment_rbf, RbfCost
from src.trend_filtering import segment_trend_filter


def partitions(n, min_size, jump=1, continuous=False):
    limit = n-1 if continuous else n
    allowed = [p for p in range(jump, limit, jump) if min_size <= p <= n-min_size]
    for flags in product([False, True], repeat=len(allowed)):
        points = [p for p, flag in zip(allowed, flags) if flag]
        if min(np.diff([0, *points, n])) >= min_size:
            yield points


class BreakMethodTests(unittest.TestCase):
    def test_cpop_matches_joint_spline_least_squares_all_partitions(self):
        rng = np.random.default_rng(313)
        for n in range(3, 14):
            y = rng.normal(size=n)
            z = (y-y.mean())/y.std()
            t = np.arange(n)
            for min_size, jump, penalty in [(2, 1, .03), (3, 2, .1), (4, 1, 0)]:
                if min_size > n:
                    continue
                objectives = []
                for knots in partitions(n, min_size, jump, continuous=True):
                    x = np.column_stack([np.ones(n), t, *[np.maximum(t-k, 0) for k in knots]])
                    fit = x@np.linalg.lstsq(x, z, rcond=None)[0]
                    objectives.append(np.mean((z-fit)**2)+penalty*len(knots))
                result = segment_cpop(y, penalty, min_size, jump=jump)
                self.assertAlmostEqual(result.objective, min(objectives), places=9)
                self.assertLess(result.diagnostics["objective_error"], 1e-9)
                self.assertTrue(np.all(np.diff([0, *result.breakpoints]) >= min_size))

    def test_quadratic_envelope_against_all_candidates(self):
        rng = np.random.default_rng(9)
        for _ in range(30):
            candidates = [_Quadratic(float(a), float(b), float(c), 0)
                          for a, b, c in zip(rng.uniform(.01, 5, 25), rng.normal(0, 10, 25), rng.normal(0, 10, 25))]
            envelope, _ = _lower_envelope(candidates)
            x = np.r_[np.linspace(-100, 100, 500), rng.normal(0, 2, 500)]
            full = np.min([q.a*x*x+q.b*x+q.c for q in candidates], axis=0)
            reduced = np.min([q.a*x*x+q.b*x+q.c for q in envelope], axis=0)
            np.testing.assert_allclose(full, reduced, atol=1e-10)

    def test_polynomial_prefix_cost_against_direct_lstsq(self):
        rng = np.random.default_rng(77)
        for n in [30, 2001]:
            y = rng.normal(size=n)
            for degree in [1, 2]:
                cost = PolynomialCost(y, degree)
                for _ in range(30):
                    start = int(rng.integers(0, n-degree))
                    end = int(rng.integers(start+degree+1, n+1))
                    x = np.polynomial.polynomial.polyvander(np.linspace(-1, 1, end-start), degree)
                    fit = x@np.linalg.lstsq(x, y[start:end], rcond=None)[0]
                    direct = np.sum((y[start:end]-fit)**2)/n
                    self.assertAlmostEqual(cost.error(start, end), direct, places=8)

    def test_pelt_regression_matches_exhaustive_direct_rss(self):
        rng = np.random.default_rng(17)
        for n in range(5, 13):
            y = rng.normal(size=n)
            z = (y-y.mean())/y.std()
            for degree, min_size, jump in [(1, 2, 1), (2, 3, 1), (2, 4, 2)]:
                objectives = []
                for points in partitions(n, min_size, jump):
                    total, start = .04*len(points), 0
                    for end in [*points, n]:
                        x = np.polynomial.polynomial.polyvander(np.linspace(-1, 1, end-start), degree)
                        residual = z[start:end]-x@np.linalg.lstsq(x, z[start:end], rcond=None)[0]
                        total += np.sum(residual**2)/n
                        start = end
                    objectives.append(total)
                result = segment_regression(y, .04, min_size, degree=degree, jump=jump)
                self.assertAlmostEqual(result.objective, min(objectives), places=9)

    def test_rbf_pelt_against_direct_gram_matrix(self):
        y = np.random.default_rng(48).normal(size=12)
        z = (y-y.mean())/y.std()
        gamma = .7
        gram = np.exp(-gamma*(z[:, None]-z[None, :])**2)
        cost = RbfCost(z, gamma)
        for start in range(12):
            for end in range(start+1, 13):
                expected = (end-start-gram[start:end, start:end].sum()/(end-start))/12
                self.assertAlmostEqual(cost.error(start, end), expected)
        for min_size, jump in [(2, 1), (3, 2), (5, 1)]:
            objectives = []
            for points in partitions(12, min_size, jump):
                starts, ends = [0, *points], [*points, 12]
                value = .03*len(points)
                for s, e in zip(starts, ends):
                    value += ((e-s)-gram[s:e, s:e].sum()/(e-s))/12
                objectives.append(value)
            result = segment_rbf(y, .03, min_size, gamma=gamma, jump=jump)
            self.assertAlmostEqual(result.objective, min(objectives), places=10)

    def test_trend_filter_against_independent_bounded_dual_least_squares(self):
        y = np.random.default_rng(54).normal(size=18)
        z = (y-y.mean())/y.std()
        result = segment_trend_filter(y, .1, 3, knot_tolerance=1e-6)
        d2 = np.diff(np.eye(len(y)), n=2, axis=0)
        bound = result.penalty*len(y)
        dual = lsq_linear(d2.T, z, bounds=(-bound, bound), tol=1e-13, lsq_solver="exact")
        reference = z-d2.T@dual.x
        normalized_fit = (result.fit-y.mean())/y.std()
        np.testing.assert_allclose(normalized_fit, reference, atol=2e-5)
        self.assertLess(result.diagnostics["duality_gap"], 1e-7)
        self.assertLess(result.diagnostics["stationarity_residual"], 1e-6)

    def test_amplitude_units_and_constant_input(self):
        rng = np.random.default_rng(6)
        y = np.r_[np.linspace(0, 2, 30), np.linspace(2, -1, 30)] + rng.normal(0, .05, 60)
        methods = [segment_cpop, segment_regression, segment_rbf, segment_trend_filter]
        for method in methods:
            a = method(y, min_size=5, jump=3)
            b = method(y*1e-13, min_size=5, jump=3)
            self.assertEqual(a.breakpoints, b.breakpoints)
            np.testing.assert_allclose(a.fit, b.fit/1e-13, rtol=1e-7, atol=1e-7)
            c = method(np.ones(15)*1e-13, min_size=3)
            self.assertEqual(c.breakpoints, [15])
            np.testing.assert_allclose(c.fit, 1e-13)

    def test_matching_is_one_to_one_and_handles_negative_controls(self):
        m = match_breakpoints([99, 100, 101], [100], 2)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 2, 0))
        self.assertEqual(m["timing_mae"], 0)
        self.assertEqual(match_breakpoints([], [], 2)["f1"], 1)
        self.assertEqual(match_breakpoints([.3], [.28], .02)["tp"], 1)
        self.assertEqual(match_breakpoints([5], [], 2)["f1"], 0)
        self.assertTrue(np.isnan(match_breakpoints([], [5], 2)["timing_mae"]))

    def test_rectangle_comparison_adapter_matches_existing_detector(self):
        from utils.breakpoint_comparison import _segment_rectangles
        from src.rectangle_segmentation import segment_pelt
        y = np.random.default_rng(52).normal(size=80)
        old = segment_pelt(y, penalty=.03, min_size=7)
        new = _segment_rectangles(y, penalty=.03, min_size=7, jump=1)
        self.assertEqual(old.breakpoints, new.breakpoints)
        self.assertAlmostEqual(old.objective, new.objective)

    def test_invalid_inputs(self):
        for method in [segment_cpop, segment_regression, segment_rbf, segment_trend_filter]:
            for bad in [[], [1, np.nan, 3], [[1, 2, 3]]]:
                with self.assertRaises(ValueError):
                    method(bad)
            with self.assertRaises(ValueError):
                method([1, 2, 3, 4], min_size=2, times=[0, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
