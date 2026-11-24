"""Independent cost oracle and oscillations with known change mechanisms."""
import itertools
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.breakpoint_common import pelt_search
from src.oscillation_segmentation import FeatureCost, segment_oscillations
from utils.oscillation_plots import plot_oscillations


class OscillationTests(unittest.TestCase):
    def setUp(self):
        self.sfreq = 400.
        self.times = np.arange(2400)/self.sfreq

    def oscillation(self, kind):
        amplitude = np.where(self.times >= 3, 2., 1.) if kind == "amplitude" else 1.
        frequency = np.where(self.times >= 3, 30., 10.) if kind == "frequency" else np.full(2400, 10.)
        phase = 2*np.pi*np.cumsum(frequency)/self.sfreq
        return amplitude*np.sin(phase)

    def test_cost_and_pelt_against_direct_enumeration(self):
        x = np.random.default_rng(5).normal(size=(9, 3))
        cost = FeatureCost(x)
        def direct(points, penalty):
            starts = [0, *points[:-1]]
            return sum(np.sum((x[s:e]-x[s:e].mean(axis=0))**2)
                       for s, e in zip(starts, points))/len(x) + penalty*(len(points)-1)
        for s in range(len(x)):
            for e in range(s+1, len(x)+1):
                self.assertAlmostEqual(cost.error(s, e), np.sum((x[s:e]-x[s:e].mean(axis=0))**2)/len(x))
        for minimum in [1, 2, 3, 5]:
            for penalty in [0, .03, .2, 1]:
                objectives = []
                for flags in itertools.product([False, True], repeat=len(x)-1):
                    points = [i+1 for i, flag in enumerate(flags) if flag] + [len(x)]
                    if min(np.diff([0, *points])) >= minimum:
                        objectives.append(direct(points, penalty))
                points, objective, _ = pelt_search(cost, len(x), penalty, minimum)
                self.assertAlmostEqual(objective, min(objectives), places=12)
                self.assertAlmostEqual(objective, direct(points, penalty), places=12)

    def test_frequency_and_amplitude_changes_with_noise(self):
        for kind in ["frequency", "amplitude"]:
            with self.subTest(kind=kind):
                y = self.oscillation(kind) + np.random.default_rng(42).normal(0, .05, len(self.times))
                result = segment_oscillations(y, self.sfreq)
                self.assertEqual(len(result.change_points), 1)
                self.assertLessEqual(abs(result.change_points[0]/self.sfreq - 3), .2)
                self.assertEqual(result.breakpoints[-1], len(y))
                self.assertTrue(np.all(np.diff([0, *result.diagnostics["frame_breakpoints"]]) >= result.params["min_frames"]))

    def test_stationary_oscillation_and_constant_have_no_breaks(self):
        for y in [self.oscillation("stationary"), np.ones(2400)*1e-13, np.zeros(2400)]:
            result = segment_oscillations(y, self.sfreq)
            self.assertEqual(result.breakpoints, [len(y)])
            self.assertAlmostEqual(result.objective, 0, places=10)
        self.assertTrue(np.isnan(result.diagnostics["peak_hz"]).all())

    def test_physical_estimates_and_scale_invariance(self):
        y = self.oscillation("amplitude")
        base = segment_oscillations(y, self.sfreq)
        for transformed in [y*1e-13, y*-3 + 4]:
            result = segment_oscillations(transformed, self.sfreq)
            self.assertEqual(result.breakpoints, base.breakpoints)
            self.assertAlmostEqual(result.objective, base.objective)
        d = base.diagnostics
        before = d["frame_times"] < 2.5
        after = d["frame_times"] > 3.5
        np.testing.assert_allclose(d["peak_hz"], 10, atol=2.5)
        np.testing.assert_allclose(d["band_rms"][before], 1/np.sqrt(2), rtol=.01)
        np.testing.assert_allclose(d["band_rms"][after], 2/np.sqrt(2), rtol=.01)

    def test_ablation_separates_mechanisms(self):
        freq = segment_oscillations(self.oscillation("frequency"), self.sfreq, frequency_weight=0)
        amp = segment_oscillations(self.oscillation("amplitude"), self.sfreq, amplitude_weight=0)
        self.assertEqual(freq.change_points, [])
        self.assertEqual(amp.change_points, [])

    def test_time_mapping_plot_and_no_padding(self):
        y = self.oscillation("frequency")[:-7]
        t = self.times[:-7] + 100
        result = segment_oscillations(y, self.sfreq, times=t)
        d = result.diagnostics
        self.assertGreaterEqual(d["frame_times"][0], t[0])
        self.assertLessEqual(d["frame_starts"][-1]+result.params["window_samples"], len(y))
        b = d["frame_breakpoints"][0]
        midpoint = (d["frame_times"][b-1]+d["frame_times"][b])/2
        self.assertLessEqual(abs(t[result.change_points[0]]-midpoint), .5/self.sfreq+1e-12)
        fig, axes = plot_oscillations(dict(signal=y, times=t, label="synthetic"), result)
        self.assertAlmostEqual(axes[0].lines[1].get_xdata()[0], t[result.change_points[0]])
        plt.close(fig)

    def test_validation(self):
        y = self.oscillation("stationary")
        for kwargs in [dict(penalty=-1), dict(window_seconds=10), dict(window_seconds=.001),
                       dict(step_seconds=0), dict(step_seconds=.8), dict(min_duration=20),
                       dict(frequency_band=(0, 10)), dict(frequency_band=(1, 300)),
                       dict(frequency_band=(10, 10.1)), dict(amplitude_weight=0, frequency_weight=0),
                       dict(times=self.times*2)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                segment_oscillations(y, self.sfreq, **kwargs)
        for bad in [[], [1, np.nan, 3], [[1, 2, 3]]]:
            with self.assertRaises(ValueError):
                segment_oscillations(bad, self.sfreq)


if __name__ == "__main__":
    unittest.main()
