"""Independent checks of the statistical definitions and epoch handling."""

from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import meg_eda as eda


class EdaTests(unittest.TestCase):
    def test_statistics_match_independent_definitions(self):
        rng = np.random.default_rng(4)
        y = rng.normal(size=(4, 200)) * 1e-13
        times = np.arange(200) / 1000
        metadata = eda.infer_channels(["MEG0112", "MEG0113", "MEG0122", "MEG0123"], source="test")
        table = eda.row_statistics(y, times, metadata, batch_size=3)
        np.testing.assert_allclose(table["std"], np.std(y, axis=1, ddof=1), rtol=1e-12)
        np.testing.assert_allclose(table["median"], np.median(y, axis=1), rtol=1e-12)
        np.testing.assert_allclose(table["rms"], np.sqrt(np.mean(y ** 2, axis=1)), rtol=1e-12)
        np.testing.assert_allclose(table.skewness, stats.skew(y, axis=1), atol=1e-12)
        np.testing.assert_allclose(table.excess_kurtosis, stats.kurtosis(y, axis=1), atol=1e-12)
        for i in range(4):
            self.assertAlmostEqual(table.loc[i, "acf_lag1"], np.corrcoef(y[i, :-1], y[i, 1:])[0, 1])
            self.assertAlmostEqual(table.loc[i, "linear_slope_per_s"] / 1e-13,
                                   np.polyfit(times, y[i] / 1e-13, 1)[0])

    def test_missing_and_constant_rows_are_reported_not_interpolated(self):
        data = np.array([[1, 1, 1, 1], [1, np.nan, 3, np.inf], [np.nan] * 4], dtype=float)
        metadata = eda.infer_channels(["a", "b", "c"], source="test")
        table = eda.row_statistics(data, np.arange(4), metadata)
        np.testing.assert_array_equal(table.n_finite, [4, 2, 0])
        self.assertTrue(table.loc[0, "constant"])
        self.assertEqual(table.loc[1, "mean"], 2)
        self.assertTrue(np.isnan(table.loc[1, "acf_lag1"]))
        self.assertTrue(np.isnan(table.loc[2, "mean"]))

    def test_spectral_bands_partition_and_known_frequency(self):
        times = np.arange(2000) / 1000
        y = np.sin(2 * np.pi * 10 * times)[None, :] * 1e-13
        result = eda.spectral_statistics(y, 1000, nperseg=1000)
        features = result["features"]
        band_columns = [c for c in features if c.endswith("_fraction") and not c.startswith(("line", "high"))]
        self.assertAlmostEqual(float(features[band_columns].sum(axis=1).iloc[0]), 1)
        self.assertGreater(features.alpha_8_13_fraction.iloc[0], .95)
        self.assertLess(abs(features.psd_peak_hz.iloc[0] - 10), result["frequency_resolution_hz"])
        self.assertTrue(0 <= features.spectral_entropy.iloc[0] <= 1)

    def test_excel_audit_keeps_marker_codes_without_assigning_meanings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.xlsx"
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(["Time", 0, .001, .002, .003, .004])
            sheet.append(["MEG0112", 1, 2, 3, 2, 1])
            sheet.append(["STI101", 0, 7, 7, 0, 32])
            workbook.save(path)
            result = eda.audit_excel(path)
            self.assertEqual(result["markers"].code.tolist(), [7, 32])
            self.assertEqual(result["markers"]["sample"].tolist(), [1, 4])
            np.testing.assert_allclose(result["markers"].duration_s, [.002, .001])
            self.assertAlmostEqual(result["summary"]["sfreq_hz"], 1000)

    def test_mne_trial_channel_mapping_and_bad_channel_inventory(self):
        info = mne.create_info(["MEG0112", "MEG0111", "EEG001"], 100, ["grad", "mag", "eeg"])
        info["bads"] = ["MEG0112"]
        values = np.arange(2 * 3 * 61).reshape(2, 3, 61).astype(float) * 1e-13
        epochs = mne.EpochsArray(values, info, events=np.array([[100, 0, 1], [200, 0, 3]]),
                                 event_id={"auditory/left": 1, "visual/left": 3}, tmin=-.1,
                                 baseline=None, verbose=False)
        data, times, metadata, inventory = eda.epochs_arrays(epochs)
        self.assertEqual(data.shape, (6, 61))
        np.testing.assert_array_equal(data[3], values[1, 0])
        self.assertEqual(metadata.condition.tolist(), ["auditory/left"] * 3 + ["visual/left"] * 3)
        self.assertEqual(metadata.epoch_idx.tolist(), [0] * 3 + [1] * 3)
        self.assertEqual(inventory.unit.tolist(), ["T/m", "T", "V"])
        self.assertEqual(inventory.marked_bad.tolist(), [True, False, False])

    def test_duplicate_or_irregular_time_axis_is_rejected(self):
        metadata = eda.infer_channels(["x"], source="test")
        with self.assertRaises(ValueError):
            eda.analyze_series([[1, 2, 3]], [0, .001, .004], metadata)
        with self.assertRaises(ValueError):
            eda.analyze_series([[1, 2, 3]], [0, 0, 1], metadata)


if __name__ == "__main__":
    unittest.main()
