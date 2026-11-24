"""Excel layout/window regression checks using synthetic workbooks only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import openpyxl

from utils.data_loading import load_xlsx_series


class GradientWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "synthetic.xlsx"

    def write(self, rows, window=None):
        wb = openpyxl.Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        if window is not None:
            for column, value in enumerate(window, 1):
                ws.cell(206, column, value)
        wb.save(self.path)
        wb.close()

    def test_legacy_layout_order_and_stimulus_skip(self):
        self.write([["Time", 0, .001, .002], ["a", 1, 2, 3],
                    ["STI101", 0, 1, 0], ["b", 4, 5, 6]])
        result = load_xlsx_series(self.path)
        self.assertEqual([s["channel_name"] for s in result], ["a", "b"])
        np.testing.assert_array_equal(result[0]["signal"], [1, 2, 3])
        self.assertAlmostEqual(result[0]["sfreq"], 1000)
        result = load_xlsx_series(self.path, ["b", "a"])
        self.assertEqual([s["channel_name"] for s in result], ["b", "a"])

    def test_sensor_column_window_and_absolute_time(self):
        self.write([[None, "Time", 10, 10.001, 10.002, 10.003, 10.004],
                    [1.2, "MEG0112", 1, 2, 3, 4, 5],
                    [1.3, "MEG0113", 6, 7, 8, 9, 10]],
                   ["window", None, None, -1, 0, 1, None])
        result = load_xlsx_series(self.path, ["MEG0113", "MEG0112"], window_row=206)
        self.assertEqual(len(result), 2)
        s = result[0]
        np.testing.assert_array_equal(s["signal"], [7, 8, 9])
        np.testing.assert_allclose(s["times"], [10.001, 10.002, 10.003])
        np.testing.assert_array_equal(s["window_values"], [-1, 0, 1])
        self.assertEqual((s["sample_start"], s["sample_stop"]), (1, 4))
        self.assertEqual(s["sensor_id"], 1.3)
        self.assertEqual(len(load_xlsx_series(self.path)), 2)

    def test_bad_window_missing_channel_and_nonfinite_data(self):
        rows = [[None, "Time", 0, .001, .002, .003], [1.2, "a", 1, 2, 3, 4]]
        for window in [["window", None, None, None],
                       ["window", None, -1, None, 1],
                       ["window", None, None, None, None, 0, 1]]:
            self.write(rows, window)
            with self.assertRaises(ValueError):
                load_xlsx_series(self.path, window_row=206)
        self.write(rows)
        for kwargs in [dict(window_row=206), dict(window_row=1),
                       dict(window_row=True), dict(channel_names=["missing"])]:
            with self.assertRaises(ValueError):
                load_xlsx_series(self.path, **kwargs)
        self.write([["Time", 0, .001, .002], ["a", 1, None, 3]])
        with self.assertRaises(ValueError):
            load_xlsx_series(self.path)

    def test_invalid_times_and_duplicates(self):
        for times in [[0, 0, 1], [0, .001, .004], [0, None, .002]]:
            self.write([["Time", *times], ["a", 1, 2, 3]])
            with self.assertRaises(ValueError):
                load_xlsx_series(self.path)
        self.write([["Time", 0, .001, .002], ["a", 1, 2, 3], ["a", 4, 5, 6]])
        with self.assertRaises(ValueError):
            load_xlsx_series(self.path)


if __name__ == "__main__":
    unittest.main()
