"""Загрузка MNE sample и Excel, выбор каналов и проб."""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import openpyxl
import mne
from mne.datasets import sample
from .paths import DATA_DIR


def load_sample_epochs(
    condition: Optional[str] = "visual/left",
    tmin: float = -0.1,
    tmax: float = 0.5,
    *,
    data_dir: Optional[str] = None,
) -> mne.Epochs:
    """Load the MNE sample dataset and return bandpass-filtered, epoched data.

    Parameters
    ----------
    condition:
        Event label to epoch on. Available: ``"visual/left"``,
        ``"visual/right"``, ``"auditory/left"``, ``"auditory/right"``.
        ``None`` returns all four conditions in one load (for EDA).
    tmin, tmax:
        Epoch window in seconds relative to the stimulus onset.
    data_dir:
        Parent directory for the cached MNE sample dataset. Defaults to the
        project data/ directory, independently of the working directory.

    Returns
    -------
    mne.Epochs
        Baseline-corrected epochs for the requested condition.
    """
    data_path = sample.data_path(path=data_dir if data_dir is not None else str(DATA_DIR))
    meg_path = data_path / "MEG" / "sample"
    raw_fname = meg_path / "sample_audvis_raw.fif"

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)

    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    event_id = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
        preload=True,
        verbose=False,
    )
    return epochs if condition is None else epochs[condition]


# ---------------------------------------------------------------------------
# Series selection
# ---------------------------------------------------------------------------


def select_series(
    epochs: mne.Epochs,
    epoch_indices: Optional[list[int]] = None,
    channel_names: Optional[list[str]] = None,
    n_random: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Extract single-channel, single-trial time series from epochs.

    Parameters
    ----------
    epochs:
        MNE Epochs object (already loaded into memory).
    epoch_indices:
        If given together with *channel_names*, use exactly these
        (epoch, channel) pairs.  Must have the same length as *channel_names*.
    channel_names:
        See *epoch_indices*.
    n_random:
        Number of (epoch, channel) pairs to sample when *epoch_indices* /
        *channel_names* are not provided.
    seed:
        Random seed for reproducible random selection.

    Returns
    -------
    list[dict]
        Each element has keys: ``signal``, ``times``, ``sfreq``,
        ``epoch_idx``, ``channel_name``, ``label``.
    """
    data = epochs.get_data(verbose=False)  # (n_epochs, n_channels, n_times)
    times = epochs.times
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    if epoch_indices is not None and channel_names is not None:
        pairs = list(zip(epoch_indices, channel_names))
    else:
        rng = np.random.default_rng(seed)
        ep_idx = rng.integers(0, data.shape[0], size=n_random).tolist()
        ch_idx = rng.integers(0, data.shape[1], size=n_random).tolist()
        pairs = [(e, ch_names[c]) for e, c in zip(ep_idx, ch_idx)]

    series_list = []
    for ep_idx, ch_name in pairs:
        ch_idx = ch_names.index(ch_name)
        signal = data[ep_idx, ch_idx, :]
        series_list.append(
            {
                "signal": signal,
                "times": times,
                "sfreq": sfreq,
                "epoch_idx": ep_idx,
                "channel_name": ch_name,
                "label": f"ep{ep_idx} / {ch_name}",
            }
        )
    return series_list


def _cells_to_floats(cells) -> np.ndarray:
    """Convert a row of Excel cell values to floats (empty cells -> NaN)."""
    return np.array([np.nan if c is None else float(c) for c in cells], dtype=float)


def load_xlsx_series(
    path: str,
    channel_names: Optional[list[str]] = None,
    trial_label: Optional[str] = None,
) -> list[dict]:
    """Load single-trial MEG/EEG series from a collaborator Excel export.

    Expects the "sideways" layout the collaborators use:

    * column A holds row labels;
    * the ``Time`` row gives the time axis in seconds, one sample per column;
    * every other row is one channel (label in column A, samples across the
      remaining columns);
    * stimulus rows (label starting with ``STI``) are skipped.

    The sampling frequency is inferred from the ``Time`` row. Returns the same
    list-of-dicts structure as :func:`select_series`, so the segmentation and
    plotting helpers work on it unchanged. This is the entry point for new
    files the collaborators send in the same format.

    Parameters
    ----------
    path:
        Path to the ``.xlsx`` file.
    channel_names:
        If given, return only these channels, in this order. Otherwise return
        every channel in the file, in sheet order.
    trial_label:
        Short tag used in plot titles. Defaults to the file stem.

    Returns
    -------
    list[dict]
        One dict per channel with keys ``signal``, ``times``, ``sfreq``,
        ``epoch_idx``, ``channel_name``, ``label``.
    """
    if trial_label is None:
        trial_label = os.path.splitext(os.path.basename(path))[0]

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    times = None
    channels: dict[str, np.ndarray] = {}
    for row in ws.iter_rows(values_only=True):
        label = row[0]
        if label is None:
            continue
        if label == "Time":
            times = _cells_to_floats(row[1:])
        elif str(label).startswith("STI"):
            continue  # stimulus/marker row, not a data channel
        else:
            channels[str(label)] = _cells_to_floats(row[1:])
    wb.close()

    if times is None:
        raise ValueError(f"No 'Time' row found in {path}")

    # Trim any trailing empty columns (openpyxl padding) using the Time row.
    finite = np.flatnonzero(np.isfinite(times))
    n = int(finite[-1]) + 1 if finite.size else 0
    times = times[:n]
    sfreq = 1.0 / np.median(np.diff(times))

    names = channel_names if channel_names is not None else list(channels)
    series_list = []
    for name in names:
        series_list.append(
            {
                "signal": channels[name][:n],
                "times": times,
                "sfreq": sfreq,
                "epoch_idx": 0,  # one trial per file
                "channel_name": name,
                "label": f"{trial_label} / {name}",
            }
        )
    return series_list


