"""Changes in band amplitude and spectral shape using the shared PELT solver.

Hann-window periodograms give band RMS A and normalized power p. Features
are [sqrt(w_A)*log2(A/A_ref), sqrt(w_f)*sqrt(p)]. No per-bin whitening is
used: weak/noisy bins cannot acquire unit variance independently. Minimize
sum of within-segment feature SSE / n_frames + penalty * n_changes.

This is an offline detector on overlapping windows. PELT is exact on that
feature grid; reported sample indices are midpoints between frame centers,
not sample-accurate estimates. Windowing blurs abrupt transitions.
"""
from time import perf_counter

import numpy as np
from scipy.signal import periodogram

from .breakpoint_common import BreakResult, nonnegative, prepare_signal, pelt_search


class FeatureCost:
    """Multivariate SSE / number of frames, with prefix-sum queries."""

    def __init__(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("features must be a finite nonempty 2-D array")
        self.n = len(x)
        # Center before accumulation to reduce cancellation.
        x = (x - x.mean(axis=0)).astype(np.longdouble)
        self.prefix = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
        self.squares = np.r_[np.longdouble(0), np.cumsum(np.sum(x*x, axis=1))]

    def error_many(self, starts, end):
        starts = np.asarray(starts, dtype=int)
        total = self.prefix[end] - self.prefix[starts]
        rss = self.squares[end] - self.squares[starts] - np.sum(total*total, axis=1)/(end-starts)
        return np.maximum(np.asarray(rss/self.n, dtype=float), 0)

    def error(self, start, end):
        if not 0 <= start < end <= self.n:
            raise ValueError("Invalid segment")
        return float(self.error_many(np.array([start]), end)[0])


def _positive(value, name):
    value = nonnegative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def segment_oscillations(
    signal, sfreq, penalty=.02, *, window_seconds=.4, step_seconds=.05,
    min_duration=.5, frequency_band=(2., 100.), amplitude_weight=1.,
    frequency_weight=1., times=None,
):
    """Return a BreakResult with endpoints in ORIGINAL sample coordinates.

    ``min_duration`` constrains the number of consecutive frame steps per
    segment (ceil(duration / actual step)); it is not a raw-sample PELT
    constraint. Full windows only, without padding or data outside signal.
    The uncovered tail, shorter than a step, inherits the final regime.

    Amplitude is sqrt(integrated band PSD), in input units. Its log is
    floored at 1e-12 of maximum band RMS. Below that floor, spectral shape
    is set to zero and peak/centroid to NaN (frequency is undefined).
    Global nonzero multiplication or DC offset does not affect segmentation.
    With zero weight, the corresponding feature block is ignored.

    ``fit`` is the raw segment mean for compatibility/display only, not an
    oscillation reconstruction. Features, their segment means, frame times,
    band RMS and frequencies are exposed in ``diagnostics``. No physiological
    labels or calibrated significance are inferred from the boundaries.
    """
    started = perf_counter()
    y, z, _, scale, _, _ = prepare_signal(signal, times=times)
    sfreq = _positive(sfreq, "sfreq")
    penalty = nonnegative(penalty, "penalty")
    window_seconds = _positive(window_seconds, "window_seconds")
    step_seconds = _positive(step_seconds, "step_seconds")
    min_duration = _positive(min_duration, "min_duration")
    amplitude_weight = nonnegative(amplitude_weight, "amplitude_weight")
    frequency_weight = nonnegative(frequency_weight, "frequency_weight")
    if amplitude_weight + frequency_weight == 0:
        raise ValueError("At least one feature weight must be positive")
    t = np.arange(len(y))/sfreq if times is None else np.asarray(times, dtype=float)
    if not np.allclose(np.diff(t), 1/sfreq, rtol=1e-5, atol=1e-12):
        raise ValueError("sfreq must agree with times")
    band = np.asarray(frequency_band, dtype=float)
    if band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] <= sfreq/2:
        raise ValueError("frequency_band must satisfy 0 < low < high <= Nyquist")
    window = int(round(window_seconds*sfreq))
    step = int(round(step_seconds*sfreq))
    if not 4 <= window <= len(y):
        raise ValueError("window must contain >=4 samples and fit inside signal")
    if not 1 <= step <= window:
        raise ValueError("step must contain >=1 sample and not exceed window")
    starts = np.arange(0, len(y)-window+1, step)
    min_frames = max(1, int(np.ceil(min_duration*sfreq/step - 1e-10)))
    if len(starts) < max(2, min_frames):
        raise ValueError("Too few complete windows for min_duration")
    frames = np.lib.stride_tricks.sliding_window_view(z, window)[::step]
    freqs, psd = periodogram(frames, fs=sfreq, window="hann", detrend="constant", axis=-1)
    selected = (freqs >= band[0]) & (freqs <= band[1])
    if np.count_nonzero(selected) < 2:
        raise ValueError("frequency_band must contain >=2 Fourier bins; increase window")
    freqs, psd = freqs[selected], psd[:, selected]
    power = psd.sum(axis=1)
    rms = np.sqrt(power * sfreq/window)
    reference = float(np.max(rms)) or 1.
    relative = rms/reference
    active = relative > 1e-12
    shape = np.divide(psd, power[:, None], out=np.zeros_like(psd), where=power[:, None] > 0)
    shape[~active] = 0
    log_amplitude = np.log2(np.maximum(relative, 1e-12))
    features = np.column_stack([
        np.sqrt(amplitude_weight)*log_amplitude,
        np.sqrt(frequency_weight)*np.sqrt(shape),
    ])
    cost = FeatureCost(features)
    frame_points, objective, diagnostics = pelt_search(cost, len(starts), penalty, min_frames)
    centers = starts + (window-1)/2
    # b indexes the first frame of the right regime. Round the midpoint to
    # the closest raw sample, choosing the right sample on an exact tie.
    internal = np.asarray(frame_points[:-1], dtype=int)
    sample_points = np.floor((centers[internal-1] + centers[internal])/2 + .5).astype(int).tolist()
    sample_points.append(len(y))
    fit, feature_fit = np.empty_like(y), np.empty_like(features)
    start = 0
    for end in sample_points:
        fit[start:end] = y[start:end].mean()
        start = end
    start = 0
    for end in frame_points:
        feature_fit[start:end] = features[start:end].mean(axis=0)
        start = end
    peak = freqs[shape.argmax(axis=1)].copy()
    centroid = shape @ freqs
    peak[~active], centroid[~active] = np.nan, np.nan
    diagnostics.update(
        optimization_domain="window features", full_sample_grid=False,
        frame_breakpoints=frame_points, frame_starts=starts,
        frame_times=np.interp(centers, np.arange(len(y)), t),
        features=features, feature_fit=feature_fit, band_rms=rms*scale,
        frequencies_hz=freqs, relative_power=shape, peak_hz=peak,
        centroid_hz=centroid, frequency_resolution_hz=sfreq/window,
        uncovered_tail_samples=int(len(y)-(starts[-1]+window)),
        window_half_width_s=window/(2*sfreq),
        fit_role="raw segment mean for display only",
    )
    return BreakResult(
        "PELT-spectrum", sample_points, fit, objective, penalty,
        perf_counter()-started,
        dict(sfreq=sfreq, window_samples=window, step_samples=step,
             window_seconds=window/sfreq, step_seconds=step/sfreq,
             min_duration=min_duration, min_frames=min_frames,
             frequency_band=band.tolist(), amplitude_weight=amplitude_weight,
             frequency_weight=frequency_weight), diagnostics,
    )
