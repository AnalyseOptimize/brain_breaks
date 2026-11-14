"""Детекция ruptures и непрерывная кусочно-полиномиальная аппроксимация."""
from __future__ import annotations
from typing import Optional
import numpy as np
import ruptures as rpt


def segment_ruptures(
    signal: np.ndarray,
    penalty: float = 3.0,
    model: str = "l2",
    min_size: int = 10,
    jump: int = 5,
) -> list[int]:
    """Detect change points in *signal* using ``ruptures.Pelt``.

    Parameters
    ----------
    signal:
        1-D time series.
    penalty:
        Regularisation penalty for ``Pelt``.  Higher values produce fewer
        breakpoints.  Typical range for MEG at ~600 Hz / 0.6 s epochs: 1–10.
    model:
        Cost model passed to ``Pelt``.  ``"l2"`` detects changes in mean;
        ``"rbf"`` is more general (detects changes in mean *and* variance).
    min_size:
        Minimum number of samples in a segment.
    jump:
        Subsampling step used by the dynamic-programming search.

    Returns
    -------
    list[int]
        Sorted breakpoint indices (end-of-segment convention, last element is
        ``len(signal)``).
    """
    # Z-score normalise so the penalty is scale-independent.
    # MEG signals are O(1e-13 T); without normalisation the squared-error cost
    # is O(1e-26) and no penalty value triggers a breakpoint.
    std = signal.std()
    signal_norm = signal / std if std > 0 else signal

    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    algo.fit(signal_norm)
    breakpoints = algo.predict(pen=penalty)
    return breakpoints


# ---------------------------------------------------------------------------
# Polynomial fitting between breakpoints
# ---------------------------------------------------------------------------


def fit_piecewise_poly(
    signal: np.ndarray,
    breakpoints: list[int],
    degree: int = 3,
) -> np.ndarray:
    """Fit a continuous piecewise polynomial via the truncated power basis.

    Parameters
    ----------
    signal:
        1-D time series of length *N*.
    breakpoints:
        Breakpoint list in end-of-segment convention (last element must equal
        ``len(signal)``).  Produced by :func:`segment_ruptures`.
    degree:
        Polynomial degree per segment.  3 (cubic) is the default.

    Returns
    -------
    np.ndarray
        Piecewise polynomial reconstruction, same shape as *signal*.

    Notes
    -----
    Uses the truncated power basis::

        1, t, t², …, t^d,  (t − τ_j)^1_+, (t − τ_j)^2_+, …, (t − τ_j)^d_+

    for each internal knot τ_j.  The linear term ``(t − τ_j)^1_+`` introduces
    a kink (first-derivative jump) at each knot while the function value stays
    continuous — giving exactly C⁰ smoothness.  Higher-order truncated powers
    allow jumps in the 2nd and higher derivatives as well.

    The global time axis is normalised to ``[0, 1]`` before constructing the
    basis to keep the design matrix well-conditioned.  Coefficients are found
    by least-squares (``numpy.linalg.lstsq``).
    """
    n = len(signal)
    t = np.linspace(0.0, 1.0, n)

    # Internal knots in normalised coordinates (exclude the sentinel)
    knots = [bp / n for bp in breakpoints[:-1]]

    # Build design matrix: global polynomial + truncated powers at each knot
    cols = [t**k for k in range(degree + 1)]
    for knot in knots:
        for k in range(1, degree + 1):
            cols.append(np.maximum(0.0, t - knot) ** k)

    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    return X @ coeffs


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def segment_and_fit(
    signal: np.ndarray,
    degree: int = 3,
    penalty: float = 3.0,
    model: str = "l2",
    min_size: int = 10,
    jump: int = 5,
    min_deriv_jump: Optional[float] = None,
) -> dict:
    """Detect change points and fit piecewise polynomials.

    Convenience wrapper that calls :func:`segment_ruptures` followed by
    :func:`fit_piecewise_poly`.  Optionally prunes breakpoints where the
    derivative discontinuity in the fitted curve is too small.

    Parameters
    ----------
    signal:
        1-D time series.
    degree:
        Polynomial degree for segment fitting.
    penalty, model, min_size, jump:
        Forwarded to :func:`segment_ruptures`.
    min_deriv_jump:
        If given, breakpoints where the absolute first-derivative jump in the
        fitted curve is below this threshold (measured in units of
        ``signal.std() / sample``, i.e. normalised) are removed and the
        polynomial is refit on the merged segments.  ``None`` disables
        pruning.  Typical range: 0.01 – 0.5.

    Returns
    -------
    dict
        Keys:

        ``breakpoints``
            List of breakpoint indices (end-of-segment convention).
        ``fit``
            Piecewise polynomial reconstruction (1-D array, same length as
            *signal*).
        ``params``
            Dict of parameters used (useful for plot captions).
    """
    breakpoints = segment_ruptures(
        signal, penalty=penalty, model=model, min_size=min_size, jump=jump
    )
    fit = fit_piecewise_poly(signal, breakpoints, degree=degree)

    if min_deriv_jump is not None and len(breakpoints) > 1:
        # Derivative jump at breakpoint bp: difference of finite-difference
        # slopes on the right and left of bp, normalised by signal std.
        sig_std = signal.std() or 1.0
        internal = breakpoints[:-1]  # exclude sentinel
        kept = []
        for bp in internal:
            if bp == 0 or bp >= len(fit) - 1:
                kept.append(bp)
                continue
            left_slope = fit[bp] - fit[bp - 1]
            right_slope = fit[bp + 1] - fit[bp]
            jump_norm = abs(right_slope - left_slope) / sig_std
            if jump_norm >= min_deriv_jump:
                kept.append(bp)
        breakpoints = kept + [len(signal)]
        fit = fit_piecewise_poly(signal, breakpoints, degree=degree)

    return {
        "breakpoints": breakpoints,
        "fit": fit,
        "params": {
            "degree": degree,
            "penalty": penalty,
            "model": model,
            "min_size": min_size,
            "jump": jump,
            "min_deriv_jump": min_deriv_jump,
        },
    }


