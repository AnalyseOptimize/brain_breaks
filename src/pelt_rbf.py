"""Exact-on-grid PELT with Gaussian-kernel within-segment dispersion.

C(s,e) = ((e-s) - sum(K[s:e,s:e])/(e-s))/n. K has diagonal one.
This cost detects distributional changes, not specifically slope changes.
https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costrbf/
"""
from time import perf_counter
import numpy as np
from scipy.spatial.distance import pdist
from .breakpoint_common import BreakResult, nonnegative, prepare_signal, pelt_search


class RbfCost:
    """O(n²) memory/preparation, O(1) cost via a 2-D prefix sum."""
    def __init__(self, signal, gamma=None):
        z = np.asarray(signal, dtype=float)
        self.n = len(z)
        if gamma is None:
            distances = pdist(z[:, None], metric="sqeuclidean")
            median = float(np.median(distances))
            if median == 0:
                positive = distances[distances > 0]
                median = float(np.median(positive)) if len(positive) else 1.0
            gamma = 1.0/median
        if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        self.gamma = float(gamma)
        gram = np.exp(-self.gamma*(z[:, None]-z[None, :])**2)
        self.prefix = np.zeros((self.n+1, self.n+1))
        self.prefix[1:, 1:] = np.cumsum(np.cumsum(gram, axis=0), axis=1)

    def error_many(self, starts, end):
        starts = np.asarray(starts, dtype=int)
        length = end-starts
        p = self.prefix
        total = p[end, end]-p[starts, end]-p[end, starts]+p[starts, starts]
        return np.maximum((length-total/length)/self.n, 0)

    def error(self, start, end):
        if not 0 <= start < end <= self.n:
            raise ValueError("Invalid segment")
        return float(self.error_many(np.array([start]), end)[0])


def segment_rbf(signal, penalty=.02, min_size=10, *, gamma=None, jump=1, times=None):
    """Return RBF segmentation; fit is a mean per segment for display ONLY."""
    started = perf_counter()
    y, z, offset, scale, min_size, jump = prepare_signal(signal, min_size, jump, times)
    penalty = nonnegative(penalty, "penalty")
    cost = RbfCost(z, gamma)
    points, objective, diagnostics = pelt_search(cost, len(y), penalty, min_size, jump)
    fit, start = np.empty_like(z), 0
    for end in points:
        fit[start:end] = z[start:end].mean()
        start = end
    diagnostics.update(gamma=cost.gamma, fit_role="segment mean for display, not the kernel model")
    return BreakResult("PELT-RBF", points, offset+scale*fit, objective, penalty,
                       perf_counter()-started, dict(min_size=min_size, jump=jump, gamma=cost.gamma), diagnostics)
