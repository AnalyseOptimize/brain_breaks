"""PELT with independent linear/quadratic least-squares segment models.

Minimizes RSS(z)/n + penalty*k, z globally standardized. The basis is local
to each segment for conditioning. No continuity constraint is imposed.
https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costlinear/
"""
from time import perf_counter
import numpy as np
from .breakpoint_common import BreakResult, integer, nonnegative, prepare_signal, pelt_search


class PolynomialCost:
    """O(1) sufficient-statistic RSS queries for an already normalized signal.

    Local symmetric orthogonal basis: 1, u and u²-mean(u²), u in [-1,1].
    Prefix arithmetic uses long double to reduce cancellation on short
    segments near the end of long records.
    """
    def __init__(self, signal, degree=1):
        degree = integer(degree, "degree")
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.signal = np.asarray(signal, dtype=float)
        self.degree, self.n = degree, len(signal)
        y = self.signal.astype(np.longdouble)
        x = np.arange(self.n, dtype=np.longdouble)
        self.prefix = [np.r_[np.longdouble(0), np.cumsum(v)] for v in (y, x*y, x*x*y, y*y)]

    def error_many(self, starts, end):
        starts = np.asarray(starts, dtype=int)
        length = (end - starts).astype(np.longdouble)
        a = starts.astype(np.longdouble)
        sy, sxy, sxxy, syy = [p[end] - p[starts] for p in self.prefix]
        h = length - 1
        local_xy = sxy - a*sy
        yu = 2*local_xy/h - sy
        u2 = length*(length+1)/(3*h)
        rss = syy - sy*sy/length - yu*yu/u2
        if self.degree == 2:
            local_xxy = sxxy - 2*a*sxy + a*a*sy
            yu2 = 4*local_xxy/(h*h) - 4*local_xy/h + sy
            yq = yu2 - u2*sy/length
            u4 = length*(length+1)*(3*length*length-7)/(15*h**3)
            norm2 = u4-u2*u2/length
            rss -= yq*yq/norm2
        return np.maximum(np.asarray(rss / self.n, dtype=float), 0)

    def error(self, start, end):
        if not 0 <= start < end <= self.n or end-start < self.degree+1:
            raise ValueError("Invalid or rank-deficient segment")
        return float(self.error_many(np.array([start]), end)[0])


def segment_regression(signal, penalty=.02, min_size=10, *, degree=1, jump=1, times=None):
    """Return boundaries and the independent per-segment polynomial fits."""
    started = perf_counter()
    y, z, offset, scale, min_size, jump = prepare_signal(signal, min_size, jump, times)
    penalty = nonnegative(penalty, "penalty")
    degree = integer(degree, "degree")
    if degree not in (1, 2) or min_size < degree+1:
        raise ValueError("degree must be 1/2 and min_size >= degree+1")
    cost = PolynomialCost(z, degree)
    points, optimum, diagnostics = pelt_search(cost, len(y), penalty, min_size, jump)
    fit = np.empty_like(z)
    coefficients, start = [], 0
    for end in points:
        x = np.linspace(-1, 1, end-start)
        coefficient = np.polynomial.legendre.legfit(x, z[start:end], degree)
        fit[start:end] = np.polynomial.legendre.legval(x, coefficient)
        coefficients.append(coefficient.tolist())
        start = end
    objective = float(np.mean((z-fit)**2) + penalty*(len(points)-1))
    diagnostics.update(dp_objective=optimum, objective_error=abs(optimum-objective),
                       coefficients_normalized=coefficients, continuous=False)
    return BreakResult(f"PELT-poly{degree}", points, offset+scale*fit, objective, penalty,
                       perf_counter()-started, dict(degree=degree, min_size=min_size, jump=jump), diagnostics)
