"""L1 linear trend filtering, with certified convex-QP solution.

Minimize 0.5*mean((z-f)^2) + lambda*||D2 f||_1, D2=[1,-2,1].
The signal is globally standardized. strength=lambda/lambda_max, where
lambda_max is the smallest penalty giving a global straight-line fit.
Clarabel solves the epigraph QP; primal/dual objectives and gap are reported.

The optimized fit is never refitted after knot extraction. min_size and
jump act ONLY on reported knots through explicitly documented nonmaximum
suppression and grid snapping; they are not constraints of the convex QP.
https://arxiv.org/abs/1304.2986
https://clarabel.org/stable/python/getting_started_py/
"""
from bisect import bisect_left
from time import perf_counter
import numpy as np
from scipy import sparse
import clarabel
from .breakpoint_common import BreakResult, nonnegative, prepare_signal


def _report_knots(curvature, n, min_size, jump, tolerance):
    active = np.flatnonzero(np.abs(curvature) > tolerance)+1
    grid = np.arange(jump, n, jump)
    grid = grid[(grid >= min_size) & (grid <= n-min_size)]
    scores = {}
    for point in active:
        if point < min_size or n-point < min_size or not len(grid):
            continue
        nearest = int(grid[np.argmin(np.abs(grid-point))])
        scores[nearest] = max(scores.get(nearest, 0.), abs(curvature[point-1]))
    kept = [0, n]
    for point in sorted(scores, key=lambda p: (-scores[p], p)):
        where = bisect_left(kept, point)
        if point-kept[where-1] >= min_size and kept[where]-point >= min_size:
            kept.insert(where, point)
    return kept[1:], active.tolist()


def segment_trend_filter(signal, strength=.03, min_size=10, *, lambda_value=None, jump=1, times=None,
                         knot_tolerance=1e-5, solver_tolerance=1e-9, max_iterations=250):
    """Return the L1 trend fit and consolidated detected slope changes.

    strength>=1 gives the global straight line; strength=0 gives the data.
    lambda_value, when supplied, sets the mean-loss L1 penalty directly and
    overrides strength. The returned strength is then lambda/lambda_max.
    tolerance is in globally standardized second-difference units.
    """
    started = perf_counter()
    y, z, offset, scale, min_size, jump = prepare_signal(signal, min_size, jump, times)
    strength = nonnegative(strength, "strength")
    knot_tolerance = nonnegative(knot_tolerance, "knot_tolerance")
    solver_tolerance = nonnegative(solver_tolerance, "solver_tolerance")
    if solver_tolerance == 0 or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("solver_tolerance and max_iterations must be positive")
    n, m = len(y), len(y)-2
    x = np.linspace(-1, 1, n)
    design = np.column_stack([np.ones(n), x])
    linear = design @ np.linalg.lstsq(design, z, rcond=None)[0]
    # D2.T u = residual, with residual orthogonal to 1 and x.
    dual_max = np.cumsum(np.cumsum(z-linear))[:-2]
    lambda_max = float(np.max(np.abs(dual_max)) / n)
    lam = strength*lambda_max if lambda_value is None else nonnegative(lambda_value, "lambda_value")
    effective_strength = lam/lambda_max if lambda_max > 0 else 0.
    d2 = sparse.diags([np.ones(m), -2*np.ones(m), np.ones(m)], [0, 1, 2], shape=(m, n), format="csc")
    iterations, status = 0, "analytic"
    if lam == 0:
        fit, dual = z.copy(), np.zeros(m)
    elif effective_strength >= 1 or lambda_max < 1e-14:
        fit, dual = linear, dual_max/n
    else:
        p = sparse.diags(np.r_[np.full(n, 1/n), np.zeros(m)], format="csc")
        q = np.r_[-z/n, np.full(m, lam)]
        identity = sparse.eye(m, format="csc")
        a = sparse.bmat([[d2, -identity], [-d2, -identity]], format="csc")
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = max_iterations
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = solver_tolerance
        solution = clarabel.DefaultSolver(p, q, a, np.zeros(2*m),
                                          [clarabel.NonnegativeConeT(2*m)], settings).solve()
        status = str(solution.status)
        if status != "Solved":
            raise RuntimeError(f"Trend filtering QP did not converge: {status}")
        fit = np.asarray(solution.x[:n])
        multipliers = np.asarray(solution.z)
        dual = multipliers[:m]-multipliers[m:]
        iterations = solution.iterations
    curvature = np.asarray(d2 @ fit)
    objective = float(.5*np.mean((z-fit)**2)+lam*np.sum(np.abs(curvature)))
    feasible_dual = np.clip(dual, -lam, lam)
    dt_dual = d2.T @ feasible_dual
    dual_objective = float(z@dt_dual-.5*n*(dt_dual@dt_dual))
    gap = max(0., objective-dual_objective)
    points, raw_knots = _report_knots(curvature, n, min_size, jump, knot_tolerance)
    return BreakResult("Trend-filter-L1", points, offset+scale*fit, objective, lam, perf_counter()-started,
        dict(strength=effective_strength, lambda_value=lam, lambda_max=lambda_max, min_size=min_size,
             jump=jump, knot_tolerance=knot_tolerance), dict(
            solver_status=status, converged=True, iterations=iterations,
            dual_objective=dual_objective, duality_gap=gap,
            stationarity_residual=float(np.max(np.abs((fit-z)/n+d2.T@dual))),
            raw_knots=raw_knots, raw_knot_count=len(raw_knots),
            curvature=curvature, postprocessing="strongest-curvature NMS; grid snapping; no refit",
            exact_segmentation=False, continuous_trend=True,
        ))
