"""Continuous piecewise-linear optimal partitioning with functional pruning.

Minimizes RSS(z)/n + penalty*k, with fitted values continuous at all knots.
For each possible last knot, conditional costs are quadratics in the fitted
endpoint value. Only quadratics on their lower envelope survive. Endpoint
values are NOT discretized. This implements the functional-pruning CPOP
recurrence (h=0); the paper's additional inequality pruning is omitted.

All observations enter the loss, even when jump restricts candidate knot
locations. Exactness is relative to that grid and the stated length rule.
Uniform sampling is required. No cap/beam search or heuristic state deletion.
https://www.lancaster.ac.uk/staff/letchfoa/articles/cpop.pdf
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np
from .breakpoint_common import BreakResult, nonnegative, prepare_signal


@dataclass(eq=False)
class _Quadratic:
    a: float
    b: float
    c: float
    end: int
    parent: _Quadratic | None = None
    back_a: float = 0.0
    back_b: float = 0.0
    n_changes: int = 0


def _roots(a, b, c):
    """Stable real roots of a quadratic difference, without dividing tiny a."""
    if a == 0:
        return [] if b == 0 else [-c/b]
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return []
    root = math.sqrt(discriminant)
    q = -.5*(b+math.copysign(root, b))
    if q == 0:
        return [-b/(2*a)]
    return sorted(set([q/a, c/q]))


def _insert_envelope(pieces, new):
    """Exact algebraic envelope insertion over (-inf, inf), up to roundoff."""
    if not pieces:
        return [(-np.inf, np.inf, new)]
    output = []
    for lower, upper, old in pieces:
        da, db, dc = new.a-old.a, new.b-old.b, new.c-old.c
        roots = [x for x in _roots(da, db, dc) if np.isfinite(x) and lower < x < upper]
        edges = [lower, *roots, upper]
        for left, right in zip(edges[:-1], edges[1:]):
            if np.isfinite(left) and np.isfinite(right):
                point = left + (right-left)/2
            elif np.isfinite(left):
                point = left + max(1.0, abs(left)+1.0)
            elif np.isfinite(right):
                point = right-max(1.0, abs(right)+1.0)
            else:
                point = 0.0
            difference = (da*point+db)*point+dc
            chosen = new if difference < 0 or (difference == 0 and new.n_changes < old.n_changes) else old
            if output and output[-1][2] is chosen and output[-1][1] == left:
                output[-1] = (output[-1][0], right, chosen)
            else:
                output.append((left, right, chosen))
    return output


def _lower_envelope_reference(candidates):
    pieces = []
    # Good incumbents first reduce the intermediate envelope size.
    for candidate in sorted(candidates, key=lambda q: (q.c-q.b*q.b/(4*q.a), q.n_changes)):
        pieces = _insert_envelope(pieces, candidate)
    kept, seen = [], set()
    for _, _, candidate in pieces:
        if id(candidate) not in seen:
            seen.add(id(candidate))
            kept.append(candidate)
    return kept, len(pieces)


def _lower_envelope(candidates):
    """Vectorized left-to-right line search over quadratic intersections.

    A quadratic enters below the current one at its downward-crossing root.
    The earliest such crossing is the next envelope boundary. This avoids
    Python loops over every existing envelope piece for every candidate.
    The insertion implementation remains a fallback for numerical cycling.
    """
    coefficients = np.array([(q.a, q.b, q.c) for q in candidates], dtype=np.longdouble)
    aa, bb, cc = coefficients.T
    changes = np.array([q.n_changes for q in candidates])
    current = int(np.lexsort((changes, cc, -bb, aa))[0])
    left = np.longdouble(-np.inf)
    kept, seen, pieces = [], set(), 0
    while True:
        pieces += 1
        if current not in seen:
            seen.add(current)
            kept.append(candidates[current])
        if pieces > 10*len(candidates)+10:
            return _lower_envelope_reference(candidates)
        a, b, c = (coefficients-coefficients[current]).T
        enter = np.full(len(candidates), np.longdouble(np.inf))
        linear = (a == 0) & (b < 0)
        enter[linear] = -c[linear]/b[linear]
        discriminant = b*b-4*a*c
        quadratic = (a != 0) & (discriminant > 0)
        ids = np.flatnonzero(quadratic)
        if len(ids):
            root = np.sqrt(discriminant[ids])
            q = -.5*(b[ids]+np.copysign(root, b[ids]))
            r1 = q/a[ids]
            r2 = np.divide(c[ids], q, out=-b[ids]/(2*a[ids]), where=q != 0)
            enter[ids] = np.where(a[ids] > 0, np.minimum(r1, r2), np.maximum(r1, r2))
        margin = 128*np.finfo(np.longdouble).eps*max(1., abs(left)) if np.isfinite(left) else 0.
        enter[enter <= left+margin] = np.inf
        next_at = np.min(enter)
        if not np.isfinite(next_at):
            break
        tied = np.flatnonzero(np.abs(enter-next_at) <= 128*np.finfo(np.longdouble).eps*max(1., abs(next_at)))
        slopes = 2*aa[tied]*next_at+bb[tied]
        current = int(tied[np.lexsort((changes[tied], aa[tied], slopes))[0]])
        left = next_at
    return kept, pieces


def segment_cpop(signal, penalty=.02, min_size=10, *, jump=1, times=None):
    """Continuous least-squares trend and internal slope-change indices.

    Knots lie at observed sample indices. A knot b belongs to the right
    half-open reported segment; continuity means its fitted value is shared.
    min_size applies to lengths diff([0, *knots, n]). The final segment thus
    has one fewer interpolation interval than observations. The first and
    last observed samples are always fitted; neither is a detected knot.
    """
    started = perf_counter()
    y, z, offset, scale, min_size, jump = prepare_signal(signal, min_size, jump, times)
    penalty = nonnegative(penalty, "penalty")
    n = len(y)
    indices = np.arange(n, dtype=float)
    prefix_y = np.r_[0., np.cumsum(z)]
    prefix_xy = np.r_[0., np.cumsum(indices*z)]
    prefix_yy = np.r_[0., np.cumsum(z*z)]
    interior = [int(t) for t in range(jump, n-1, jump) if min_size <= t <= n-min_size]
    endpoints = [*interior, n-1]
    base = _Quadratic(1.0, -2*z[0], z[0]**2, 0)
    states = {0: [base]}
    generated, max_states, max_pieces = 0, 1, 1
    state_counts = []
    for end in endpoints:
        candidates = []
        for start, previous in states.items():
            if start >= end:
                continue
            required_length = (n-start) if end == n-1 else (end-start)
            if required_length < min_size:
                continue
            length = end-start
            sw = (length+1)/2
            sw2 = (length+1)*(2*length+1)/(6*length)
            aa = length-2*sw+sw2
            bb = 2*(sw-sw2)
            cc = sw2
            sy = prefix_y[end+1]-prefix_y[start+1]
            swy = (prefix_xy[end+1]-prefix_xy[start+1]-start*sy)/length
            dd, ee = -2*(sy-swy), -2*swy
            ff = prefix_yy[end+1]-prefix_yy[start+1]
            charge = penalty*n if start else 0.0
            for old in previous:
                denominator = old.a+aa
                linear = old.b+dd
                candidates.append(_Quadratic(
                    cc-bb*bb/(4*denominator), ee-bb*linear/(2*denominator),
                    old.c+ff+charge-linear*linear/(4*denominator), end, old,
                    -bb/(2*denominator), -linear/(2*denominator),
                    old.n_changes+int(start > 0),
                ))
        if not candidates:
            continue
        generated += len(candidates)
        kept, n_pieces = _lower_envelope(candidates)
        states[end] = kept
        state_counts.append((end, len(kept)))
        max_states, max_pieces = max(max_states, len(kept)), max(max_pieces, n_pieces)
    final = min(states[n-1], key=lambda q: (q.c-q.b*q.b/(4*q.a), q.n_changes))
    dp_objective = (final.c-final.b*final.b/(4*final.a))/n
    value, state = -final.b/(2*final.a), final
    knot_indices, knot_values = [], []
    while True:
        knot_indices.append(state.end)
        knot_values.append(value)
        if state.parent is None:
            break
        value = state.back_a*value+state.back_b
        state = state.parent
    knot_indices, knot_values = knot_indices[::-1], knot_values[::-1]
    fit = np.interp(indices, knot_indices, knot_values)
    points = [*knot_indices[1:-1], n]
    objective = float(np.mean((z-fit)**2)+penalty*(len(points)-1))
    return BreakResult("CPOP", points, offset+scale*fit, objective, penalty, perf_counter()-started,
        dict(min_size=min_size, jump=jump), dict(
            exact_on_grid=True, full_sample_grid=jump == 1, continuous=True,
            pruning="functional lower envelope only", candidates_generated=generated,
            max_states_per_endpoint=max_states, max_envelope_pieces=max_pieces,
            state_counts=state_counts, dp_objective=dp_objective,
            objective_error=abs(objective-dp_objective),
            knot_indices=knot_indices, knot_values=(offset+scale*np.array(knot_values)).tolist(),
        ))
