"""Structural breaks from axis-aligned rectangle covers (Doerr et al., 2016).

Both optimizers minimize the SAME additive criterion

    J(B) = sum((end-start) * range(y[start:end])) / (n * range(y))
           + penalty * number_of_internal_breaks.

This is the area/complexity trade-off in penalty form. A weighted criterion
``(1-alpha) * area_fraction + alpha * k / K`` gives the same minimizer with
``penalty = alpha / ((1-alpha) * K)``; constants from counting endpoints do
not affect the minimizer. Here the linear penalty is NOT clipped at K.

Segments use Python's [start, end) convention; the final endpoint n is always
returned and is not a detected break. Time is measured in samples, as in the
paper. Plotting can use an arbitrary strictly increasing physical time axis.

The GA implements sparse genomes, mixed uniform/one-point crossover, biased
mutation, independent populations and a final champions population. Budgets
are configurable. PELT is an exact alternative for this additive criterion,
with delayed pruning when min_size > 1. Static sparse tables replace the
paper's range tree: O(n log n) setup/space, O(1) range-extrema queries.

Sources:
https://doi.org/10.1007/s00500-016-2079-0 (published online 2016, issue 2017)
https://link.springer.com/article/10.1007/s00500-016-2079-0/tables/3
https://www2.imm.dtu.dk/~pafi/TSX/ (authors' reference implementation)
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class SegmentRectangle:
    """A segment and its enclosing amplitude interval, in original units."""

    start: int
    end: int
    lower: float
    upper: float

    @property
    def area(self):
        """Sample-cell area; width is end-start, including singleton cells."""
        return (self.end - self.start) * (self.upper - self.lower)


@dataclass
class RectangleSegmentation:
    """Optimizer output; breakpoints includes n, change_points excludes it."""

    method: str
    breakpoints: list[int]
    rectangles: list[SegmentRectangle]
    area_fraction: float
    penalty: float
    objective: float
    elapsed_seconds: float
    min_size: int
    history: np.ndarray = field(default_factory=lambda: np.empty(0))
    diagnostics: dict = field(default_factory=dict)

    @property
    def change_points(self):
        return self.breakpoints[:-1]


class RectangleCost:
    """Normalized enclosing-rectangle area on a finite univariate signal.

    Signals are range-normalized globally (not independently per segment).
    Constant signals have zero area. No smoothing, interpolation or removal
    of observations is performed. NaN/Inf and multichannel input are rejected.
    """

    def __init__(self, signal):
        y = np.asarray(signal, dtype=float)
        if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
            raise ValueError("signal must be a nonempty finite 1-D array")
        self.signal = y.copy()
        self.n = len(y)
        # Scale before subtraction to avoid overflow for large finite inputs
        # and underflow for signals in the MEG amplitude range.
        scale = float(np.max(np.abs(y))) or 1.0
        z = y / scale
        span = float(np.max(z) - np.min(z))
        z = (z - np.min(z)) / span if span else np.zeros_like(z)
        self.logs = np.zeros(self.n + 1, dtype=int)
        for i in range(2, self.n + 1):
            self.logs[i] = self.logs[i // 2] + 1
        shape = (int(self.logs[-1]) + 1, self.n)
        self.minimum = np.empty(shape)
        self.maximum = np.empty(shape)
        self.minimum[0] = self.maximum[0] = z
        for level in range(1, shape[0]):
            half = 1 << (level - 1)
            count = self.n - (1 << level) + 1
            self.minimum[level, :count] = np.minimum(
                self.minimum[level - 1, :count], self.minimum[level - 1, half:half + count]
            )
            self.maximum[level, :count] = np.maximum(
                self.maximum[level - 1, :count], self.maximum[level - 1, half:half + count]
            )

    def _areas(self, starts, ends):
        """Vectorized cost for validated nonempty half-open intervals."""
        lengths = np.asarray(ends) - np.asarray(starts)
        levels = self.logs[lengths]
        right = np.asarray(ends) - np.left_shift(1, levels)
        lo = np.minimum(self.minimum[levels, starts], self.minimum[levels, right])
        hi = np.maximum(self.maximum[levels, starts], self.maximum[levels, right])
        return (lengths / self.n) * (hi - lo)

    def error(self, start, end):
        """Rectangle cost C(start, end), excluding the complexity penalty."""
        start = _positive_int(start, "start", 0)
        end = _positive_int(end, "end")
        if not 0 <= start < end <= self.n:
            raise ValueError("expected 0 <= start < end <= len(signal)")
        return float(self._areas(start, end))

    def validate_breakpoints(self, breakpoints, min_size=1):
        min_size = _positive_int(min_size, "min_size")
        points = list(breakpoints)
        if not points:
            raise ValueError("breakpoints must include the final endpoint n")
        points = [_positive_int(p, "breakpoint") for p in points]
        if points[-1] != self.n or np.any(np.diff([0, *points]) < min_size):
            raise ValueError("breakpoints must increase, end at n, and respect min_size")
        return points

    def evaluate(self, breakpoints, penalty=0.02, min_size=1):
        """Return J(B) for a complete segmentation, including endpoint n."""
        penalty = _validate_penalty(penalty)
        points = self.validate_breakpoints(breakpoints, min_size)
        ends = np.asarray(points)
        area = float(np.sum(self._areas(np.r_[0, ends[:-1]], ends)))
        return area + penalty * (len(points) - 1)


def _validate_penalty(penalty):
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be a finite nonnegative number")
    return float(penalty)


def _prepare(signal, penalty, min_size):
    cost = signal if isinstance(signal, RectangleCost) else RectangleCost(signal)
    penalty = _validate_penalty(penalty)
    min_size = _positive_int(min_size, "min_size")
    if min_size > cost.n:
        raise ValueError("min_size cannot exceed len(signal)")
    return cost, penalty, min_size


def _result(cost, points, penalty, min_size, method, started, history=(), **diagnostics):
    points = cost.validate_breakpoints(points, min_size)
    ends = np.asarray(points)
    starts = np.r_[0, ends[:-1]]
    area = float(np.sum(cost._areas(starts, ends)))
    rectangles = [
        SegmentRectangle(int(s), int(e), float(np.min(cost.signal[s:e])), float(np.max(cost.signal[s:e])))
        for s, e in zip(starts, ends)
    ]
    return RectangleSegmentation(
        method, points, rectangles, area, penalty,
        area + penalty * (len(points) - 1), perf_counter() - started, min_size,
        np.asarray(history, dtype=float), diagnostics,
    )


def segment_pelt(signal, penalty=0.02, min_size=10):
    """Find the global minimum of normalized rectangle area + penalty*k.

    All sample positions are considered (no ``jump`` subsampling). Rectangle
    area is superadditive: C(a,b)+C(b,c) <= C(a,c), so PELT uses K=0.
    A candidate dominated at t only expires at t+min_size: before then the
    replacement segment starting at t would violate the length constraint.
    Time can be O(n²) in the worst case; linear complexity is not guaranteed.
    Equal objective values prefer fewer segments. Floating-point pruning
    uses a conservative tolerance and never removes numerical ties.
    """
    started = perf_counter()
    cost, penalty, min_size = _prepare(signal, penalty, min_size)
    n = cost.n
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    previous = np.full(n + 1, -1, dtype=int)
    counts = np.zeros(n + 1, dtype=int)
    expires = np.full(n + 1, n + min_size + 1, dtype=int)
    active = np.empty(0, dtype=int)
    evaluations = 0
    largest_set = 0
    for end in range(min_size, n + 1):
        active = active[expires[active] > end]
        new_start = end - min_size
        if np.isfinite(best[new_start]):
            active = np.append(active, new_start)
        unpenalized = best[active] + cost._areas(active, end)
        values = unpenalized + penalty
        winner = np.lexsort((active, counts[active], values))[0]
        start = active[winner]
        best[end], previous[end] = values[winner], start
        counts[end] = counts[start] + 1
        tolerance = 64 * np.finfo(float).eps * max(1.0, abs(best[end]))
        dominated = active[unpenalized > best[end] + tolerance]
        expires[dominated] = np.minimum(expires[dominated], end + min_size)
        evaluations += len(active)
        largest_set = max(largest_set, len(active))
    points = []
    end = n
    while end:
        points.append(int(end))
        end = previous[end]
    return _result(
        cost, points[::-1], penalty, min_size, "PELT", started,
        cost_evaluations=evaluations, max_candidates=largest_set, exact=True,
    )


def _repair(points, n, min_size, rng):
    """Keep a random maximal compatible subset, avoiding left-edge bias."""
    kept = [0, n]
    for point in rng.permutation(sorted(set(points))):
        point = int(point)
        if not 0 < point < n:
            continue
        i = bisect_left(kept, point)
        if point - kept[i - 1] >= min_size and kept[i] - point >= min_size:
            kept.insert(i, point)
    return tuple(kept[1:-1])


def _random_genome(n, min_size, max_initial_breaks, rng):
    k = int(rng.integers(min(max_initial_breaks, n // min_size - 1) + 1))
    # Uniform positive compositions, then add min_size-1 to each part.
    compressed_n = n - (k + 1) * (min_size - 1)
    separators = np.sort(rng.choice(compressed_n - 1, size=k, replace=False) + 1)
    return tuple((separators + np.arange(1, k + 1) * (min_size - 1)).tolist())


def segment_genetic(
    signal, penalty=0.02, min_size=10, *, population_size=50,
    n_restarts=8, max_iterations=10000, patience=800,
    max_initial_breaks=20, p_uniform=0.3, p_one_point=0.3,
    p_insert=0.6, seed=42, champions=True,
):
    """Stochastic rectangle-cover search using the paper's GA framework.

    A sparse genome contains internal break indices. Each steady-state round
    performs ONE operator: uniform crossover, one-point crossover, or mutation.
    Roulette-wheel selection uses positive fitness 1/(1+J). A child replaces
    the worst member; the best-ever solution is retained separately. Uniform
    crossover selects either parent's allele with equal probability.

    Mutation samples loci with p_m=min(1, 2*max(k,1)/n), then sets each sampled
    locus to a break with probability p_insert (0.6 in Table 3), or deletes it.
    Binomial sampling avoids scanning the full chromosome. max(k,1) keeps the
    no-break solution mutable. Incompatible boundaries are repaired randomly.

    After independent restarts, their champions form a final population.
    The default budget is smaller than the paper's 100 populations; set
    n_restarts=100 for that budget. This is a heuristic, with no optimality
    guarantee and no PELT warm start or deterministic local polishing.
    """
    started = perf_counter()
    cost, penalty, min_size = _prepare(signal, penalty, min_size)
    population_size = _positive_int(population_size, "population_size", 2)
    n_restarts = _positive_int(n_restarts, "n_restarts")
    max_iterations = _positive_int(max_iterations, "max_iterations", 0)
    patience = _positive_int(patience, "patience")
    max_initial_breaks = _positive_int(max_initial_breaks, "max_initial_breaks", 0)
    for name, p in [("p_uniform", p_uniform), ("p_one_point", p_one_point), ("p_insert", p_insert)]:
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if p_uniform + p_one_point > 1:
        raise ValueError("p_uniform + p_one_point cannot exceed 1")
    rng = np.random.default_rng(seed)
    n = cost.n
    evaluation_count = 0
    histories = []
    winners = []
    winner_scores = []
    iteration_counts = []

    def evaluate(genome):
        nonlocal evaluation_count
        evaluation_count += 1
        ends = np.asarray((*genome, n), dtype=int)
        return float(np.sum(cost._areas(np.r_[0, ends[:-1]], ends))) + penalty * len(genome)

    def evolve(population):
        population = list(population)
        scores = np.array([evaluate(g) for g in population])
        best_index = int(np.argmin(scores))
        best_genome, best_score = population[best_index], float(scores[best_index])
        history = [best_score]
        stale = 0
        for _ in range(max_iterations):
            weights = 1.0 / (1.0 + scores)
            parents = rng.choice(len(population), size=2, replace=False, p=weights / weights.sum())
            a, b = (population[i] for i in parents)
            operator = rng.random()
            if operator < p_uniform:
                shared = set(a).intersection(b)
                optional = sorted(set(a).symmetric_difference(b))
                child = shared.union(p for p in optional if rng.random() < 0.5)
            elif operator < p_uniform + p_one_point:
                cut = int(rng.integers(1, n)) if n > 1 else 1
                child = [p for p in a if p < cut] + [p for p in b if p >= cut]
            else:
                child = set(a)
                pm = min(1.0, 2 * max(len(a), 1) / n)
                n_loci = int(rng.binomial(n - 1, pm))
                for point in rng.choice(n - 1, size=n_loci, replace=False) + 1:
                    if rng.random() < p_insert:
                        child.add(int(point))
                    else:
                        child.discard(int(point))
            child = _repair(child, n, min_size, rng)
            score = evaluate(child)
            worst = int(np.argmax(scores))
            population[worst], scores[worst] = child, score
            if (score, len(child), child) < (best_score, len(best_genome), best_genome):
                improved = score < best_score
                best_genome, best_score = child, score
                stale = 0 if improved else stale + 1
            else:
                stale += 1
            history.append(best_score)
            if stale >= patience:
                break
        iteration_counts.append(len(history) - 1)
        histories.append(np.asarray(history))
        return best_genome, best_score

    for _ in range(n_restarts):
        population = [_random_genome(n, min_size, max_initial_breaks, rng) for _ in range(population_size)]
        population[0] = ()
        genome, score = evolve(population)
        winners.append(genome)
        winner_scores.append(score)
    index = int(np.argmin(winner_scores))
    best_genome, best_score = winners[index], winner_scores[index]
    champions_history = np.empty(0)
    if champions and n_restarts > 1:
        genome, score = evolve(winners)
        champions_history = histories[-1]
        if (score, len(genome), genome) < (best_score, len(best_genome), best_genome):
            best_genome, best_score = genome, score
    # Running global incumbent, including all restarts and the champions phase.
    history = np.minimum.accumulate(np.concatenate(histories))
    return _result(
        cost, [*best_genome, n], penalty, min_size, "GA", started, history,
        seed=seed, exact=False, fitness_evaluations=evaluation_count,
        restart_objectives=winner_scores, restart_breakpoints=[[*g, n] for g in winners],
        iterations=iteration_counts, champions_history=champions_history,
        population_size=population_size, n_restarts=n_restarts,
    )


