"""Границы, покрывающие прямоугольники и сходимость генетического поиска."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.rectangle_segmentation import _positive_int


def _plot_data(series, result):
    y = np.asarray(series["signal"], dtype=float)
    times = np.asarray(series.get("times", np.arange(len(y))), dtype=float)
    if y.ndim != 1 or len(y) != result.breakpoints[-1] or not np.isfinite(y).all():
        raise ValueError("series signal must be finite and match the result length")
    if times.shape != y.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite, strictly increasing, and match signal")
    # Each observation owns its sample cell. A break b is the LEFT edge of
    # cell b, consistent with the end-exclusive segmentation convention.
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    edges = np.r_[times, times[-1] + step]
    return y, times, edges


def plot_change_points(series, result, *, ax=None, true_breakpoints=None):
    """Plot raw data and internal breaks. Return (figure, axes); never show."""
    y, times, _ = _plot_data(series, result)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(times, y, color="0.25", lw=0.85, label="Сигнал")
    for i, point in enumerate(result.change_points):
        ax.axvline(times[point], color="tab:red", ls="--", lw=1.2,
                   label="Найденные сдвиги" if i == 0 else None)
    if true_breakpoints is not None:
        for i, point in enumerate(true_breakpoints):
            point = _positive_int(point, "true breakpoint")
            if point >= len(y):
                raise ValueError("true breakpoints must be internal indices")
            ax.axvline(times[point], color="tab:green", ls=":", lw=1.5,
                       label="Истинные сдвиги" if i == 0 else None)
    ax.set(title=f"{series.get('label', '')} — {result.method}: {len(result.change_points)} сдвигов",
           xlabel="Время, с" if "times" in series else "Индекс отсчёта",
           ylabel="Амплитуда (исходные единицы)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    return ax.figure, ax


def plot_rectangles(series, result, *, ax=None):
    """Plot enclosing rectangles in ORIGINAL amplitude and time units.

    Width spans sample cells [t_start, t_end), with one extrapolated final
    edge. A constant segment is drawn as a horizontal line (zero area).
    """
    y, times, edges = _plot_data(series, result)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    colors = plt.get_cmap("tab10")
    for i, r in enumerate(result.rectangles):
        color = colors(i % 10)
        patch = Rectangle((edges[r.start], r.lower), edges[r.end] - edges[r.start],
                          r.upper - r.lower, facecolor=color, edgecolor=color,
                          alpha=0.24, lw=1.5)
        ax.add_patch(patch)
        ax.hlines([r.lower, r.upper], edges[r.start], edges[r.end], color=color, lw=1.2)
        ax.vlines([edges[r.start], edges[r.end]], r.lower, r.upper, color=color, lw=1.2)
    ax.plot(times, y, color="0.2", lw=0.85, label="Сигнал")
    ax.set(title=f"{series.get('label', '')} — {result.method}: прямоугольники, J={result.objective:.4f}",
           xlabel="Время, с" if "times" in series else "Индекс отсчёта",
           ylabel="Амплитуда (исходные единицы)")
    ax.autoscale_view()
    ax.grid(alpha=0.2)
    return ax.figure, ax


def plot_comparison(series, results, *, true_breakpoints=None):
    """One row per optimizer: breaks at left, rectangle cover at right."""
    results = list(results)
    if not results:
        raise ValueError("results cannot be empty")
    fig, axes = plt.subplots(len(results), 2, figsize=(16, 3.5 * len(results)),
                             squeeze=False, layout="constrained")
    for row, result in enumerate(results):
        plot_change_points(series, result, ax=axes[row, 0], true_breakpoints=true_breakpoints)
        plot_rectangles(series, result, ax=axes[row, 1])
    return fig, axes


def plot_convergence(genetic_result, exact_result=None, *, ax=None):
    """Plot the best GA objective over rounds and an optional PELT bound."""
    if genetic_result.history.size == 0:
        raise ValueError("genetic_result must have a nonempty search history")
    if exact_result is not None and (
        exact_result.penalty != genetic_result.penalty
        or exact_result.min_size != genetic_result.min_size
        or exact_result.breakpoints[-1] != genetic_result.breakpoints[-1]
    ):
        raise ValueError("GA and PELT must use the same signal length, penalty and min_size")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(genetic_result.history, label="Лучшее J у GA", color="tab:blue")
    if exact_result is not None:
        ax.axhline(exact_result.objective, color="tab:orange", ls="--", label="Оптимум PELT")
    ax.set(xlabel="Итерации GA (все популяции)", ylabel="Целевая функция J",
           title="Сходимость генетического поиска")
    ax.legend()
    ax.grid(alpha=0.2)
    return ax.figure, ax
