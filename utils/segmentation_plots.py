"""Визуализация и сохранение результатов полиномиальной сегментации."""
from __future__ import annotations
import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from src.meg_segmentation import segment_and_fit

matplotlib.rcParams.update(
    {
        "figure.figsize": (12, 3),
        "font.size": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
    }
)


def plot_series_segmentation(
    series: dict,
    result: dict,
    save_path: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot raw signal, piecewise polynomial fit, and breakpoints.

    Parameters
    ----------
    series:
        Dict returned by :func:`select_series` for a single (epoch, channel).
    result:
        Dict returned by :func:`segment_and_fit`.
    save_path:
        If provided, save the figure to this path (PNG).
    ax:
        If provided, draw into this existing Axes.  Otherwise a new figure is
        created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    signal = series["signal"]
    times = series["times"]
    fit = result["fit"]
    breakpoints = result["breakpoints"]
    params = result["params"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.get_figure()

    # Raw signal — thin gray
    ax.plot(times, signal, color="gray", linewidth=0.8, alpha=0.8, label="raw")

    # Piecewise polynomial — thick colored line
    ax.plot(times, fit, color="C0", linewidth=2.0, label="poly fit")

    # Breakpoints — dashed vertical lines (exclude the final sentinel)
    inner_bps = breakpoints[:-1]
    for bp in inner_bps:
        if bp < len(times):
            ax.axvline(times[bp], color="C3", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_xlabel("Time (s)")

    # Y-axis label: derive unit from channel type
    ch_name = series["channel_name"]
    if ch_name.startswith("MEG"):
        ax.set_ylabel("B (T)")
    elif ch_name.startswith("EEG"):
        ax.set_ylabel("V (V)")
    else:
        ax.set_ylabel("amplitude")

    mdj = params.get("min_deriv_jump")
    mdj_str = f", min_deriv_jump={mdj}" if mdj is not None else ""
    title = (
        f"{series['label']}  |  "
        f"penalty={params['penalty']}, degree={params['degree']}{mdj_str}, "
        f"n_bp={len(inner_bps)}"
    )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_multiple(
    series_list: list[dict],
    results_list: list[dict],
    save_dir: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Grid of segmentation plots, one row per series.

    Parameters
    ----------
    series_list:
        List of series dicts from :func:`select_series`.
    results_list:
        Corresponding list of result dicts from :func:`segment_and_fit`.
    save_dir:
        Directory where figures are saved.  Individual series are saved as
        ``series_<i>_ep<epoch>_<channel>.png``; the combined grid is saved as
        ``grid.png``.

    Returns
    -------
    matplotlib.figure.Figure
        The combined grid figure.
    """
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Consistent X range across all rows
    all_times = [s["times"] for s in series_list]
    xmin = min(t[0] for t in all_times)
    xmax = max(t[-1] for t in all_times)

    for i, (series, result, ax) in enumerate(zip(series_list, results_list, axes)):
        plot_series_segmentation(series, result, ax=ax)
        ax.set_xlim(xmin, xmax)

        if save_dir is not None:
            ch_safe = series["channel_name"].replace(" ", "")
            fname = f"series_{i}_ep{series['epoch_idx']}_{ch_safe}.png"
            fpath = os.path.join(save_dir, fname)
            os.makedirs(save_dir, exist_ok=True)
            # Save individual panel as its own figure
            ind_fig, ind_ax = plt.subplots(figsize=(12, 3))
            plot_series_segmentation(series, result, ax=ind_ax)
            ind_ax.set_xlim(xmin, xmax)
            ind_fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(ind_fig)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        grid_path = os.path.join(save_dir, "grid.png")
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")

    return fig


def segment_and_plot_all(
    series_list: list[dict],
    save_dir: str,
    close: bool = True,
    **seg_params,
) -> list[dict]:
    """Segment every series and save one figure per channel.

    Convenience entry point for a whole file: runs :func:`segment_and_fit` on
    each series and saves ``<channel>.png`` into *save_dir* — one panel per
    channel, without building a single oversized combined grid (which is
    impractical for hundreds of channels).

    Parameters
    ----------
    series_list:
        Series dicts, e.g. from :func:`load_xlsx_series` or :func:`select_series`.
    save_dir:
        Directory to write the per-channel PNGs into (created if missing).
    close:
        Close each figure after saving to keep memory flat over many channels.
    **seg_params:
        Forwarded to :func:`segment_and_fit` (e.g. ``penalty``, ``degree``,
        ``min_deriv_jump``).

    Returns
    -------
    list[dict]
        Result dicts from :func:`segment_and_fit`, aligned with *series_list*.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    for s in series_list:
        result = segment_and_fit(s["signal"], **seg_params)
        fname = f"{s['channel_name'].replace(' ', '')}.png"
        fig = plot_series_segmentation(
            s, result, save_path=os.path.join(save_dir, fname)
        )
        if close:
            plt.close(fig)
        results.append(result)
    return results
