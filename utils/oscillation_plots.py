"""Raw boundaries, band RMS and spectral diagnostics; no implicit exports."""
import matplotlib.pyplot as plt
import numpy as np

from .rectangle_plots import plot_change_points


def plot_oscillations(series, result):
    """Four aligned axes in original time/units; return (figure, axes)."""
    d = result.diagnostics
    times = d["frame_times"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout="constrained")
    plot_change_points(series, result, ax=axes[0])
    axes[1].plot(times, d["band_rms"], color="tab:blue")
    axes[1].set(ylabel="RMS полосы\n(исходные единицы)")
    axes[2].plot(times, d["peak_hz"], label="Пик спектра", color="tab:purple")
    axes[2].plot(times, d["centroid_hz"], label="Центр мощности", color="tab:orange")
    axes[2].set(ylabel="Частота, Гц")
    axes[2].legend(loc="upper right")
    axes[3].pcolormesh(times, d["frequencies_hz"],
                       10*np.log10(np.maximum(d["relative_power"].T, 1e-6)),
                       shading="nearest", cmap="magma", vmin=-50, vmax=0)
    axes[3].set(ylabel="Частота, Гц", xlabel="Время, с",
                title="Форма спектра: доля мощности бина, дБ (−50…0)")
    raw_times = np.asarray(series["times"])
    for ax in axes[1:]:
        for point in result.change_points:
            ax.axvline(raw_times[point], color="tab:red", ls="--", lw=1)
        ax.grid(alpha=.15)
    axes[-1].set_xlim(raw_times[0], raw_times[-1])
    return fig, axes
