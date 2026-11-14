"""Exploratory analysis helpers for eda.ipynb; no implicit downloads.

Signal loading is in utils/data_loading.py. Excel channel types are
inferred from Neuromag names; physical units and event-code meanings are NOT
inferred from MNE sample. Analysis never changes the source signals.
"""

from __future__ import annotations

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import openpyxl
import pandas as pd
from scipy import signal, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import uniform_filter1d


FEATURE_DESCRIPTIONS = {
    "n_samples": "Число отсчётов, включая нечисловые значения",
    "n_finite": "Число конечных значений; Inf не включаются",
    "missing_fraction": "Доля NaN и Inf; они не интерполируются",
    "mean": "Среднее, исходные единицы",
    "median": "Медиана, исходные единицы",
    "std": "Выборочное стандартное отклонение (ddof=1)",
    "rms": "Среднеквадратичная амплитуда относительно нуля",
    "min": "Минимум",
    "max": "Максимум",
    "ptp": "Размах max−min",
    "q01": "Квантиль 1%",
    "q05": "Квантиль 5%",
    "q25": "Квантиль 25%",
    "q75": "Квантиль 75%",
    "q95": "Квантиль 95%",
    "q99": "Квантиль 99%",
    "iqr": "Межквартильный размах q75−q25",
    "mad": "Медиана |y−median(y)|, без множителя",
    "skewness": "Асимметрия: третий стандартизованный момент",
    "excess_kurtosis": "Эксцесс: четвёртый стандартизованный момент минус 3",
    "robust_outlier_fraction": "Доля |y−median| > 6×1.4826×MAD; эвристика, не диагноз",
    "acf_lag1": "Корреляция соседних отсчётов; только для полного ряда",
    "roughness": "std(diff(y), ddof=1) / std(y, ddof=1)",
    "zero_crossing_fraction": "Доля смен знака относительно среднего",
    "flat_step_fraction": "Доля точно равных соседних отсчётов",
    "linear_slope_per_s": "Наклон линейной регрессии по времени, исходные единицы/с",
    "half_mean_shift_sd": "Разность средних второй и первой половин / std всего ряда",
    "half_std_log2_ratio": "log2(std второй половины / std первой половины)",
    "constant": "Все конечные значения равны; динамические признаки не определены",
    "psd_peak_hz": "Частота максимума Welch PSD в диапазоне 1–100 Гц (в пределах Nyquist)",
    "spectral_centroid_hz": "Центр масс мощности PSD в диапазоне 1–100 Гц",
    "spectral_entropy": "Энтропия дискретного спектра 1–100 Гц / log числа бинов, 0…1",
    "delta_1_4_fraction": "Доля мощности 1–4 Гц в мощности 1–45 Гц",
    "theta_4_8_fraction": "Доля мощности 4–8 Гц в мощности 1–45 Гц",
    "alpha_8_13_fraction": "Доля мощности 8–13 Гц в мощности 1–45 Гц",
    "beta_13_30_fraction": "Доля мощности 13–30 Гц в мощности 1–45 Гц",
    "gamma_30_45_fraction": "Доля мощности 30–45 Гц в мощности 1–45 Гц",
    "line50_fraction": "Доля мощности 48–52 Гц в мощности 1–100 Гц; не доказательство помехи",
    "high80_fraction": "Доля мощности 80–100 Гц в мощности 1–100 Гц",
}


def infer_channels(names, *, source, verified_types=None, bads=()):
    """Channel inventory; Excel naming is only a type hypothesis."""
    rows = []
    for i, name in enumerate(names):
        match = re.fullmatch(r"MEG\s*(\d{3})([123])", name)
        inferred = {"1": "mag", "2": "grad", "3": "grad"}.get(match[2]) if match else None
        kind = verified_types[i] if verified_types is not None else (inferred or "unknown")
        unit = {"mag": "T", "grad": "T/m", "eeg": "V", "eog": "V"}.get(kind, "unknown")
        rows.append(dict(
            source=source, channel_name=name, channel_type=kind,
            type_evidence="MNE Info" if verified_types is not None else "name only (unverified)",
            unit=unit if verified_types is not None else "export units (unverified)",
            site=match[1] if match else "", orientation=match[2] if match else "",
            marked_bad=name in bads,
        ))
    return pd.DataFrame(rows)


def audit_excel(path):
    """Inspect rows/time and retain stimulus pulses skipped by the loader."""
    path = Path(path)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows, time_values, stim_rows = [], None, []
    for row in sheet.iter_rows(values_only=True):
        if row[0] is None:
            continue
        name = str(row[0])
        values = np.array([np.nan if v is None else float(v) for v in row[1:]])
        kind = "time" if name == "Time" else ("stimulus" if name.startswith("STI") else "signal")
        rows.append(dict(row_name=name, row_type=kind, n_columns=len(values),
                         n_nonfinite=int((~np.isfinite(values)).sum())))
        if kind == "time":
            time_values = values
        if kind == "stimulus":
            stim_rows.append((name, values))
    sheet_name = sheet.title
    workbook.close()
    if time_values is None:
        raise ValueError("Excel has no Time row")
    finite = np.flatnonzero(np.isfinite(time_values))
    if not finite.size:
        raise ValueError("Time row has no finite samples")
    times = time_values[:finite[-1] + 1]
    if not np.isfinite(times).all() or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Time must be finite and strictly increasing")
    dt = np.diff(times)
    sfreq = 1 / np.median(dt)
    pulses = []
    for name, values in stim_rows:
        values = values[:len(times)]
        edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for start, end in zip(edges[:-1], edges[1:]):
            code = values[start]
            if np.isfinite(code) and code != 0:
                pulses.append(dict(marker_channel=name, sample=int(start), time_s=float(times[start]),
                                   code=float(code), duration_s=float((end - start) / sfreq)))
    pulse_table = pd.DataFrame(pulses, columns=["marker_channel", "sample", "time_s", "code", "duration_s"])
    summary = dict(file=path.name, sheet=sheet_name, samples=len(times), sfreq_hz=float(sfreq),
                   t_start_s=float(times[0]), t_end_s=float(times[-1]),
                   max_dt_error_s=float(np.max(np.abs(dt - np.median(dt)))),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(summary=summary, rows=pd.DataFrame(rows), markers=pulse_table, times=times)


def series_arrays(series, source):
    """Prepare an existing load_xlsx_series/select_series result for analysis."""
    if not series:
        raise ValueError("series cannot be empty")
    times = np.asarray(series[0]["times"], dtype=float)
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("At least three finite increasing times are required")
    for s in series:
        if not np.array_equal(s["times"], times) or np.shape(s["signal"]) != times.shape:
            raise ValueError("All series must share the same time grid")
    metadata = infer_channels([s["channel_name"] for s in series], source=source)
    metadata["epoch_idx"] = [s["epoch_idx"] for s in series]
    metadata["condition"] = "not documented"
    metadata["series_id"] = [f"{source}/ep{s['epoch_idx']}/{s['channel_name']}" for s in series]
    return np.asarray([s["signal"] for s in series], dtype=float), times, metadata


def _safe_divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)


def row_statistics(data, times, metadata, batch_size=512):
    """Statistics for EVERY row, computed in bounded-memory batches.

    Marginal statistics ignore nonfinite values and report their fraction.
    Time-order statistics require a complete row; gaps are never interpolated.
    Moments are computed after scaling to avoid MEG-unit underflow.
    """
    data = np.asarray(data, dtype=float)
    times = np.asarray(times, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(times) or data.shape[0] != len(metadata):
        raise ValueError("Expected data=(series, time) and matching metadata/times")
    if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be finite, increasing, with at least three samples")
    frames = []
    for first in range(0, len(data), batch_size):
        y = data[first:first + batch_size]
        finite = np.isfinite(y)
        y = np.where(finite, y, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmax(np.abs(y), axis=1)
            scale[~np.isfinite(scale) | (scale == 0)] = 1
            z = y / scale[:, None]
            quantiles = np.nanquantile(z, [.01, .05, .25, .5, .75, .95, .99], axis=1)
            mean = np.nanmean(z, axis=1)
            sd = np.nanstd(z, axis=1, ddof=1)
            centered = z - mean[:, None]
            moment2 = np.nanmean(centered ** 2, axis=1)
            mad = np.nanmedian(np.abs(z - quantiles[3, :, None]), axis=1)
            complete = finite.all(axis=1)
            result = dict(
                n_samples=np.full(len(y), y.shape[1]), n_finite=finite.sum(axis=1),
                missing_fraction=1 - finite.mean(axis=1), mean=mean * scale,
                median=quantiles[3] * scale, std=sd * scale,
                rms=np.sqrt(np.nanmean(z ** 2, axis=1)) * scale,
                min=np.nanmin(z, axis=1) * scale, max=np.nanmax(z, axis=1) * scale,
                ptp=(np.nanmax(z, axis=1) - np.nanmin(z, axis=1)) * scale,
                iqr=(quantiles[4] - quantiles[2]) * scale, mad=mad * scale,
                skewness=_safe_divide(np.nanmean(centered ** 3, axis=1), moment2 ** 1.5),
                excess_kurtosis=_safe_divide(np.nanmean(centered ** 4, axis=1), moment2 ** 2) - 3,
                robust_outlier_fraction=np.where(
                    mad > 0, _safe_divide(np.sum(np.abs(z - quantiles[3, :, None]) > 6 * 1.4826 * mad[:, None], axis=1), finite.sum(axis=1)), np.nan
                ),
                constant=(np.nanmax(z, axis=1) == np.nanmin(z, axis=1)),
            )
            for name, values in zip(["q01", "q05", "q25", "median", "q75", "q95", "q99"], quantiles):
                result[name] = values * scale
            left, right = z[:, :-1], z[:, 1:]
            left = left - np.mean(left, axis=1, keepdims=True)
            right = right - np.mean(right, axis=1, keepdims=True)
            acf = _safe_divide(np.sum(left * right, axis=1), np.sqrt(np.sum(left ** 2, axis=1) * np.sum(right ** 2, axis=1)))
            half = y.shape[1] // 2
            tc = times - times.mean()
            sd_first = np.std(z[:, :half], axis=1, ddof=1)
            sd_second = np.std(z[:, half:], axis=1, ddof=1)
            ordered = dict(
                acf_lag1=acf,
                roughness=_safe_divide(np.std(np.diff(z, axis=1), axis=1, ddof=1), sd),
                zero_crossing_fraction=np.mean(centered[:, :-1] * centered[:, 1:] < 0, axis=1),
                flat_step_fraction=np.mean(np.diff(z, axis=1) == 0, axis=1),
                linear_slope_per_s=np.sum(z * tc, axis=1) / np.sum(tc ** 2) * scale,
                half_mean_shift_sd=_safe_divide(np.mean(z[:, half:], axis=1) - np.mean(z[:, :half], axis=1), sd),
                half_std_log2_ratio=np.log2(_safe_divide(sd_second, sd_first)),
            )
            for name, values in ordered.items():
                result[name] = np.where(complete, values, np.nan)
        frames.append(pd.DataFrame(result))
    return pd.concat([metadata.reset_index(drop=True), pd.concat(frames, ignore_index=True)], axis=1)


def spectral_statistics(data, sfreq, *, nperseg=512, batch_size=512, keep_psd=True):
    """Welch features per series; never concatenate separate trials.

    Relative band powers are sums of equally spaced PSD bins times df.
    Band intervals are [low, high); 45 Hz is excluded from the denominator.
    No zero-padding increases the apparent frequency resolution.
    """
    data = np.asarray(data, dtype=float)
    nperseg = min(nperseg, data.shape[1])
    frequencies = np.fft.rfftfreq(nperseg, d=1 / sfreq)
    df = sfreq / nperseg
    bands = {"delta_1_4": (1, 4), "theta_4_8": (4, 8), "alpha_8_13": (8, 13),
             "beta_13_30": (13, 30), "gamma_30_45": (30, 45)}
    features, spectra = [], []
    selected = (frequencies >= 1) & (frequencies < min(100, sfreq / 2) + 1e-9)
    low_total = (frequencies >= 1) & (frequencies < 45)
    for first in range(0, len(data), batch_size):
        batch = data[first:first + batch_size]
        complete = np.isfinite(batch).all(axis=1)
        _, psd = signal.welch(batch, sfreq, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend="constant", axis=1)
        psd[~complete] = np.nan
        power = psd[:, selected]
        total = np.sum(power, axis=1)
        relative = _safe_divide(power, total[:, None])
        safe = np.where(relative > 0, relative, 1)
        feature = dict(
            psd_peak_hz=np.where(total > 0, frequencies[selected][np.argmax(np.nan_to_num(power), axis=1)], np.nan),
            spectral_centroid_hz=np.sum(relative * frequencies[selected], axis=1),
            spectral_entropy=-np.sum(relative * np.log(safe), axis=1) / np.log(max(2, power.shape[1])),
        )
        denominator = np.sum(psd[:, low_total], axis=1)
        for name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            feature[f"{name}_fraction"] = _safe_divide(np.sum(psd[:, mask], axis=1), denominator)
        for name, low, high in [("line50", 48, 52), ("high80", 80, 100.00001)]:
            mask = (frequencies >= low) & (frequencies < high)
            feature[f"{name}_fraction"] = _safe_divide(np.sum(psd[:, mask], axis=1), total)
        features.append(pd.DataFrame(feature))
        if keep_psd:
            spectra.append(psd)
    return dict(features=pd.concat(features, ignore_index=True),
                frequencies=frequencies, psd=np.concatenate(spectra) if keep_psd else None,
                nperseg=nperseg, frequency_resolution_hz=df)


def analyze_series(data, times, metadata, *, keep_psd=True):
    """Shared entry point for Excel rows or flattened MNE trial×channel rows."""
    sfreq = 1 / np.median(np.diff(times))
    if not np.allclose(np.diff(times), 1 / sfreq, rtol=1e-5, atol=1e-12):
        raise ValueError("Welch requires an approximately uniform sampling grid")
    table = row_statistics(data, times, metadata)
    spectral = spectral_statistics(data, sfreq, keep_psd=keep_psd)
    table = pd.concat([table, spectral.pop("features")], axis=1)
    return dict(data=data, times=times, metadata=metadata, statistics=table, sfreq=sfreq, **spectral)


def scroll_table(table, *, max_height=440, scientific=False):
    """Show all rows in scrollable HTML instead of pandas' truncated repr."""
    from IPython.display import HTML
    fmt = (lambda x: f"{x:.5e}") if scientific else (lambda x: f"{x:.5g}")
    html = table.to_html(index=False, float_format=fmt, na_rep="—", escape=True)
    return HTML(f'<div style="max-height:{max_height}px;overflow:auto;font-size:12px">{html}</div>')


def _markers(ax, markers):
    if markers is None:
        return
    for event in markers.itertuples():
        ax.axvline(event.time_s, color="#a23c45", ls="--", lw=1, alpha=.8)


def _standardize(data):
    centered = data - np.mean(data, axis=1, keepdims=True)
    sd = np.std(centered, axis=1, keepdims=True)
    return np.divide(centered, sd, out=np.zeros_like(centered), where=sd > 0)


def representative_indices(table, count=6):
    """Deterministic examples, chosen by documented statistics, not by eye."""
    order = table["std"].sort_values().index.to_numpy()
    choices = [order[0], order[len(order) // 2], order[-1], table["roughness"].idxmax(),
               table["robust_outlier_fraction"].idxmax(), table["half_std_log2_ratio"].abs().idxmax()]
    chosen = list(dict.fromkeys(int(i) for i in choices if pd.notna(i)))
    chosen.extend(int(i) for i in order if i not in chosen)
    return chosen[:min(count, len(table))]


def plot_all_series(analysis, markers=None):
    """Every channel, in source order; native amplitude and row-wise z scores."""
    data, times = analysis["data"], analysis["times"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    for ax, values, title, label in zip(axes, [data, _standardize(data)],
            ["Все ряды: исходные амплитуды", "Все ряды: форма после отдельной z-нормировки"],
            ["Исходные единицы экспорта", "Стандартных отклонений"]):
        limit = np.nanquantile(np.abs(values), .995)
        im = ax.imshow(values, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit,
                       extent=[times[0], times[-1], len(data) - .5, -.5], rasterized=True)
        positions = np.unique(np.linspace(0, len(data) - 1, min(18, len(data))).astype(int))
        ax.set_yticks(positions, analysis["metadata"].iloc[positions]["channel_name"], fontsize=8)
        ax.set(title=title, xlabel="Время, с", ylabel="Канал (порядок Excel)")
        fig.colorbar(im, ax=ax, label=label, pad=.01)
        _markers(ax, markers)
    return fig


def plot_feature_distributions(table):
    columns = ["std", "acf_lag1", "roughness", "skewness", "excess_kurtosis",
               "robust_outlier_fraction", "half_std_log2_ratio", "spectral_centroid_hz"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    for ax, name in zip(axes.flat, columns):
        for label, group in table.groupby("orientation", sort=True):
            values = group[name].dropna()
            if name == "std":
                values = np.log10(values[values > 0])
            ax.hist(values, bins=18, alpha=.55, label=f"Ориентация {label}", density=True)
        ax.set(title=("log10(std), исх. ед." if name == "std" else name), ylabel="Плотность")
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    return fig


def plot_representatives(analysis, indices, markers=None):
    fig, axes = plt.subplots(len(indices), 1, figsize=(14, 2.0 * len(indices)),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, i in zip(axes.flat, indices):
        row = analysis["statistics"].iloc[i]
        ax.plot(analysis["times"], analysis["data"][i], lw=.75, color="#245a81")
        ax.set(title=f"{row.channel_name} | std={row['std']:.2e}; roughness={row.roughness:.3f}; "
                     f"эксцесс={row.excess_kurtosis:.2f}", ylabel="Исх. ед.")
        ax.grid(alpha=.15)
        _markers(ax, markers)
    axes[-1, 0].set_xlabel("Время, с")
    return fig


def plot_marginal_shapes(analysis, indices):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for col, i in enumerate(indices[:3]):
        values = _standardize(analysis["data"][[i]])[0]
        ax = axes[0, col]
        ax.hist(values, bins=35, density=True, color="#428ca0", alpha=.7)
        x = np.linspace(-4, 4, 150)
        ax.plot(x, stats.norm.pdf(x), color="#b8504a", label="N(0,1)")
        ax.set(title=analysis["metadata"].iloc[i].channel_name, xlabel="Амплитуда, z", ylabel="Плотность")
        ax.legend(fontsize=8)
        stats.probplot(values, dist="norm", plot=axes[1, col])
        axes[1, col].set(title="Q–Q: форма распределения", xlabel="Квантили N(0,1)", ylabel="Квантили ряда, z")
    return fig


def plot_spectra(analysis):
    f, psd = analysis["frequencies"], analysis["psd"]
    selected = (f >= 1) & (f <= 100)
    shape = _safe_divide(psd[:, selected], np.sum(psd[:, selected], axis=1, keepdims=True))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    for orientation, group in analysis["metadata"].groupby("orientation"):
        lo, median, hi = np.nanquantile(psd[group.index][:, selected], [.25, .5, .75], axis=0)
        line, = axes[0, 0].semilogy(f[selected], median, label=f"Ориентация {orientation}")
        axes[0, 0].fill_between(f[selected], lo, hi, color=line.get_color(), alpha=.15)
    axes[0, 0].set(title="Welch PSD: медиана и межквартильный диапазон по каналам",
                   xlabel="Частота, Гц", ylabel="Исх. ед.²/Гц")
    axes[0, 0].legend()
    im = axes[0, 1].imshow(10 * np.log10(np.maximum(shape, 1e-12)), aspect="auto", cmap="magma",
                           extent=[f[selected][0], f[selected][-1], len(psd) - .5, -.5])
    axes[0, 1].set(title="Спектральная форма всех рядов", xlabel="Частота, Гц", ylabel="Индекс канала")
    fig.colorbar(im, ax=axes[0, 1], label="10 log10 доли мощности бина")
    band_cols = [c for c in FEATURE_DESCRIPTIONS if c.endswith("_fraction") and c.split("_")[0] in ["delta", "theta", "alpha", "beta", "gamma"]]
    band_data = analysis["statistics"][band_cols].to_numpy()
    im = axes[1, 0].imshow(band_data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[1, 0].set_xticks(np.arange(len(band_cols)), [c.rsplit("_", 1)[0] for c in band_cols], rotation=25)
    axes[1, 0].set(title="Распределение мощности 1–45 Гц: каждый ряд", ylabel="Индекс канала")
    fig.colorbar(im, ax=axes[1, 0], label="Доля мощности")
    axes[1, 1].hist(analysis["statistics"]["line50_fraction"], bins=22, color="#b17948", alpha=.8)
    axes[1, 1].set(title="Доля мощности около 50 Гц (48–52 / 1–100)", xlabel="Доля", ylabel="Число каналов")
    for ax in axes.flat:
        ax.grid(alpha=.15)
    return fig


def temporal_profiles(analysis, window_seconds=.101, stride=10):
    n = analysis["data"].shape[1]
    window = max(3, min(n, round(window_seconds * analysis["sfreq"])))
    window = window if window % 2 else window - 1
    z = _standardize(analysis["data"])
    trend = uniform_filter1d(z, window, axis=1)
    variance = np.maximum(uniform_filter1d(z ** 2, window, axis=1) - trend ** 2, 0)
    indices = np.arange(window // 2, n - window // 2, stride)
    return dict(times=analysis["times"][indices], trend=trend[:, indices], std=np.sqrt(variance[:, indices]),
                window_samples=window, window_seconds=window / analysis["sfreq"])


def plot_temporal_profiles(analysis, profiles, markers=None):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), layout="constrained")
    for ax, values, title, cmap in zip(axes[:2], [profiles["trend"], profiles["std"]],
            ["Локальный уровень: среднее в окне, z", "Локальный разброс: std в окне / std всего канала"], ["RdBu_r", "viridis"]):
        limit = np.quantile(np.abs(values), .99)
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-limit if cmap == "RdBu_r" else 0, vmax=limit,
                       extent=[profiles["times"][0], profiles["times"][-1], len(values) - .5, -.5])
        ax.set(title=title, xlabel="Время, с", ylabel="Индекс канала")
        fig.colorbar(im, ax=ax)
        _markers(ax, markers)
    energy = np.sqrt(np.mean(_standardize(analysis["data"]) ** 2, axis=0))
    axes[2].plot(analysis["times"], energy, label="RMS по отдельно нормированным каналам", color="#245a81")
    axes[2].plot(profiles["times"], np.median(profiles["std"], axis=0), label="Медиана локального std", color="#ce8840")
    axes[2].set(xlabel="Время, с", ylabel="Безразмерная величина", title="Общая временная динамика")
    axes[2].legend()
    axes[2].grid(alpha=.15)
    _markers(axes[2], markers)
    return fig


def dependence_analysis(analysis):
    """Correlation/PCA within a common channel type; no source localization."""
    if analysis["metadata"]["channel_type"].nunique() != 1:
        raise ValueError("Analyze correlation/PCA separately for each channel type")
    z = _standardize(analysis["data"])
    correlation = np.clip(z @ z.T / z.shape[1], -1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues, eigenvectors = np.maximum(eigenvalues[::-1], 0), eigenvectors[:, ::-1]
    ratio = eigenvalues / eigenvalues.sum()
    pairs = []
    channels = analysis["metadata"]
    for site, group in channels.groupby("site"):
        orientations = dict(zip(group.orientation, group.index))
        if site and "2" in orientations and "3" in orientations:
            a, b = orientations["2"], orientations["3"]
            pairs.append(dict(site=site, channel_2=channels.loc[a, "channel_name"], channel_3=channels.loc[b, "channel_name"],
                              correlation=correlation[a, b], std_2=analysis["statistics"].loc[a, "std"],
                              std_3=analysis["statistics"].loc[b, "std"]))
    upper = np.triu_indices(len(z), 1)
    all_pairs = pd.DataFrame(dict(
        channel_a=channels.iloc[upper[0]].channel_name.to_numpy(),
        channel_b=channels.iloc[upper[1]].channel_name.to_numpy(),
        same_site=channels.iloc[upper[0]].site.to_numpy() == channels.iloc[upper[1]].site.to_numpy(),
        correlation=correlation[upper],
    ))
    return dict(correlation=correlation, variance_ratio=ratio, pca_timecourses=eigenvectors[:, :4].T @ z,
                n_components_90=int(np.searchsorted(np.cumsum(ratio), .9) + 1),
                effective_rank=float(1 / np.sum(ratio ** 2)), pairs=pd.DataFrame(pairs), all_pairs=all_pairs)


def plot_dependence(analysis, dependence):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    im = axes[0, 0].imshow(dependence["correlation"], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set(title="Корреляции всех каналов (вся запись)", xlabel="Индекс канала", ylabel="Индекс канала")
    fig.colorbar(im, ax=axes[0, 0], label="Pearson r")
    ratio = dependence["variance_ratio"]
    axes[0, 1].plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio))
    axes[0, 1].axhline(.9, color="#b8504a", ls="--")
    axes[0, 1].set(title=f"PCA стандартизованных каналов: 90% — {dependence['n_components_90']} компонент",
                   xlabel="Количество компонент", ylabel="Накопленная доля дисперсии", ylim=(0, 1.02))
    pairs = dependence["pairs"]
    axes[1, 0].scatter(pairs.std_2, pairs.std_3, s=22, alpha=.7, color="#418a92")
    limits = [min(pairs.std_2.min(), pairs.std_3.min()), max(pairs.std_2.max(), pairs.std_3.max())]
    axes[1, 0].plot(limits, limits, color="0.5", ls="--")
    axes[1, 0].set(xscale="log", yscale="log", title=f"{len(pairs)} сенсорных пар: разброс двух ориентаций",
                   xlabel="std ориентации 2, исх. ед.", ylabel="std ориентации 3, исх. ед.")
    for same, label in [(False, "Разные позиции"), (True, "Две ориентации одной позиции")]:
        values = dependence["all_pairs"].loc[dependence["all_pairs"].same_site == same, "correlation"]
        axes[1, 1].hist(values, bins=np.linspace(-1, 1, 32), density=True, histtype="step", lw=1.8, label=label)
    axes[1, 1].set(title="Сходство форм: совпадающие и разные позиции", xlabel="Pearson r", ylabel="Плотность")
    axes[1, 1].legend(fontsize=8)
    return fig


def exploratory_groups(table, n_groups=3):
    """Fixed-k descriptive groups of temporal/spectral SHAPE, excluding amplitude."""
    columns = ["roughness", "acf_lag1", "skewness", "excess_kurtosis", "spectral_centroid_hz", "spectral_entropy"]
    values = table[columns].copy()
    if not np.isfinite(values).all().all():
        raise ValueError("Grouping requires finite shape features; inspect flagged rows first")
    median = values.median()
    iqr = values.quantile(.75) - values.quantile(.25)
    z = ((values - median) / iqr.replace(0, 1)).clip(-10, 10)
    labels = fcluster(linkage(z.to_numpy(), method="ward"), n_groups, criterion="maxclust")
    centered = z.to_numpy() - z.to_numpy().mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return dict(labels=labels, scaled=z, projection=u[:, :2] * s[:2],
                profiles=z.assign(group=labels).groupby("group").median(),
                counts=pd.Series(labels).value_counts().sort_index())


def plot_groups(groups):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for label in sorted(set(groups["labels"])):
        mask = groups["labels"] == label
        axes[0].scatter(*groups["projection"][mask].T, s=28, alpha=.7, label=f"Группа {label}: n={mask.sum()}")
    axes[0].set(title="Проекция признаков формы (исследовательские группы)", xlabel="Компонента 1", ylabel="Компонента 2")
    axes[0].legend(fontsize=9)
    im = axes[1].imshow(groups["profiles"], cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axes[1].set_xticks(range(len(groups["profiles"].columns)), groups["profiles"].columns, rotation=40, ha="right")
    axes[1].set_yticks(range(len(groups["profiles"])), groups["profiles"].index)
    axes[1].set(title="Медианные профили групп", ylabel="Номер группы")
    fig.colorbar(im, ax=axes[1], label="Отклонение от медианы / IQR")
    return fig


def group_findings(table, groups):
    grouped = table.assign(group=groups["labels"]).groupby("group")
    profiles = grouped[["roughness", "acf_lag1", "excess_kurtosis", "spectral_centroid_hz", "spectral_entropy"]].median()
    broad = profiles.spectral_centroid_hz.idxmax()
    tails = profiles.excess_kurtosis.idxmax()
    low = profiles.spectral_centroid_hz.idxmin()
    return (
        f"**Качественное чтение текущего разбиения.** Группа **{broad}** ({grouped.size()[broad]} каналов) "
        f"имеет наибольший медианный спектральный центр ({profiles.loc[broad, 'spectral_centroid_hz']:.1f} Гц) "
        f"и roughness={profiles.loc[broad, 'roughness']:.2f}: в ней относительно больше быстрых колебаний. "
        f"У группы **{tails}** ({grouped.size()[tails]} каналов) наибольший медианный эксцесс "
        f"({profiles.loc[tails, 'excess_kurtosis']:.2f}), то есть сильнее отличие хвостов от нормальной формы. "
        f"Группа **{low}** имеет более низкий спектральный центр ({profiles.loc[low, 'spectral_centroid_hz']:.1f} Гц) "
        f"и ACF(1)={profiles.loc[low, 'acf_lag1']:.2f}. "
        "Границы этих групп зависят от признаков, их масштаба и выбранного k."
    )


def pilot_segmentation(analysis, penalty=.02, min_duration_seconds=.03):
    """Run the previously implemented PELT on every complete Excel channel."""
    from src.rectangle_segmentation import segment_pelt
    min_size = max(2, round(min_duration_seconds * analysis["sfreq"]))
    rows, results = [], []
    for y, name in zip(analysis["data"], analysis["metadata"].channel_name):
        result = segment_pelt(y, penalty=penalty, min_size=min_size)
        results.append(result)
        rows.append(dict(channel_name=name, n_breaks=len(result.change_points), objective=result.objective,
                         area_fraction=result.area_fraction, break_times_s=json.dumps(analysis["times"][result.change_points].tolist())))
    return dict(table=pd.DataFrame(rows), results=results, min_size=min_size, penalty=penalty)


def plot_segmentation_density(analysis, pilot, markers=None, bin_seconds=.02):
    times = analysis["times"]
    edges = np.arange(times[0], times[-1] + bin_seconds, bin_seconds)
    counts = np.zeros(len(edges) - 1)
    for result in pilot["results"]:
        histogram, _ = np.histogram(times[result.change_points], bins=edges)
        counts += histogram > 0  # at most one vote per channel per bin
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), layout="constrained")
    axes[0].hist(pilot["table"].n_breaks, bins=np.arange(pilot["table"].n_breaks.max() + 2) - .5, color="#428a94")
    axes[0].set(title=f"Разладки PELT, β={pilot['penalty']}", xlabel="Число внутренних границ", ylabel="Каналов")
    axes[1].stairs(counts / len(pilot["results"]), edges, fill=True, alpha=.7, color="#b9794f")
    axes[1].set(title=f"Согласованность границ: бины {bin_seconds * 1000:g} мс",
                xlabel="Время, с", ylabel="Доля каналов с ≥1 границей в бине")
    _markers(axes[1], markers)
    return fig


def export_channel_atlas(analysis, path, markers=None, per_page=12):
    """PDF containing EVERY original series with its own axes and channel name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(analysis["data"]), per_page):
            fig, axes = plt.subplots(6, 2, figsize=(16, 12), layout="constrained")
            for ax, i in zip(axes.flat, range(start, start + per_page)):
                if i >= len(analysis["data"]):
                    ax.set_visible(False)
                    continue
                row = analysis["statistics"].iloc[i]
                ax.plot(analysis["times"], analysis["data"][i], lw=.55, color="#285c79", rasterized=True)
                ax.set(title=f"{row.channel_name} | std={row['std']:.2e}", xlabel="Время, с", ylabel="Исх. ед.")
                ax.tick_params(labelsize=7)
                ax.title.set_fontsize(9)
                _markers(ax, markers)
            fig.suptitle(f"Все каналы: {start + 1}–{min(start + per_page, len(analysis['data']))}; исходные единицы экспорта")
            pdf.savefig(fig)
            plt.close(fig)
    return path


def export_tables(analysis, output_dir, *, audit=None, dependence=None, groups=None, pilot=None):
    """CSVs with 12 significant digits plus provenance; no raw data copies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = analysis["statistics"].copy()
    if groups is not None:
        table["exploratory_group"] = groups["labels"]
    table.to_csv(output_dir / "all_series_statistics.csv", index=False, float_format="%.12g")
    analysis["metadata"].to_csv(output_dir / "channel_inventory.csv", index=False)
    pd.DataFrame(FEATURE_DESCRIPTIONS.items(), columns=["feature", "definition"]).to_csv(output_dir / "statistics_dictionary.csv", index=False)
    if audit is not None:
        audit["markers"].to_csv(output_dir / "stimulus_markers.csv", index=False)
    if dependence is not None:
        dependence["pairs"].to_csv(output_dir / "sensor_pairs.csv", index=False)
        dependence["all_pairs"].to_csv(output_dir / "channel_correlations.csv", index=False)
    if pilot is not None:
        pilot["table"].to_csv(output_dir / "pelt_all_channels.csv", index=False)
    manifest = dict(
        versions={p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "mne", "openpyxl"]},
        source=audit["summary"] if audit is not None else "MNE epochs; see notebook metadata",
        n_series=len(table), n_samples=analysis["data"].shape[1],
        spectral_window_samples=analysis["nperseg"], frequency_resolution_hz=analysis["frequency_resolution_hz"],
        units="native export; unverified" if audit is not None else "MNE SI units by channel type",
        outlier_rule="abs(y-median) > 6 * 1.4826 * MAD; descriptive flag only",
        grouping="Ward; k=3 fixed; shape features; median/IQR scaling; clipped to ±10" if groups else None,
        pelt=dict(penalty=pilot["penalty"], min_size=pilot["min_size"]) if pilot else None,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return pd.DataFrame([dict(file=p.name, size_kb=round(p.stat().st_size / 1024, 1)) for p in sorted(output_dir.iterdir()) if p.is_file()])


def excel_findings(analysis, audit, dependence, groups, pilot):
    """Data-driven conclusions computed on the current file, not hard-coded."""
    table = analysis["statistics"]
    biggest = table.loc[table["std"].idxmax()]
    most_tails = table.loc[table["robust_outlier_fraction"].idxmax()]
    n_missing = int((table.n_samples - table.n_finite).sum())
    p = pilot["table"]
    tail_statement = (
        f"Максимальная доля экстремальных значений по правилу 6×1.4826×MAD — **{most_tails.channel_name}**, "
        f"{most_tails.robust_outlier_fraction:.2%}; это повод посмотреть форму, а не основание удалить канал. "
        if most_tails.robust_outlier_fraction > 0 else
        "По правилу 6×1.4826×MAD экстремальных отсчётов не найдено ни в одном канале; "
        "это не исключает артефакты других типов. "
    )
    dominant = table[["delta_1_4_fraction", "theta_4_8_fraction", "alpha_8_13_fraction", "beta_13_30_fraction", "gamma_30_45_fraction"]].median().sort_values(ascending=False)
    return (
        f"**Факты по этому файлу.** Исследованы все **{len(table)}** рядов, по **{analysis['data'].shape[1]}** отсчётов; "
        f"частота {analysis['sfreq']:.1f} Гц. Пропущенных/бесконечных значений **{n_missing}**, постоянных рядов **{int(table.constant.sum())}**. "
        f"Найдено **{len(dependence['pairs'])}** пар ориентаций 2/3; других типов датчиков в экспорте по именам не обнаружено.\n\n"
        f"**Амплитуды и форма.** std меняется от {table['std'].min():.3g} до {table['std'].max():.3g} исх. ед. "
        f"(разница в {table['std'].max() / table['std'].min():.2f} раза). Максимум у **{biggest.channel_name}**. "
        f"{tail_statement}"
        f"Медианная корреляция соседних отсчётов {table.acf_lag1.median():.3f}: временные отсчёты явно не независимы.\n\n"
        f"**Спектр и общая структура.** Наибольшая медианная относительная мощность приходится на **{dominant.index[0].replace('_fraction', '')}** "
        f"({dominant.iloc[0]:.1%} мощности 1–45 Гц). Частотный шаг {analysis['frequency_resolution_hz']:.3f} Гц; "
        f"низкочастотные полосы оцениваются грубо. Первая PCA-компонента объясняет **{dependence['variance_ratio'][0]:.1%}** "
        f"дисперсии стандартизованных каналов, 90% — **{dependence['n_components_90']}** компонент. "
        f"Медиана корреляции двух ориентаций одной позиции {dependence['pairs'].correlation.median():.3f}; "
        f"ориентации не взаимозаменяемы.\n\n"
        f"**Применимость к разладкам.** При β={pilot['penalty']} и минимуме {pilot['min_size']} отсчётов PELT даёт "
        f"от {p.n_breaks.min()} до {p.n_breaks.max()} границ на канал, медиана **{p.n_breaks.median():g}**. "
        "Это характеристика выбранного критерия; без ручной разметки нельзя оценить точность обнаружения.\n\n"
        "**Границы вывода.** Один участник/одна проба не дают популяционного вывода. "
        "Коды STI101 требуют словаря именно этого эксперимента. Единицы и история фильтрации экспорта не документированы. "
        "Названия каналов поддерживают гипотезу о планарных градиометрах, но не заменяют метаданные прибора. "
        "Три группы по форме — описательный результат при фиксированном k, а не три типа мозговой активности."
    )


def epochs_arrays(epochs, source="MNE sample"):
    """Flatten trial×channel into rows without joining time across trials."""
    data = epochs.get_data(copy=False, verbose=False)
    n_epochs, n_channels, n_times = data.shape
    inventory = infer_channels(epochs.ch_names, source=source,
                              verified_types=epochs.get_channel_types(), bads=epochs.info["bads"])
    metadata = pd.concat([inventory] * n_epochs, ignore_index=True)
    labels = {code: label for label, code in epochs.event_id.items()}
    conditions = [labels[code] for code in epochs.events[:, -1]]
    metadata["epoch_idx"] = np.repeat(np.arange(n_epochs), n_channels)
    metadata["condition"] = np.repeat(conditions, n_channels)
    metadata["series_id"] = [f"mne/ep{e}/{c}" for e in range(n_epochs) for c in epochs.ch_names]
    return data.reshape(-1, n_times), epochs.times.copy(), metadata, inventory


def summarize_mne(epochs, analysis, *, seed=42):
    """Condition comparisons use trials, never individual time samples as n."""
    table = analysis["statistics"]
    # Include known-bad channels in exported per-series statistics but omit
    # them from condition-level descriptive comparisons.
    good = table.loc[~table.marked_bad & table.channel_type.isin(["grad", "mag", "eeg"])]
    trial = good.groupby(["epoch_idx", "condition", "channel_type"], as_index=False).agg(
        median_channel_std=("std", "median"), median_channel_ptp=("ptp", "median"),
        median_channel_roughness=("roughness", "median"),
    )
    channel_summary = table.groupby(["condition", "channel_name", "channel_type", "unit", "marked_bad"], as_index=False).agg(
        n_trials=("std", "size"), median_std=("std", "median"), q25_std=("std", lambda x: x.quantile(.25)),
        q75_std=("std", lambda x: x.quantile(.75)), median_ptp=("ptp", "median"),
        median_roughness=("roughness", "median"), median_acf1=("acf_lag1", "median"),
        median_excess_kurtosis=("excess_kurtosis", "median"),
        median_outlier_fraction=("robust_outlier_fraction", "median"),
    )
    baseline = epochs.times < 0
    response = (epochs.times >= .05) & (epochs.times < .3)
    if not baseline.any() or not response.any():
        raise ValueError("Need a prestimulus baseline and a 50–300 ms response window")
    rng = np.random.default_rng(seed)
    curves, response_rows, reliability = [], [], []
    for condition in epochs.event_id:
        selected = epochs[condition]
        condition_data = selected.get_data(copy=False, verbose=False)
        types = np.array(selected.get_channel_types())
        for kind in ["grad", "mag", "eeg"]:
            picks = np.flatnonzero((types == kind) & ~np.isin(selected.ch_names, epochs.info["bads"]))
            if not len(picks):
                continue
            values = condition_data[:, picks]
            # Scale first for numerically stable ratios in MEG units.
            scale = {"grad": 1e13, "mag": 1e15, "eeg": 1e6}[kind]
            values = values * scale
            evoked = np.mean(values, axis=0)
            rms_evoked = np.sqrt(np.mean(evoked ** 2, axis=0))
            curves.append(dict(condition=condition, channel_type=kind, n_trials=len(values),
                               n_channels=len(picks), rms_evoked=rms_evoked,
                               unit={"grad": "fT/cm", "mag": "fT", "eeg": "µV"}[kind]))
            base_rms = np.sqrt(np.mean(values[:, :, baseline] ** 2, axis=2))
            post_rms = np.sqrt(np.mean(values[:, :, response] ** 2, axis=2))
            ratios = np.nanmedian(_safe_divide(post_rms, base_rms), axis=1)
            for i, ratio in enumerate(ratios):
                response_rows.append(dict(condition=condition, channel_type=kind, trial_in_condition=i,
                                          median_response_baseline_rms_ratio=float(ratio)))
            order = rng.permutation(len(values))
            half = len(order) // 2
            a = values[order[:half]].mean(axis=0)[:, epochs.times >= 0].ravel()
            b = values[order[half:]].mean(axis=0)[:, epochs.times >= 0].ravel()
            reliability.append(dict(condition=condition, channel_type=kind, n_trials=len(values),
                                    split_half_correlation=float(np.corrcoef(a, b)[0, 1])))
    return dict(trial_statistics=trial, channel_statistics=channel_summary, curves=curves,
                response_ratios=pd.DataFrame(response_rows), reliability=pd.DataFrame(reliability))


def plot_mne_conditions(epochs, summary):
    kinds = ["grad", "mag", "eeg"]
    conditions = list(epochs.event_id)
    colors = dict(zip(conditions, ["#245a81", "#69a0bc", "#c46a43", "#dfac68"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for col, kind in enumerate(kinds):
        curves = [c for c in summary["curves"] if c["channel_type"] == kind]
        for curve in curves:
            axes[0, col].plot(epochs.times, curve["rms_evoked"], color=colors[curve["condition"]],
                               label=f"{curve['condition']} (n={curve['n_trials']})")
        axes[0, col].axvline(0, color="0.4", ls="--")
        axes[0, col].set(title=f"{kind}: RMS усреднённого отклика", xlabel="Время от стимула, с",
                         ylabel=curves[0]["unit"] if curves else "")
        axes[0, col].legend(fontsize=8)
        ratios = summary["response_ratios"]
        arrays = [ratios.loc[(ratios.channel_type == kind) & (ratios.condition == c),
                             "median_response_baseline_rms_ratio"].dropna() for c in conditions]
        boxes = axes[1, col].boxplot(arrays, patch_artist=True, tick_labels=conditions, showfliers=True)
        for patch, c in zip(boxes["boxes"], conditions):
            patch.set_facecolor(colors[c])
            patch.set_alpha(.7)
        axes[1, col].tick_params(axis="x", rotation=30)
        axes[1, col].axhline(1, color="0.4", ls="--")
        axes[1, col].set(title=f"{kind}: отдельные пробы", ylabel="Медиана по каналам: RMS 50–300 мс / RMS до 0")
    return fig


def plot_mne_trial_variation(summary):
    table = summary["trial_statistics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    scales = {"grad": (1e13, "fT/cm"), "mag": (1e15, "fT"), "eeg": (1e6, "µV")}
    for ax, kind in zip(axes, scales):
        for condition, group in table.loc[table.channel_type == kind].groupby("condition"):
            ax.scatter(group.epoch_idx, group.median_channel_std * scales[kind][0], s=13, alpha=.7, label=condition)
        ax.set(title=f"{kind}: вариабельность между пробами", xlabel="Порядковый номер эпохи",
               ylabel=f"Медиана std каналов, {scales[kind][1]}", yscale="log")
    axes[0].legend(fontsize=8)
    return fig


def plot_mne_all_series(epochs, analysis):
    """One cell per trial×channel, including technical rows and known bads."""
    kinds = [k for k in ["grad", "mag", "eeg", "eog", "stim"] if k in epochs.get_channel_types()]
    types = np.array(epochs.get_channel_types())
    names = np.array(epochs.ch_names)
    n_channels = len(names)
    matrix = analysis["statistics"]["std"].to_numpy().reshape(len(epochs), n_channels)
    conditions = analysis["metadata"].condition.to_numpy()[::n_channels]
    order = np.argsort(conditions, kind="stable")
    sorted_conditions = conditions[order]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(15, 2.5 * len(kinds)), layout="constrained", squeeze=False)
    scales = {"grad": (1e13, "fT/cm"), "mag": (1e15, "fT"), "eeg": (1e6, "µV"),
              "eog": (1e6, "µV"), "stim": (1, "код")}
    for ax, kind in zip(axes.flat, kinds):
        picks = np.flatnonzero(types == kind)
        values = matrix[order][:, picks] * scales[kind][0]
        logged = np.full_like(values, np.nan)
        np.log10(values, out=logged, where=values > 0)
        im = ax.imshow(np.ma.masked_invalid(logged), aspect="auto", cmap="viridis", interpolation="nearest")
        ticks, labels = [], []
        for condition in sorted(set(conditions)):
            positions = np.flatnonzero(sorted_conditions == condition)
            ticks.append(np.mean(positions))
            labels.append(f"{condition} (n={len(positions)})")
            ax.axhline(positions[-1] + .5, color="white", lw=.7)
        ax.set_yticks(ticks, labels, fontsize=8)
        x_ticks = np.unique(np.linspace(0, len(picks) - 1, min(10, len(picks))).astype(int))
        ax.set_xticks(x_ticks, names[picks[x_ticks]], fontsize=8)
        ax.set(title=f"{kind}: все отдельные ряды — std каждой пары «проба × канал»", xlabel="Канал")
        fig.colorbar(im, ax=ax, label=f"log10(std в {scales[kind][1]}); белый = 0", pad=.01)
    return fig


def mne_condition_metrics(epochs, summary):
    """RMS peaks are descriptive sensor-space maxima within 50–300 ms."""
    response_mask = (epochs.times >= .05) & (epochs.times < .3)
    response_times = epochs.times[response_mask]
    rows = []
    for curve in summary["curves"]:
        key = (curve["condition"], curve["channel_type"])
        values = curve["rms_evoked"][response_mask]
        ratios = summary["response_ratios"]
        ratios = ratios.loc[(ratios.condition == key[0]) & (ratios.channel_type == key[1]),
                            "median_response_baseline_rms_ratio"]
        reliability = summary["reliability"]
        reliability = reliability.loc[(reliability.condition == key[0]) & (reliability.channel_type == key[1]),
                                      "split_half_correlation"].iloc[0]
        rows.append(dict(condition=key[0], channel_type=key[1], n_trials=curve["n_trials"],
                         n_good_channels=curve["n_channels"], peak_time_ms=1000 * response_times[np.argmax(values)],
                         peak_rms=values.max(), unit=curve["unit"], median_response_ratio=ratios.median(),
                         split_half_correlation=reliability))
    return pd.DataFrame(rows)


def export_mne_tables(epochs, analysis, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis["statistics"].to_csv(output_dir / "all_trial_channel_statistics.csv.gz", index=False,
                                   float_format="%.12g", compression={"method": "gzip", "mtime": 0})
    for name, table in [
        ("condition_channel_statistics", summary["channel_statistics"]),
        ("trial_statistics", summary["trial_statistics"]),
        ("response_ratios", summary["response_ratios"]),
        ("condition_metrics", mne_condition_metrics(epochs, summary)),
    ]:
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.12g")
    manifest = dict(
        n_epochs=len(epochs), n_channels=len(epochs.ch_names), n_rows=len(analysis["statistics"]),
        sfreq_hz=epochs.info["sfreq"], tmin_s=epochs.tmin, tmax_s=epochs.tmax,
        bads=epochs.info["bads"], event_id=epochs.event_id,
        preprocessing="Existing load_sample_epochs: MEG/EEG bandpass 1–40 Hz, baseline (None,0), default MNE projectors; EOG/stim are not physiological data picks for filtering",
        comparisons="Known bads omitted from condition summaries, retained in per-series CSV and inventory",
        versions={p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "mne"]},
        seed=42,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return pd.DataFrame([dict(file=p.name, size_mb=round(p.stat().st_size / 1024 ** 2, 2))
                         for p in sorted(output_dir.iterdir()) if p.is_file()])


def mne_findings(epochs, analysis, metrics):
    n_channels = len(epochs.ch_names)
    table = analysis["statistics"]
    lines = [
        f"Исследовано **{len(epochs)} проб × {n_channels} каналов = {len(table):,} рядов** "
        f"по **{len(epochs.times)} отсчёту**. Пропусков: **{int((table.n_samples-table.n_finite).sum())}**. "
        "Технические stim-каналы включены в общую таблицу статистик; их постоянство нормально для большей части эпох.",
        f"Из сравнений условий исключены каналы, заранее помеченные в Info как плохие: **{', '.join(epochs.info['bads'])}**. "
        "В полном CSV они сохранены с флагом marked_bad.",
    ]
    for kind, group in metrics.groupby("channel_type", sort=False):
        strongest = group.loc[group.median_response_ratio.idxmax()]
        lines.append(
            f"**{kind}:** наибольшее медианное отношение RMS после/до стимула у **{strongest.condition}** "
            f"({strongest.median_response_ratio:.2f}). Пики RMS усреднённого отклика в окне 50–300 мс лежат "
            f"в диапазоне {group.peak_time_ms.min():.0f}–{group.peak_time_ms.max():.0f} мс; "
            f"корреляция двух случайных половин проб — {group.split_half_correlation.min():.2f}–{group.split_half_correlation.max():.2f}."
        )
    peak_table = metrics.pivot(index="condition", columns="channel_type", values="peak_time_ms")
    if set(["auditory/left", "auditory/right", "visual/left", "visual/right"]).issubset(peak_table.index):
        auditory = peak_table.loc[["auditory/left", "auditory/right"], ["grad", "mag"]].to_numpy()
        vl = peak_table.loc["visual/left"]
        vr = peak_table.loc["visual/right"]
        lines.append(
            "**Как отличаются формы отклика в этой записи.** "
            f"У двух слуховых условий максимум пространственного MEG-RMS возникает около "
            f"{auditory.min():.0f}–{auditory.max():.0f} мс. У visual/left максимумы трёх типов "
            f"датчиков приходятся на {vl.min():.0f}–{vl.max():.0f} мс. У visual/right максимум "
            f"grad-RMS — {vr['grad']:.0f} мс, mag-RMS — {vr['mag']:.0f} мс, а EEG-RMS — {vr['eeg']:.0f} мс. "
            "Таким образом, даже внутри одной модальности форма и максимум зависят от условия и типа датчика. "
            "Все эти максимумы выбраны в окне 50–300 мс; пространственный RMS объединяет разные сенсорные компоненты."
        )
    lines.append(
        "Это наблюдения одного участника после заданной предобработки. Различия сенсорного RMS "
        "не измеряют когнитивную значимость, а split-half зависит от конкретного разбиения. "
        "Для статистического сравнения условий нужны заранее выбранные каналы/временные окна, "
        "учёт повторных измерений и независимая проверка на других участниках."
    )
    return "\n\n".join(lines)
