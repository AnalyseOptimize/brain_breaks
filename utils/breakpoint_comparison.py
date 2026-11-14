"""Reproducible experiment/plotting helpers for breakpoint_comparison.ipynb.

Hyperparameters are selected only on the calibration simulations. Real MEG
has no breakpoint annotations and is NEVER used to tune or rank accuracy.
"""
from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import data_loading as mseg
from .paths import DATA_DIR
from src.breakpoint_common import BreakResult, match_breakpoints, pelt_search, prepare_signal
from src.cpop_segmentation import segment_cpop
from src.pelt_regression import segment_regression
from src.pelt_rbf import segment_rbf
from src.trend_filtering import segment_trend_filter
from src.rectangle_segmentation import RectangleCost


METHODS = ["CPOP", "PELT-linear", "PELT-quadratic", "Trend-filter-L1", "PELT-RBF", "Rectangles"]
PARAMETER_GRID = {
    "CPOP": [.003, .01, .035, .12, .4],
    "PELT-linear": [.003, .01, .035, .12, .4],
    "PELT-quadratic": [.003, .01, .035, .12, .4, 1.2],
    "Trend-filter-L1": [.003, .01, .035, .12, .4, 1.2],
    "PELT-RBF": [.003, .01, .035, .12],
    "Rectangles": [.01, .025, .06, .15],
}
SCENARIOS = ["slope", "quadratic", "mean", "variance", "ar_slope", "noise", "spike", "oscillation"]


@dataclass
class SeriesCase:
    case_id: str
    scenario: str
    signal: np.ndarray
    times: np.ndarray
    true_breaks: list[int] | None
    clean_trend: np.ndarray | None = None
    source: str = "synthetic"
    markers: list[float] | None = None


def synthetic_suite(seed=100, n=240, repeats=2, sfreq=200.):
    """Eight controlled mechanisms; stationary oscillations have no breaks.

    The synthetic target is persistent regime change. A single impulse and
    a stationary 10-Hz oscillation are explicit negative controls. Separate
    seeds AND lengths are used for calibration and final testing.
    """
    cases = []
    t = np.arange(n)/sfreq
    x = np.linspace(0, 1, n)
    b1, b2 = n//3, 2*n//3
    slope = np.interp(np.arange(n), [0, b1, b2, n-1], [0, 1.5, -.8, .8])
    quadratic = 10*x*x-30*np.maximum(x-x[b1], 0)**2+35*np.maximum(x-x[b2], 0)**2
    for repeat in range(repeats):
        for index, scenario in enumerate(SCENARIOS):
            rng = np.random.default_rng(seed+1009*repeat+47*index)
            truth = [b1, b2] if scenario in SCENARIOS[:5] else []
            if scenario in ["slope", "ar_slope"]:
                clean = slope.copy()
                noise = rng.normal(0, .1, n)
                if scenario == "ar_slope":
                    noise = lfilter([.12*np.sqrt(1-.8**2)], [1, -.8], rng.normal(size=n+100))[100:]
                observed = clean+noise
            elif scenario == "quadratic":
                clean = quadratic.copy()
                observed = clean+rng.normal(0, .05, n)
            elif scenario == "mean":
                clean = np.repeat([0., 1.5, -1.], [b1, b2-b1, n-b2])
                observed = clean+rng.normal(0, .15, n)
            elif scenario == "variance":
                clean = np.zeros(n)
                observed = rng.normal(size=n)*np.repeat([.08, .65, .12], [b1, b2-b1, n-b2])
            elif scenario == "spike":
                clean = .5*x
                observed = clean+rng.normal(0, .08, n)
                observed[n//2] += 2.5
            elif scenario == "oscillation":
                clean = np.zeros(n)
                observed = .6*np.sin(2*np.pi*10*t)+rng.normal(0, .08, n)
            else:
                clean = np.zeros(n)
                observed = rng.normal(0, .15, n)
            cases.append(SeriesCase(f"{scenario}/seed{seed}/rep{repeat}", scenario,
                                    observed, t.copy(), truth, clean))
    return cases


def _segment_rectangles(signal, penalty, min_size, jump):
    started = perf_counter()
    cost = RectangleCost(signal)
    class AreaAdapter:
        def error_many(self, starts, end):
            return cost._areas(starts, end)
    points, objective, diagnostics = pelt_search(AreaAdapter(), len(signal), penalty, min_size, jump)
    fit, start = np.empty_like(signal), 0
    for end in points:
        fit[start:end] = signal[start:end].mean()
        start = end
    diagnostics["fit_role"] = "segment mean for display, not the rectangle objective"
    return BreakResult("Rectangles", points, fit, objective, penalty, perf_counter()-started,
                       dict(min_size=min_size, jump=jump), diagnostics)


def run_method(case, method, parameter, *, min_duration=.03, grid_seconds=.01):
    sfreq = 1/np.median(np.diff(case.times))
    min_size = max(3, int(np.ceil(min_duration*sfreq-1e-9)))
    jump = max(1, round(grid_seconds*sfreq))
    prepare_signal(case.signal, min_size, jump, times=case.times)
    kwargs = dict(min_size=min_size, jump=jump)
    if method == "CPOP":
        result = segment_cpop(case.signal, penalty=parameter, **kwargs)
    elif method in ["PELT-linear", "PELT-quadratic"]:
        result = segment_regression(case.signal, penalty=parameter, degree=1 if method == "PELT-linear" else 2, **kwargs)
    elif method == "Trend-filter-L1":
        _, z, _, _, _, _ = prepare_signal(case.signal, min_size, jump)
        differences = np.diff(z, n=2)
        sigma = float(np.median(np.abs(differences-np.median(differences))) / (.6744897501960817*np.sqrt(6)))
        # For the mean-RSS criterion, a noise-scaled integrated-noise penalty
        # is c*sigma*sqrt(n). Unlike lambda/lambda_max it is not inflated by
        # the amplitude of the unknown deterministic trend itself.
        result = segment_trend_filter(case.signal, lambda_value=parameter*sigma*np.sqrt(len(z)), **kwargs)
        result.params.update(noise_multiplier=parameter, noise_sd_estimate=sigma)
    elif method == "PELT-RBF":
        result = segment_rbf(case.signal, penalty=parameter, **kwargs)
    elif method == "Rectangles":
        result = _segment_rectangles(case.signal, parameter, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    result.method = method
    return result


def evaluate_result(case, result, tolerance_seconds=.02):
    duration = case.times[-1]-case.times[0]
    _, z, offset, scale, _, _ = prepare_signal(case.signal, min_size=3)
    fit_z = (result.fit-offset)/scale
    residual = z-fit_z
    sd = residual.std()
    acf = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if sd > 1e-12 else np.nan
    row = dict(
        case_id=case.case_id, source=case.source, scenario=case.scenario, method=result.method,
        n_samples=len(case.signal), n_breaks=len(result.change_points),
        elapsed_seconds=result.elapsed_seconds, objective=result.objective,
        parameter=result.params.get("noise_multiplier", result.penalty),
        min_size=result.params["min_size"], jump=result.params["jump"],
        sfreq_hz=1/np.median(np.diff(case.times)), matching_tolerance_seconds=tolerance_seconds,
        fit_rmse_normalized=float(np.sqrt(np.mean(residual**2))), residual_acf1=acf,
        break_times_s=json.dumps(case.times[result.change_points].tolist()),
        tf_raw_knots=result.diagnostics.get("raw_knot_count", np.nan),
        solver_duality_gap=result.diagnostics.get("duality_gap", np.nan),
    )
    if case.true_breaks is not None:
        matched = match_breakpoints(case.times[result.change_points], case.times[case.true_breaks], tolerance_seconds)
        row.update({k: v for k, v in matched.items() if k not in ["matches", "timing_mae"]})
        row.update(n_true=len(case.true_breaks), timing_mae_ms=1000*matched["timing_mae"],
                   false_alarms_per_second=matched["fp"]/duration)
    else:
        row.update({key: np.nan for key in ["tp", "fp", "fn", "precision", "recall", "f1", "n_true", "timing_mae_ms", "false_alarms_per_second"]})
    row["trend_rmse_normalized"] = (
        float(np.sqrt(np.mean(((result.fit-case.clean_trend)/scale)**2))) if case.clean_trend is not None else np.nan
    )
    return row


def calibrate(cases, parameter_grid=None, *, min_duration=.03, grid_seconds=.01, tolerance_seconds=.02):
    """Pick each method's parameter on calibration cases only; no real data."""
    if any(c.source != "synthetic" or c.true_breaks is None for c in cases):
        raise ValueError("Calibration requires labeled synthetic cases only")
    parameter_grid = PARAMETER_GRID if parameter_grid is None else parameter_grid
    rows = []
    for method, parameters in parameter_grid.items():
        for parameter in parameters:
            for case in cases:
                result = run_method(case, method, parameter, min_duration=min_duration, grid_seconds=grid_seconds)
                rows.append(evaluate_result(case, result, tolerance_seconds))
    table = pd.DataFrame(rows)
    # Equal scenario weights, even if the caller supplies different repeat counts.
    scores = table.groupby(["method", "parameter", "scenario"], as_index=False).agg(
        f1=("f1", "mean"), fp_per_s=("false_alarms_per_second", "mean"))
    scores = scores.groupby(["method", "parameter"], as_index=False).agg(
        calibration_macro_f1=("f1", "mean"), calibration_fp_per_s=("fp_per_s", "mean"))
    selected = {}
    for method, group in scores.groupby("method"):
        best = group.sort_values(["calibration_macro_f1", "calibration_fp_per_s", "parameter"],
                                 ascending=[False, True, False]).iloc[0]
        selected[method] = float(best.parameter)
    return selected, scores, table


def compare_cases(cases, parameters, *, min_duration=.03, grid_seconds=.01, tolerance_seconds=.02):
    rows, results = [], {}
    for case in cases:
        results[case.case_id] = {}
        for method in METHODS:
            result = run_method(case, method, parameters[method], min_duration=min_duration, grid_seconds=grid_seconds)
            results[case.case_id][method] = result
            rows.append(evaluate_result(case, result, tolerance_seconds))
    return pd.DataFrame(rows), results


def load_real_cases(data_dir=DATA_DIR, include_mne=True):
    """Three Excel amplitude quantiles and first single trial per MNE condition.

    Selection does not inspect fitted breakpoints or ground truth. MNE known
    bad channels are avoided. Existing project loaders handle all signal I/O.
    """
    data_dir = Path(data_dir)
    series = mseg.load_xlsx_series(str(data_dir/"excel-data"/"58voln_trial001_12.xlsx"))
    order = np.argsort([np.std(s["signal"]) for s in series])
    cases = []
    from .meg_eda import audit_excel
    marker_times = audit_excel(data_dir/"excel-data"/"58voln_trial001_12.xlsx")["markers"].time_s.tolist()
    for index, label in zip([order[0], order[len(order)//2], order[-1]], ["low-std", "median-std", "high-std"]):
        s = series[index]
        cases.append(SeriesCase(f"Excel/{s['channel_name']}/{label}", "unlabeled", s["signal"], s["times"],
                                None, source="Excel", markers=marker_times))
    if include_mne:
        epochs = mseg.load_sample_epochs(condition=None, data_dir=str(data_dir))
        channel = "MEG 0113"
        if channel in epochs.info["bads"]:
            channel = next(name for name, kind in zip(epochs.ch_names, epochs.get_channel_types())
                           if kind == "grad" and name not in epochs.info["bads"])
        for condition in epochs.event_id:
            s = mseg.select_series(epochs[condition], epoch_indices=[0], channel_names=[channel])[0]
            cases.append(SeriesCase(f"MNE/{condition}/{channel}/first-trial", "unlabeled",
                                    s["signal"], s["times"], None, source="MNE", markers=[0.]))
    return cases


def summary_tables(table):
    synthetic = table.loc[table.n_true.notna()]
    by_scenario = synthetic.groupby(["scenario", "method"], as_index=False).agg(
        f1=("f1", "mean"), timing_mae_ms=("timing_mae_ms", "mean"),
        fp_per_s=("false_alarms_per_second", "mean"), runtime_s=("elapsed_seconds", "mean"),
        trend_rmse=("trend_rmse_normalized", "mean"))
    positives = synthetic.loc[synthetic.n_true > 0].groupby("method").agg(
        positive_f1=("f1", "mean"), timing_mae_ms=("timing_mae_ms", "mean"))
    negatives = synthetic.loc[synthetic.n_true == 0].groupby("method").agg(
        negative_fp_per_s=("false_alarms_per_second", "mean"),
        negative_zero_detection_fraction=("n_breaks", lambda x: np.mean(x == 0)))
    overall = positives.join(negatives).join(synthetic.groupby("method").elapsed_seconds.mean().rename("mean_runtime_s"))
    return overall.reindex(METHODS), by_scenario


def plot_scenarios(cases):
    selected = [next(c for c in cases if c.scenario == s) for s in SCENARIOS]
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), layout="constrained")
    for ax, case in zip(axes.flat, selected):
        ax.plot(case.times, case.signal, lw=.7, color=".5", label="Наблюдения")
        ax.plot(case.times, case.clean_trend, lw=1.8, color="#246783", label="Заданный тренд")
        for b in case.true_breaks:
            ax.axvline(case.times[b], color="#36864d", ls=":")
        ax.set(title=case.scenario, xlabel="Время, с", ylabel="Амплитуда")
    axes[0, 0].legend(fontsize=8)
    return fig


def plot_calibration(scores, selected):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for ax, method in zip(axes.flat, METHODS):
        group = scores.loc[scores.method == method].sort_values("parameter")
        ax.semilogx(group.parameter, group.calibration_macro_f1, marker="o", color="#246783")
        ax.axvline(selected[method], color="#b47a3b", ls="--")
        ax.set(title=method, xlabel="c (noise multiplier)" if method == "Trend-filter-L1" else "β", ylabel="Calibration macro-F1", ylim=(0, 1.02))
        ax.grid(alpha=.2)
    return fig


def plot_test_summary(by_scenario, overall):
    matrix = by_scenario.pivot(index="scenario", columns="method", values="f1").reindex(index=SCENARIOS, columns=METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    im = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[0].set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right")
    axes[0].set_yticks(range(len(matrix)), matrix.index)
    axes[0].set_title("Независимый тест: F1 по механизмам")
    for row in range(len(matrix)):
        for col in range(len(METHODS)):
            value = matrix.iloc[row, col]
            axes[0].text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value > .65 else "black", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="F1")
    axes[1].barh(METHODS, overall.negative_fp_per_s, color="#b9794f")
    axes[1].set(title="Отрицательные контроли", xlabel="Ложных границ в секунду")
    axes[2].barh(METHODS, overall.mean_runtime_s, color="#428a94")
    axes[2].set(title="Среднее время одного тестового ряда", xlabel="Секунды", xscale="log")
    return fig


def plot_case_comparison(case, results):
    fig, axes = plt.subplots(len(METHODS)+1, 1, figsize=(15, 2.0*(len(METHODS)+1)), sharex=True, layout="constrained")
    axes[0].plot(case.times, case.signal, color=".35", lw=.7, label="Сигнал")
    if case.clean_trend is not None:
        axes[0].plot(case.times, case.clean_trend, color="#36864d", lw=1.8, label="Заданный тренд")
    axes[0].set_title(case.case_id)
    axes[0].legend(fontsize=8)
    for ax, method in zip(axes[1:], METHODS):
        result = results[method]
        ax.plot(case.times, case.signal, color=".7", lw=.55)
        role = "среднее сегментов" if method in ["PELT-RBF", "Rectangles"] else "оценка тренда"
        ax.plot(case.times, result.fit, color="#246783", lw=1.4, label=role)
        for i, b in enumerate(result.change_points):
            ax.axvline(case.times[b], color="#bb6545", ls="--", lw=.9, label="Найдено" if i == 0 else None)
        for i, b in enumerate(case.true_breaks or []):
            ax.axvline(case.times[b], color="#36864d", ls=":", lw=1.4, label="Истина" if i == 0 else None)
        for marker in case.markers or []:
            ax.axvline(marker, color="#8755a0", ls="-.", lw=.8)
        ax.set(title=f"{method}: {len(result.change_points)} границ, {result.elapsed_seconds:.3f} с", ylabel="Исх. ед.")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Время, с")
    return fig


def plot_real_boundaries(cases, results):
    fig, axes = plt.subplots(len(cases), 1, figsize=(15, 2.3*len(cases)), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, cases):
        for row, method in enumerate(METHODS):
            points = case.times[results[case.case_id][method].change_points]
            ax.scatter(points, np.full(len(points), row), marker="|", s=130, color=f"C{row}")
        for marker in case.markers or []:
            ax.axvline(marker, color="#8755a0", ls="--", lw=.8)
        ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
        ax.set(title=case.case_id, xlabel="Время, с", xlim=(case.times[0], case.times[-1]))
        ax.grid(axis="x", alpha=.2)
    return fig


def boundary_agreement(case, results, tolerance_seconds=.02):
    matrix = np.ones((len(METHODS), len(METHODS)))
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            matrix[i, j] = match_breakpoints(case.times[results[a].change_points], case.times[results[b].change_points], tolerance_seconds)["f1"]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=40, ha="right")
    ax.set_yticks(range(len(METHODS)), METHODS)
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] > .65 else "black")
    ax.set_title("Согласованность границ между методами\nЭто не точность относительно разметки")
    fig.colorbar(im, ax=ax, label="Парный F1")
    return fig


def tolerance_sensitivity(cases, results, tolerances=(.01, .02, .04)):
    rows = []
    for tolerance in tolerances:
        for case in cases:
            if not case.true_breaks:
                continue
            for method in METHODS:
                result = results[case.case_id][method]
                matched = match_breakpoints(case.times[result.change_points], case.times[case.true_breaks], tolerance)
                rows.append(dict(method=method, tolerance_ms=1000*tolerance, f1=matched["f1"]))
    table = pd.DataFrame(rows).groupby(["method", "tolerance_ms"], as_index=False).f1.mean()
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for method, group in table.groupby("method", sort=False):
        ax.plot(group.tolerance_ms, group.f1, marker="o", label=method)
    ax.set(xlabel="Допуск совпадения границ, мс", ylabel="Средний F1 на положительных сценариях", ylim=(0, 1.03))
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.2)
    return table, fig


def parameter_sensitivity(case, selected, factors=(.5, 1., 2.)):
    rows = []
    for method in METHODS:
        reference = run_method(case, method, selected[method])
        for factor in factors:
            result = reference if factor == 1 else run_method(case, method, selected[method]*factor)
            agreement = match_breakpoints(case.times[result.change_points], case.times[reference.change_points], .02)
            rows.append(dict(method=method, parameter_factor=factor, n_breaks=len(result.change_points),
                             agreement_with_selected=agreement["f1"], elapsed_seconds=result.elapsed_seconds))
    return pd.DataFrame(rows)


def export_comparison(output_dir, selected, scores, calibration_table, test_table, real_table, cases, results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in [("calibration_scores", scores), ("calibration_predictions", calibration_table),
                            ("test_metrics", test_table), ("real_descriptive_metrics", real_table)]:
        table.to_csv(output_dir/f"{filename}.csv", index=False, float_format="%.12g")
    predictions = []
    for case in cases:
        for method, result in results[case.case_id].items():
            diagnostics = {k: v for k, v in result.diagnostics.items() if k not in ["curvature", "state_counts", "coefficients_normalized"]}
            predictions.append(dict(case_id=case.case_id, method=method, breakpoints=result.breakpoints,
                                    params=result.params, diagnostics=diagnostics))
    (output_dir/"predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    manifest = dict(
        selected_parameters=selected,
        calibration_case_ids=sorted(calibration_table.case_id.unique().tolist()),
        test_case_ids=sorted(test_table.case_id.unique().tolist()),
        calibration_lengths=sorted(calibration_table.n_samples.unique().tolist()),
        test_lengths=sorted(test_table.n_samples.unique().tolist()),
        matching_tolerance_seconds=test_table.matching_tolerance_seconds.unique().tolist(),
        calibration_rule="Equal scenario weights, mean F1; ties: fewer FP/s, stronger regularization",
        normalization="Each method globally standardizes its signal; rectangles use global range",
        trend_filter_postprocessing="curvature threshold 1e-5, grid snapping, min-size NMS; raw optimum kept",
        trend_filter_penalty="lambda=c*sigma_hat*sqrt(n) for mean RSS; sigma_hat=MAD(D2 z)/(.67448975*sqrt(6)); iid-noise scaling is a calibration heuristic",
        versions={p: version(p) for p in ["numpy", "scipy", "pandas", "matplotlib", "mne", "clarabel"]},
        cases=[dict(case_id=c.case_id, source=c.source, scenario=c.scenario,
                    n_samples=len(c.signal), true_breakpoints=c.true_breaks, markers_s=c.markers,
                    signal_sha256=hashlib.sha256(np.asarray(c.signal, dtype="<f8").tobytes()).hexdigest()) for c in cases],
    )
    (output_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return pd.DataFrame([dict(file=p.name, size_kb=round(p.stat().st_size/1024, 1)) for p in sorted(output_dir.iterdir()) if p.is_file()])


def comparison_findings(overall, by_scenario, real_table):
    lines = []
    for scenario in ["slope", "quadratic", "mean", "variance", "ar_slope"]:
        group = by_scenario.loc[by_scenario.scenario == scenario]
        top = group.f1.max()
        leaders = group.loc[np.isclose(group.f1, top, atol=1e-10), "method"].tolist()
        lines.append(f"**{scenario}:** наибольший средний тестовый F1={top:.2f} у {', '.join(leaders)}.")
    negatives = overall.sort_values(["negative_fp_per_s", "positive_f1"], ascending=[True, False])
    lines.append(
        f"**Отрицательные контроли:** минимальная средняя частота ложных границ — "
        f"{negatives.negative_fp_per_s.iloc[0]:.2f}/с у {negatives.index[0]}. "
        "Этот показатель нужно читать вместе с F1 на рядах с настоящими изменениями: отказ от любых границ тоже убирает ложные тревоги."
    )
    for source, group in real_table.groupby("source"):
        counts = group.groupby("method").n_breaks.agg(["min", "max"])
        description = "; ".join(f"{m}: {int(counts.loc[m, 'min'])}–{int(counts.loc[m, 'max'])}" for m in METHODS)
        lines.append(f"**{source}, число найденных границ:** {description}. Это результаты фиксированных параметров без разметки.")
    lines.append(
        "Калибровка отражает выбранную смесь восьми синтетических механизмов. На реальных данных "
        "нет оценки точности; большое число совпадающих границ между методами также не заменяет аннотации. "
        "Проверка ручной разметкой и устойчивость к предобработке остаются следующим этапом."
    )
    return "\n\n".join(lines)
