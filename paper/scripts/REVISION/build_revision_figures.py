from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_PHYS = REPO_ROOT / "REVISION" / "physiology_v2"
DEFAULT_PHASE1 = REPO_ROOT / "REVISION" / "phase1_v4"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "figures_v2"

FIGURE4_DISPLAY_QUANTILE = 0.99
FIGURE4_COLOR_QUANTILE = 0.95
FIGURE4_DEFAULT_YMAX_SS = 15.0
FIGURE4_DEFAULT_YMAX_AHI = 50.0
FIGURE4_DEFAULT_YMAX_HYPOXIC_BURDEN = 75.0
FIGURE4_HIDE_YMAX_TICK_LABEL_COLS = {"ss_percent_main", "ahi"}
FIGURE4_DEMOGRAPHIC_MODEL = "bsi_age_sex_bmi_cohort"
FIGURE4_MUTUAL_MODEL = "bsi_plus_other_comparators_age_sex_bmi_cohort"
PAIRWISE_DISPLAY_QUANTILE = 0.99
COMPARATOR_ANALYSIS_COHORTS = ("S0001", "I0002", "mros")
COHORT_DISPLAY_LABELS = {"S0001": "S0001", "I0002": "I0002", "mros": "MrOS"}
BSI_MEDIAN_SLEEP_COL = "bsi_median_sleep"
BSI_MEAN_SLEEP_COL = "bsi_mean_sleep"
BSI_MEAN_SLEEP_SOURCE_COL = "bsi_robust_mean_w2_ov0p9_sleep_stability_auc"
BSI_STABILITY_OUTPUT_FS_HZ = 10.0


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


COMPARATOR_META = [
    ("ss_percent_main", "Periodic breathing (%)", "#b85c38"),
    ("ahi", "AHI (events/h)", "#356c80"),
    ("hypoxic_burden", "Hypoxic burden (%min/h)", "#5d8a3f"),
    ("arousal_index", "Arousal index (events/h)", "#6f55a4"),
]

BSI_WINDOW_META = [
    (
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_stable",
        "BSI stable 5-min windows (n)",
        "#4d7c3a",
    ),
    (
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_unstable",
        "BSI unstable 5-min windows (n)",
        "#8a4a8f",
    ),
]

FIGURE4_META = [*COMPARATOR_META, *BSI_WINDOW_META]

PAIRWISE_META = [
    ("bsi_median_sleep", "Median\nBSI", "Median\nBSI"),
    (
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_stable",
        "N 5-Minute\nWindows\nStable",
        "N 5Min\nWindows\nStable",
    ),
    (
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_unstable",
        "N 5-Minute\nWindows\nUnstable",
        "N 5Min\nWindows\nUnstable",
    ),
    ("ahi", "AHI", "Apnea-Hypopnea\nIndex"),
    ("rdi", "RDI", "Respiratory\nDisturbance\nIndex"),
    ("hypoxic_burden", "Hypoxic\nBurden", "Hypoxic\nBurden"),
]

PAIRWISE_FIXED_LIMITS = {
    "ahi": FIGURE4_DEFAULT_YMAX_AHI,
    "hypoxic_burden": FIGURE4_DEFAULT_YMAX_HYPOXIC_BURDEN,
}

EXPOSURE_LABELS = {
    "bsi_median_sleep": "BSI",
    "ss_percent_main": "SS",
    "ahi": "AHI",
    "hypoxic_burden": "Hypoxic burden",
}

MODEL_LABELS = {
    "unadjusted": "Unadjusted",
    "age_sex": "Age + sex + cohort",
    "age_sex_bmi": "Age + sex + cohort + BMI",
}

MODEL_COLORS = {
    "unadjusted": "#c56d1d",
    "age_sex": "#2f6c9f",
    "age_sex_bmi": "#3f8f5b",
}

OUTCOME_LABELS = {
    "cog_fluid": "Fluid cognition",
    "cog_crystallized": "Crystallized cognition",
    "cog_total": "Total cognition",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build revision figures 4, 5, 6, and supplemental pairwise matrix.")
    p.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--physiology-dir", type=Path, default=DEFAULT_PHYS)
    p.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--refresh-figure4-results", action="store_true", help="Recompute saved Figure 4 source/correlation/model CSVs.")
    p.add_argument("--figure4-only", action="store_true", help="Redraw only Figure 4 and its cohort-stratified supplements.")
    p.add_argument(
        "--figure4-color-quantile",
        type=float,
        default=FIGURE4_COLOR_QUANTILE,
        help="Quantile of nonzero hexbin counts used as the Figure 4 color saturation cap.",
    )
    p.add_argument(
        "--figure4-axis-quantile",
        type=float,
        default=FIGURE4_DISPLAY_QUANTILE,
        help="Quantile used for Figure 4 x/y display limits when no fixed axis limit is provided.",
    )
    p.add_argument("--figure4-xmax-bsi", type=float, default=None, help="Optional fixed Figure 4 BSI x-axis maximum.")
    p.add_argument("--figure4-xmax-bsi-mean", type=float, default=None, help="Optional fixed x-axis maximum for the mean-BSI companion Figure 4.")
    p.add_argument("--figure4-ymax-ss", type=float, default=FIGURE4_DEFAULT_YMAX_SS, help="Fixed Figure 4 periodic breathing y-axis maximum.")
    p.add_argument("--figure4-ymax-ahi", type=float, default=FIGURE4_DEFAULT_YMAX_AHI, help="Fixed Figure 4 AHI y-axis maximum.")
    p.add_argument("--figure4-ymax-hypoxic-burden", type=float, default=FIGURE4_DEFAULT_YMAX_HYPOXIC_BURDEN, help="Fixed Figure 4 hypoxic burden y-axis maximum.")
    p.add_argument("--figure4-ymax-arousal-index", type=float, default=None, help="Optional fixed Figure 4 arousal index y-axis maximum.")
    p.add_argument("--figure4-ymax-bsi-stable-windows", type=float, default=None, help="Optional fixed Figure 4 BSI stable 5-minute window count y-axis maximum.")
    p.add_argument("--figure4-ymax-bsi-unstable-windows", type=float, default=None, help="Optional fixed Figure 4 BSI unstable 5-minute window count y-axis maximum.")
    return p.parse_args()


def save_figure(
    fig: plt.Figure,
    stem: str,
    outdir: Path,
    rect: tuple[float, float, float, float] | None = None,
    extra_formats: tuple[str, ...] = (),
) -> None:
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    for fmt in extra_formats:
        fmt = fmt.lower().lstrip(".")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"jpg", "jpeg"}:
            save_kwargs["facecolor"] = "white"
            save_kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(outdir / f"{stem}.{fmt}", **save_kwargs)
    plt.close(fig)


def parse_iqr_text(text: object) -> tuple[float, float, float]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return np.nan, np.nan, np.nan
    match = re.match(r"\s*([-\d\.]+)\s*\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]\s*", str(text))
    if not match:
        return np.nan, np.nan, np.nan
    return tuple(float(match.group(i)) for i in range(1, 4))


def comparator_analysis_data(master: pd.DataFrame) -> pd.DataFrame:
    return master[master["cohort"].isin(COMPARATOR_ANALYSIS_COHORTS)].copy()


def add_derived_bsi_metrics(master: pd.DataFrame) -> pd.DataFrame:
    if BSI_MEAN_SLEEP_COL in master.columns:
        return master
    if BSI_MEAN_SLEEP_SOURCE_COL not in master.columns:
        raise KeyError(f"Missing required source column for mean BSI: {BSI_MEAN_SLEEP_SOURCE_COL}")
    master = master.copy()
    master[BSI_MEAN_SLEEP_COL] = pd.to_numeric(master[BSI_MEAN_SLEEP_SOURCE_COL], errors="coerce") / BSI_STABILITY_OUTPUT_FS_HZ
    return master


def saved_results_current(
    paths: list[Path],
    master_csv: Path,
    required_source_cols: list[str] | None = None,
    required_adjusted_models: list[str] | None = None,
) -> bool:
    if not all(path.exists() for path in paths):
        return False
    if required_source_cols is not None:
        source_path = next((path for path in paths if path.name.endswith("_source_data.csv")), None)
        if source_path is None:
            return False
        source_cols = pd.read_csv(source_path, nrows=0).columns
        if any(col not in source_cols for col in required_source_cols):
            return False
    if required_adjusted_models is not None:
        adjusted_path = next((path for path in paths if path.name.endswith("_adjusted_models.csv")), None)
        if adjusted_path is None or not adjusted_path.exists():
            return False
        adjusted_cols = pd.read_csv(adjusted_path, nrows=0).columns
        if "model_name" not in adjusted_cols:
            return False
        model_names = set(pd.read_csv(adjusted_path, usecols=["model_name"])["model_name"].dropna().unique())
        if any(model_name not in model_names for model_name in required_adjusted_models):
            return False
    if not master_csv.exists():
        return True
    master_mtime = master_csv.stat().st_mtime
    return all(path.stat().st_mtime >= master_mtime for path in paths)


def figure4_result_paths(outdir: Path, stem: str, include_adjusted: bool = True) -> dict[str, Path]:
    paths = {
        "source": outdir / f"{stem}_source_data.csv",
        "correlations": outdir / f"{stem}_correlations.csv",
    }
    if include_adjusted:
        paths["adjusted"] = outdir / f"{stem}_adjusted_models.csv"
    return paths


def figure4_master_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                "fileid",
                "cohort",
                "age",
                "sex",
                "bmi",
                BSI_MEDIAN_SLEEP_COL,
                BSI_MEAN_SLEEP_SOURCE_COL,
                *[m[0] for m in FIGURE4_META],
            ]
        )
    )


def figure4_fixed_y_limits(args: argparse.Namespace) -> dict[str, float]:
    candidates = {
        "ss_percent_main": args.figure4_ymax_ss,
        "ahi": args.figure4_ymax_ahi,
        "hypoxic_burden": args.figure4_ymax_hypoxic_burden,
        "arousal_index": args.figure4_ymax_arousal_index,
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_stable": args.figure4_ymax_bsi_stable_windows,
        "bsi_robust_mean_w2_ov0p9_sleep_n_5min_windows_unstable": args.figure4_ymax_bsi_unstable_windows,
    }
    return {col: float(value) for col, value in candidates.items() if value is not None}


def figure4_axis_quantile(axis_quantile: float) -> float:
    return min(max(float(axis_quantile), 0.0), 1.0)


def figure4_x_cap(dat: pd.DataFrame, axis_quantile: float, x_max: float | None, x_col: str = BSI_MEDIAN_SLEEP_COL) -> float:
    if x_max is not None:
        return float(x_max)
    return float(dat[x_col].quantile(figure4_axis_quantile(axis_quantile)))


def figure4_y_cap(dat: pd.DataFrame, col: str, axis_quantile: float, fixed_y_limits: dict[str, float]) -> float:
    fixed = fixed_y_limits.get(col)
    if fixed is not None:
        return fixed
    return float(dat[col].quantile(figure4_axis_quantile(axis_quantile)))


def figure4_panel_title(ylabel: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", ylabel).strip()




def cap_hexbin_color(hb: matplotlib.collections.PolyCollection, color_quantile: float) -> float | None:
    counts = np.asarray(hb.get_array(), dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if counts.size == 0:
        return None
    q = min(max(float(color_quantile), 0.0), 1.0)
    vmax = float(np.quantile(counts, q))
    if vmax <= 0:
        return None
    hb.set_clim(vmin=0, vmax=vmax)
    return vmax


def style_figure4_colorbar(cb: matplotlib.colorbar.Colorbar) -> None:
    cb.set_label("Count", fontsize=9)
    cb.ax.tick_params(labelsize=8, length=2, width=0.8)


def style_figure4_y_ticks(ax: plt.Axes, col: str, y_cap: float) -> None:
    if col not in FIGURE4_HIDE_YMAX_TICK_LABEL_COLS:
        return

    def format_tick(value: float, _: int) -> str:
        if np.isclose(value, y_cap, rtol=0, atol=max(1e-8, abs(y_cap) * 1e-8)):
            return ""
        return f"{value:g}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_tick))


def standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - values.mean()) / sd


def compute_comparator_correlations(
    data: pd.DataFrame,
    include_overall: bool = True,
    meta: list[tuple[str, str, str]] | None = None,
    x_col: str = BSI_MEDIAN_SLEEP_COL,
) -> pd.DataFrame:
    if meta is None:
        meta = FIGURE4_META
    rows = []
    cohorts = list(COMPARATOR_ANALYSIS_COHORTS)
    if include_overall:
        cohorts = ["overall", *cohorts]
    for cohort in cohorts:
        sub = data if cohort == "overall" else data[data["cohort"] == cohort]
        for col, _, _ in meta:
            dat = sub[[x_col, col]].dropna()
            rows.append(
                {
                    "cohort": cohort,
                    "bsi_exposure": x_col,
                    "comparator": col,
                    "n": int(len(dat)),
                    "spearman_rho": float(dat[x_col].corr(dat[col], method="spearman")) if len(dat) >= 3 else np.nan,
                    "pearson_r": float(dat[x_col].corr(dat[col], method="pearson")) if len(dat) >= 3 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def append_ols_rows(
    rows: list[dict[str, object]],
    model: pd.DataFrame,
    formula: str,
    model_name: str,
    outcome: str,
    x_col: str,
    terms: list[str],
) -> None:
    try:
        fit = smf.ols(formula, data=model).fit()
        ci = fit.conf_int()
    except Exception as exc:
        for term in terms:
            rows.append(
                {
                    "model_name": model_name,
                    "outcome": outcome,
                    "bsi_exposure": x_col,
                    "predictor": term,
                    "n": int(len(model)),
                    "beta_std": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "model_error": str(exc),
                }
            )
        return

    for term in terms:
        rows.append(
            {
                "model_name": model_name,
                "outcome": outcome,
                "bsi_exposure": x_col,
                "predictor": term,
                "n": int(fit.nobs),
                "beta_std": float(fit.params.get(term, np.nan)),
                "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "r_squared": float(fit.rsquared),
                "model_error": "",
            }
        )


def fit_adjusted_comparator_models(data: pd.DataFrame, x_col: str = BSI_MEDIAN_SLEEP_COL) -> pd.DataFrame:
    rows = []
    comparator_cols = [col for col, _, _ in COMPARATOR_META]
    for outcome, _, _ in COMPARATOR_META:
        demographic_needed = list(dict.fromkeys(["cohort", "age", "sex", "bmi", x_col, outcome]))
        demographic_model = data[demographic_needed].copy()
        for col in [c for c in ["age", "sex", "bmi", x_col, outcome] if c in demographic_model.columns]:
            demographic_model[col] = pd.to_numeric(demographic_model[col], errors="coerce")
        demographic_model = demographic_model.dropna()
        if len(demographic_model) >= 20:
            demographic_model["outcome_z"] = standardize(demographic_model[outcome])
            demographic_model["bsi_z"] = standardize(demographic_model[x_col])
            demographic_model["age_z"] = standardize(demographic_model["age"])
            demographic_model["bmi_z"] = standardize(demographic_model["bmi"])
            demographic_model = demographic_model.dropna()
            if len(demographic_model) >= 20:
                cohort_term = " + C(cohort)" if demographic_model["cohort"].nunique() > 1 else ""
                formula = f"outcome_z ~ bsi_z + age_z + sex + bmi_z{cohort_term}"
                append_ols_rows(rows, demographic_model, formula, FIGURE4_DEMOGRAPHIC_MODEL, outcome, x_col, ["bsi_z"])

        other_cols = [col for col in comparator_cols if col != outcome]
        mutual_needed = list(dict.fromkeys(["cohort", "age", "sex", "bmi", x_col, outcome, *other_cols]))
        mutual_model = data[mutual_needed].copy()
        for col in [c for c in ["age", "sex", "bmi", x_col, outcome, *other_cols] if c in mutual_model.columns]:
            mutual_model[col] = pd.to_numeric(mutual_model[col], errors="coerce")
        mutual_model = mutual_model.dropna()
        if len(mutual_model) < 20:
            continue

        mutual_model["outcome_z"] = standardize(mutual_model[outcome])
        mutual_model["bsi_z"] = standardize(mutual_model[x_col])
        mutual_model["age_z"] = standardize(mutual_model["age"])
        mutual_model["bmi_z"] = standardize(mutual_model["bmi"])
        other_terms = []
        for col in other_cols:
            term = f"{col}_z"
            mutual_model[term] = standardize(mutual_model[col])
            other_terms.append(term)
        mutual_model = mutual_model.dropna()
        if len(mutual_model) < 20:
            continue

        cohort_term = " + C(cohort)" if mutual_model["cohort"].nunique() > 1 else ""
        formula = f"outcome_z ~ bsi_z + {' + '.join(other_terms)} + age_z + sex + bmi_z{cohort_term}"
        append_ols_rows(rows, mutual_model, formula, FIGURE4_MUTUAL_MODEL, outcome, x_col, ["bsi_z", *other_terms])
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def smooth_quantile_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 10 or np.nanmin(x) == np.nanmax(x):
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n_bins = min(80, max(10, len(x) // 250))
    splits = np.array_split(np.arange(len(x)), n_bins)
    x_bin = np.array([np.nanmedian(x[idx]) for idx in splits if len(idx) > 0])
    y_bin = np.array([np.nanmedian(y[idx]) for idx in splits if len(idx) > 0])
    finite_bins = np.isfinite(x_bin) & np.isfinite(y_bin)
    x_bin = x_bin[finite_bins]
    y_bin = y_bin[finite_bins]
    if len(x_bin) < 4:
        return np.array([]), np.array([])

    y_smooth = pd.Series(y_bin).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    x_fit = np.linspace(float(np.nanmin(x_bin)), float(np.nanmax(x_bin)), 200)
    y_fit = np.interp(x_fit, x_bin, y_smooth)
    return x_fit, y_fit


def style_matrix_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


def build_figure4(
    master: pd.DataFrame,
    phys_dir: Path,
    outdir: Path,
    master_csv: Path,
    refresh_results: bool,
    color_quantile: float,
    axis_quantile: float,
    x_max: float | None,
    fixed_y_limits: dict[str, float],
    x_col: str = BSI_MEDIAN_SLEEP_COL,
    x_label: str = "BSI median",
    result_stem: str = "figure4",
    figure_stem: str = "figure4_comparator_physiology",
    title: str | None = "Figure 4. BSI comparator physiology",
) -> None:
    del phys_dir
    source_cols = list(dict.fromkeys(["fileid", "cohort", x_col] + [m[0] for m in FIGURE4_META]))
    paths = figure4_result_paths(outdir, result_stem)
    required_models = [FIGURE4_DEMOGRAPHIC_MODEL, FIGURE4_MUTUAL_MODEL]
    if not refresh_results and saved_results_current(
        list(paths.values()),
        master_csv,
        source_cols,
        required_adjusted_models=required_models,
    ):
        analysis = pd.read_csv(paths["source"])
        corr = pd.read_csv(paths["correlations"])
        adjusted = pd.read_csv(paths["adjusted"])
    else:
        analysis_cols = list(dict.fromkeys(["fileid", "cohort", "age", "sex", "bmi", x_col] + [m[0] for m in FIGURE4_META]))
        full_analysis = comparator_analysis_data(master[analysis_cols].copy())
        analysis = full_analysis[source_cols].copy()
        analysis.to_csv(paths["source"], index=False)
        corr = compute_comparator_correlations(full_analysis, x_col=x_col)
        corr.to_csv(paths["correlations"], index=False)
        adjusted = fit_adjusted_comparator_models(full_analysis, x_col=x_col)
        adjusted.to_csv(paths["adjusted"], index=False)

    ncols = 2
    nrows = math.ceil(len(FIGURE4_META) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.4 * nrows))
    axes = axes.ravel()

    for idx, (col, ylabel, color) in enumerate(FIGURE4_META):
        ax = axes[idx]
        dat = analysis[[x_col, col]].dropna().copy()
        x_cap = figure4_x_cap(dat, axis_quantile, x_max, x_col=x_col)
        y_cap = figure4_y_cap(dat, col, axis_quantile, fixed_y_limits)
        plot_dat = dat[(dat[x_col] <= x_cap) & (dat[col] <= y_cap)].copy()
        x = plot_dat[x_col].to_numpy()
        y = plot_dat[col].to_numpy()

        hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0)
        cap_hexbin_color(hb, color_quantile)
        x_fit, y_fit = smooth_quantile_fit(x, y)
        if len(x_fit) > 0:
            ax.plot(x_fit, y_fit, color=color, linewidth=2.0)
        cb = fig.colorbar(hb, ax=ax)
        style_figure4_colorbar(cb)

        row = corr[(corr["cohort"] == "overall") & (corr["comparator"] == col)].iloc[0]
        note = [f"n={int(row['n']):,}", f"Spearman ρ={row['spearman_rho']:.3f}"]
        if col in {item[0] for item in COMPARATOR_META}:
            adj_row = adjusted[
                (adjusted["model_name"] == FIGURE4_DEMOGRAPHIC_MODEL)
                & (adjusted["outcome"] == col)
                & (adjusted["predictor"] == "bsi_z")
            ].iloc[0]
            note.append(f"Adj β={adj_row['beta_std']:.3f}")
        ax.text(
            0.03,
            0.97,
            "\n".join(note),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none"},
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(figure4_panel_title(ylabel))
        ax.set_xlim(0, x_cap)
        ax.set_ylim(0, y_cap)
        style_figure4_y_ticks(ax, col, y_cap)
        add_panel_label(ax, chr(ord("A") + idx))

    for ax in axes[len(FIGURE4_META):]:
        ax.set_axis_off()

    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    extra_formats = ("eps", "jpg") if figure_stem == "figure4_comparator_physiology" else ()
    save_figure(fig, figure_stem, outdir, extra_formats=extra_formats)


def build_figure4_median_bsi(
    master: pd.DataFrame,
    phys_dir: Path,
    outdir: Path,
    master_csv: Path,
    refresh_results: bool,
    color_quantile: float,
    axis_quantile: float,
    x_max: float | None,
    fixed_y_limits: dict[str, float],
) -> None:
    build_figure4(
        master,
        phys_dir,
        outdir,
        master_csv,
        refresh_results,
        color_quantile,
        axis_quantile,
        x_max,
        fixed_y_limits,
        x_col=BSI_MEDIAN_SLEEP_COL,
        x_label="BSI median",
        result_stem="figureS_figure4_median_bsi",
        figure_stem="figureS_figure4_median_bsi_comparator_physiology",
        title="Supplemental Figure. Median BSI comparator physiology",
    )


def build_figure4_by_cohort(
    master: pd.DataFrame,
    outdir: Path,
    master_csv: Path,
    refresh_results: bool,
    color_quantile: float,
    axis_quantile: float,
    x_max: float | None,
    fixed_y_limits: dict[str, float],
) -> None:
    analysis_cols = ["fileid", "cohort", "bsi_median_sleep"] + [m[0] for m in FIGURE4_META]
    paths = figure4_result_paths(outdir, "figureS_comparator_physiology_by_cohort", include_adjusted=False)
    if not refresh_results and saved_results_current(list(paths.values()), master_csv, analysis_cols):
        analysis = pd.read_csv(paths["source"])
        corr = pd.read_csv(paths["correlations"])
    else:
        analysis = comparator_analysis_data(master[analysis_cols].copy())
        analysis.to_csv(paths["source"], index=False)
        corr = compute_comparator_correlations(analysis, include_overall=False)
        corr.to_csv(paths["correlations"], index=False)

    cohorts = [cohort for cohort in COMPARATOR_ANALYSIS_COHORTS if cohort in set(analysis["cohort"])]
    fig, axes = plt.subplots(len(cohorts), len(FIGURE4_META), figsize=(21, 10.5), squeeze=False)

    for row_idx, cohort in enumerate(cohorts):
        cohort_data = analysis[analysis["cohort"] == cohort]
        for col_idx, (col, ylabel, color) in enumerate(FIGURE4_META):
            ax = axes[row_idx, col_idx]
            dat = cohort_data[["bsi_median_sleep", col]].dropna().copy()
            if len(dat) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x_cap = figure4_x_cap(dat, axis_quantile, x_max)
            y_cap = figure4_y_cap(dat, col, axis_quantile, fixed_y_limits)
            plot_dat = dat[(dat["bsi_median_sleep"] <= x_cap) & (dat[col] <= y_cap)].copy()
            x = plot_dat["bsi_median_sleep"].to_numpy()
            y = plot_dat[col].to_numpy()

            hb = ax.hexbin(x, y, gridsize=32, cmap="YlOrRd", mincnt=1, linewidths=0)
            cap_hexbin_color(hb, color_quantile)
            x_fit, y_fit = smooth_quantile_fit(x, y)
            if len(x_fit) > 0:
                ax.plot(x_fit, y_fit, color=color, linewidth=1.8)

            rho = dat["bsi_median_sleep"].corr(dat[col], method="spearman")
            ax.text(
                0.03,
                0.97,
                f"n={len(dat):,}\nρ={rho:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none"},
            )
            ax.set_xlim(0, x_cap)
            ax.set_ylim(0, y_cap)
            style_figure4_y_ticks(ax, col, y_cap)
            if row_idx == 0:
                ax.set_title(figure4_panel_title(ylabel), fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{COHORT_DISPLAY_LABELS.get(cohort, cohort)}\n{ylabel}", fontsize=9)
            else:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_idx == len(cohorts) - 1:
                ax.set_xlabel("BSI median", fontsize=9)
            else:
                ax.set_xlabel("")

    fig.suptitle("Supplemental Figure. Median BSI comparator physiology by cohort", y=1.01, fontsize=14)
    save_figure(fig, "figureS_comparator_physiology_by_cohort", outdir)


def build_figure4_cohort_strata(
    master: pd.DataFrame,
    outdir: Path,
    master_csv: Path,
    refresh_results: bool,
    color_quantile: float,
    axis_quantile: float,
    x_max: float | None,
    fixed_y_limits: dict[str, float],
) -> None:
    analysis_cols = ["fileid", "cohort", "age", "sex", "bmi", "bsi_median_sleep"] + [m[0] for m in FIGURE4_META]
    analysis = comparator_analysis_data(master[analysis_cols].copy())
    source_cols = ["fileid", "cohort", "bsi_median_sleep"] + [m[0] for m in FIGURE4_META]

    for cohort in COMPARATOR_ANALYSIS_COHORTS:
        stem = f"figureS_figure4_comparator_physiology_{cohort.lower()}"
        paths = figure4_result_paths(outdir, stem)
        required_models = [FIGURE4_DEMOGRAPHIC_MODEL, FIGURE4_MUTUAL_MODEL]
        if not refresh_results and saved_results_current(
            list(paths.values()),
            master_csv,
            source_cols,
            required_adjusted_models=required_models,
        ):
            cohort_data = pd.read_csv(paths["source"])
            corr = pd.read_csv(paths["correlations"])
            adjusted = pd.read_csv(paths["adjusted"])
        else:
            full_cohort_data = analysis[analysis["cohort"] == cohort].copy()
            if full_cohort_data.empty:
                continue
            cohort_data = full_cohort_data[source_cols].copy()
            cohort_data.to_csv(paths["source"], index=False)
            corr = compute_comparator_correlations(full_cohort_data, include_overall=False)
            corr = corr[corr["cohort"] == cohort].copy()
            corr.to_csv(paths["correlations"], index=False)
            adjusted = fit_adjusted_comparator_models(full_cohort_data)
            adjusted.to_csv(paths["adjusted"], index=False)

        if cohort_data.empty:
            continue

        ncols = 2
        nrows = math.ceil(len(FIGURE4_META) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.4 * nrows))
        axes = axes.ravel()
        for idx, (col, ylabel, color) in enumerate(FIGURE4_META):
            ax = axes[idx]
            dat = cohort_data[["bsi_median_sleep", col]].dropna().copy()
            if len(dat) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x_cap = figure4_x_cap(dat, axis_quantile, x_max)
            y_cap = figure4_y_cap(dat, col, axis_quantile, fixed_y_limits)
            plot_dat = dat[(dat["bsi_median_sleep"] <= x_cap) & (dat[col] <= y_cap)].copy()
            x = plot_dat["bsi_median_sleep"].to_numpy()
            y = plot_dat[col].to_numpy()

            hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0)
            cap_hexbin_color(hb, color_quantile)
            x_fit, y_fit = smooth_quantile_fit(x, y)
            if len(x_fit) > 0:
                ax.plot(x_fit, y_fit, color=color, linewidth=2.0)
            cb = fig.colorbar(hb, ax=ax)
            style_figure4_colorbar(cb)

            row = corr[(corr["cohort"] == cohort) & (corr["comparator"] == col)].iloc[0]
            note = [f"n={int(row['n']):,}", f"Spearman ρ={row['spearman_rho']:.3f}"]
            if col in {item[0] for item in COMPARATOR_META} and not adjusted.empty:
                adj = adjusted[
                    (adjusted["model_name"] == FIGURE4_DEMOGRAPHIC_MODEL)
                    & (adjusted["outcome"] == col)
                    & (adjusted["predictor"] == "bsi_z")
                ]
                if not adj.empty and pd.notna(adj.iloc[0]["beta_std"]):
                    note.append(f"Adj β={adj.iloc[0]['beta_std']:.3f}")
            ax.text(
                0.03,
                0.97,
                "\n".join(note),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none"},
            )

            ax.set_xlabel("BSI median")
            ax.set_ylabel(ylabel)
            ax.set_title(figure4_panel_title(ylabel))
            ax.set_xlim(0, x_cap)
            ax.set_ylim(0, y_cap)
            style_figure4_y_ticks(ax, col, y_cap)
            add_panel_label(ax, chr(ord("A") + idx))

        for ax in axes[len(FIGURE4_META):]:
            ax.set_axis_off()

        cohort_label = COHORT_DISPLAY_LABELS.get(cohort, cohort)
        fig.suptitle(f"Supplemental Figure. Median BSI comparator physiology: {cohort_label}", y=1.02, fontsize=14)
        save_figure(fig, stem, outdir)


def build_pairwise_scatter_matrix(
    master: pd.DataFrame,
    outdir: Path,
    stem: str = "figureS_bsi_pairwise_scatter_matrix",
    title: str = "Supplemental Figure. Pairwise BSI and respiratory comparator relationships",
) -> None:
    source_cols = ["fileid", "cohort"] + [col for col, _, _ in PAIRWISE_META]
    missing = [col for col in source_cols if col not in master.columns]
    if missing:
        raise KeyError(f"Missing required pairwise source columns: {missing}")

    source = master[source_cols].copy()
    rename_map = {col: diag_label.replace("\n", " ") for col, diag_label, _ in PAIRWISE_META}
    source.rename(columns=rename_map).to_csv(outdir / f"{stem}_source_data.csv", index=False)

    plot_cols = [col for col, _, _ in PAIRWISE_META]
    corr_rows = []
    for i, x_col in enumerate(plot_cols):
        for j, y_col in enumerate(plot_cols):
            if j <= i:
                continue
            dat = master[[x_col, y_col]].dropna()
            corr_rows.append(
                {
                    "x": rename_map[x_col],
                    "y": rename_map[y_col],
                    "n": int(len(dat)),
                    "spearman_rho": float(dat[x_col].corr(dat[y_col], method="spearman")),
                    "pearson_r": float(dat[x_col].corr(dat[y_col], method="pearson")),
                }
            )
    pd.DataFrame(corr_rows).to_csv(outdir / f"{stem}_correlations.csv", index=False)

    limits = {}
    for col in plot_cols:
        if col in PAIRWISE_FIXED_LIMITS:
            limits[col] = (0.0, float(PAIRWISE_FIXED_LIMITS[col]))
        else:
            values = master[col].dropna()
            limits[col] = (0.0, float(values.quantile(PAIRWISE_DISPLAY_QUANTILE)))

    n = len(plot_cols)
    fig, axes = plt.subplots(n, n, figsize=(8.2, 8.2))
    for i, (y_col, diag_label, y_label) in enumerate(PAIRWISE_META):
        for j, (x_col, _, x_label) in enumerate(PAIRWISE_META):
            ax = axes[i, j]
            style_matrix_axis(ax)

            if i == j:
                ax.text(
                    0.5,
                    0.5,
                    diag_label,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    alpha=0.85,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                dat = master[[x_col, y_col]].dropna()
                x_min, x_max = limits[x_col]
                y_min, y_max = limits[y_col]
                plot_dat = dat[
                    (dat[x_col] >= x_min)
                    & (dat[x_col] <= x_max)
                    & (dat[y_col] >= y_min)
                    & (dat[y_col] <= y_max)
                ]
                x = plot_dat[x_col].to_numpy()
                y = plot_dat[y_col].to_numpy()
                ax.scatter(x, y, color="black", alpha=0.055, s=1.2, linewidths=0, rasterized=True)
                x_fit, y_fit = smooth_quantile_fit(x, y)
                if len(x_fit) > 0:
                    ax.plot(x_fit, y_fit, color="cornflowerblue", linewidth=1.1, alpha=1.0)
                rho = dat[x_col].corr(dat[y_col], method="spearman")
                ax.text(
                    0.05,
                    0.95,
                    f"ρ = {rho:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="red",
                    va="top",
                    ha="left",
                    fontweight="bold",
                )
                ax.set_xlim(limits[x_col])
                ax.set_ylim(limits[y_col])
                ax.set_xticks([])
                ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(y_label, fontsize=8)
            if i == n - 1:
                ax.set_xlabel(x_label, fontsize=8)

    handles = [plt.Line2D([0], [0], color="cornflowerblue", lw=1.2, label="Smoothed fit")]
    axes[-1, 0].legend(handles=handles, fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1, -0.42))
    fig.suptitle(title, y=1.01, fontsize=12)
    save_figure(fig, stem, outdir)


def build_pairwise_scatter_matrices_by_cohort(master: pd.DataFrame, outdir: Path) -> None:
    for cohort in COMPARATOR_ANALYSIS_COHORTS:
        sub = master[master["cohort"] == cohort].copy()
        if sub.empty:
            continue
        stem = f"figureS_bsi_pairwise_scatter_matrix_{cohort.lower()}"
        title = f"Supplemental Figure. Pairwise BSI and respiratory comparators: {COHORT_DISPLAY_LABELS.get(cohort, cohort)}"
        build_pairwise_scatter_matrix(sub, outdir, stem=stem, title=title)


def build_figure5(master: pd.DataFrame, phys_dir: Path, outdir: Path) -> None:
    stage_desc = pd.read_csv(phys_dir / "stage_descriptives.csv")
    stage_results = pd.read_csv(phys_dir / "stage_model_results.csv")
    rem_summary = pd.read_csv(phys_dir / "rem_nrem_summary.csv")
    rem_pred = pd.read_csv(phys_dir / "rem_predominant_instability_summary.csv")

    rem_col = "bsi_robust_mean_w2_ov0p9_rem_median_stability"
    nrem_col = "bsi_robust_mean_w2_ov0p9_nrem_median_stability"
    raw_rem = master[["fileid", "cohort", rem_col, nrem_col]].copy()
    raw_rem["rem_minus_nrem"] = raw_rem[rem_col] - raw_rem[nrem_col]
    raw_rem.to_csv(outdir / "figure5_rem_minus_nrem_source_data.csv", index=False)
    stage_desc.to_csv(outdir / "figure5_stage_descriptives_source.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel A: stage medians and IQR
    ax = axes[0]
    overall_stage = stage_desc[stage_desc["cohort"] == "overall"].copy()
    order = ["N1", "N2", "N3", "REM"]
    overall_stage["stage"] = pd.Categorical(overall_stage["stage"], categories=order, ordered=True)
    overall_stage = overall_stage.sort_values("stage")
    meds, q1s, q3s = zip(*overall_stage["bsi_median_iqr"].map(parse_iqr_text))
    x = np.arange(len(order))
    ax.errorbar(
        x,
        meds,
        yerr=[np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)],
        fmt="o-",
        color="#356c80",
        ecolor="#356c80",
        capsize=4,
        linewidth=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("BSI median")
    ax.set_title("Stage-specific BSI")
    cat_rows = stage_results[stage_results["model_name"] == "categorical_stage_model"].copy()
    coef_map = {
        "C(stage, Treatment(reference='N2'))[T.N1]": "N1 vs N2",
        "C(stage, Treatment(reference='N2'))[T.N3]": "N3 vs N2",
        "C(stage, Treatment(reference='N2'))[T.REM]": "REM vs N2",
    }
    note_rows = cat_rows[cat_rows["term"].isin(coef_map)].copy()
    note = []
    for _, row in note_rows.iterrows():
        note.append(f"{coef_map[row['term']]}: {row['coef']:+.3f}")
    ax.text(
        0.03,
        0.97,
        "\n".join(note),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    add_panel_label(ax, "A")

    # Panel B: cohort-specific REM-NREM effect
    ax = axes[1]
    rem_plot = rem_summary.copy()
    rem_plot["cohort_display"] = rem_plot["cohort"].replace({"overall": "Overall", "S0001": "S0001", "I0002": "I0002", "mros": "MrOS", "mgh-cog": "MGH-cog"})
    rem_order = ["Overall", "S0001", "I0002", "MrOS", "MGH-cog"]
    rem_plot["cohort_display"] = pd.Categorical(rem_plot["cohort_display"], categories=rem_order, ordered=True)
    rem_plot = rem_plot.sort_values("cohort_display")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    y = np.arange(len(rem_plot))
    ax.scatter(rem_plot["median_rem_minus_nrem"], y, color="#b85c38", s=70, zorder=3)
    for i, (_, row) in enumerate(rem_plot.iterrows()):
        ax.text(row["median_rem_minus_nrem"] + 0.01, i, f"n={int(row['n_paired']):,}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(rem_plot["cohort_display"])
    ax.set_xlabel("Median REM - NREM BSI")
    ax.set_title("REM vs NREM by cohort")
    add_panel_label(ax, "B")

    # Panel C: REM-predominant instability distribution
    ax = axes[2]
    rem_vals = raw_rem["rem_minus_nrem"].dropna()
    thr = float(rem_pred.loc[rem_pred["cohort"] == "overall", "threshold_rem_minus_nrem_q75"].iloc[0])
    ax.hist(rem_vals, bins=50, color="#6f55a4", alpha=0.85)
    ax.axvline(thr, color="#b22222", linestyle="--", linewidth=2)
    ax.text(
        0.03,
        0.97,
        f"Top quartile threshold = {thr:.3f}\nREM-predominant = 25.0%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set_xlabel("REM - NREM BSI")
    ax.set_ylabel("PSGs")
    ax.set_title("REM-predominant instability")
    add_panel_label(ax, "C")

    fig.suptitle("Figure 5. Stage effects and REM-predominant instability", y=1.03, fontsize=14)
    save_figure(fig, "figure5_stage_rem_instability", outdir)


def build_figure6(phase1_dir: Path, outdir: Path) -> None:
    cog = pd.read_csv(phase1_dir / "cognition_model_results.csv")
    source = cog[cog["exposure"].isin(EXPOSURE_LABELS) & cog["outcome"].isin(OUTCOME_LABELS)].copy()
    source.to_csv(outdir / "figure6_cognition_source_data.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    exposures = ["bsi_median_sleep", "ss_percent_main", "ahi", "hypoxic_burden"]
    models = ["unadjusted", "age_sex", "age_sex_bmi"]
    offsets = {"unadjusted": -0.22, "age_sex": 0.0, "age_sex_bmi": 0.22}

    max_ci = 0.0
    for _, row in source.iterrows():
        max_ci = max(max_ci, abs(row["ci_lower"]), abs(row["ci_upper"]))
    xlim = max(0.12, math.ceil(max_ci * 100) / 100 + 0.01)

    for idx, outcome in enumerate(["cog_fluid", "cog_crystallized", "cog_total"]):
        ax = axes[idx]
        sub = source[source["outcome"] == outcome].copy()
        y_positions = np.arange(len(exposures))[::-1]
        base_map = {exp: y for exp, y in zip(exposures, y_positions)}

        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        for model in models:
            mod = sub[sub["model"] == model].copy()
            for _, row in mod.iterrows():
                y = base_map[row["exposure"]] + offsets[model]
                ax.errorbar(
                    row["beta_std"],
                    y,
                    xerr=[[row["beta_std"] - row["ci_lower"]], [row["ci_upper"] - row["beta_std"]]],
                    fmt="o",
                    color=MODEL_COLORS[model],
                    ecolor=MODEL_COLORS[model],
                    capsize=3,
                    markersize=6,
                )
        ax.set_yticks(list(base_map.values()))
        ax.set_yticklabels([EXPOSURE_LABELS[e] for e in exposures])
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.set_xlabel("Standardized beta")
        ax.set_xlim(-xlim, xlim)
        add_panel_label(ax, chr(ord("A") + idx))

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in models
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Figure 6. Cognition associations and demographic attenuation", y=1.03, fontsize=14)
    save_figure(fig, "figure6_cognition_models", outdir, rect=(0, 0.08, 1, 1))


def write_summary(outdir: Path) -> None:
    lines = [
        "# Revision Figures v2",
        "",
        "Built figures:",
        "- `figure4_comparator_physiology.png/.pdf/.eps/.jpg`",
        "- `figureS_figure4_median_bsi_comparator_physiology.png/.pdf`",
        "- `figureS_comparator_physiology_by_cohort.png/.pdf`",
        "- `figureS_figure4_comparator_physiology_s0001.png/.pdf`",
        "- `figureS_figure4_comparator_physiology_i0002.png/.pdf`",
        "- `figureS_figure4_comparator_physiology_mros.png/.pdf`",
        "- `figureS_bsi_pairwise_scatter_matrix.png/.pdf`",
        "- `figureS_bsi_pairwise_scatter_matrix_s0001.png/.pdf`",
        "- `figureS_bsi_pairwise_scatter_matrix_i0002.png/.pdf`",
        "- `figureS_bsi_pairwise_scatter_matrix_mros.png/.pdf`",
        "- `figure5_stage_rem_instability.png/.pdf`",
        "- `figure6_cognition_models.png/.pdf`",
        "",
        "Source data exports:",
        "- `figure4_source_data.csv`",
        "- `figure4_correlations.csv`",
        "- `figure4_adjusted_models.csv`",
        "- `figureS_figure4_median_bsi_source_data.csv`",
        "- `figureS_figure4_median_bsi_correlations.csv`",
        "- `figureS_figure4_median_bsi_adjusted_models.csv`",
        "- `figureS_comparator_physiology_by_cohort_source_data.csv`",
        "- `figureS_comparator_physiology_by_cohort_correlations.csv`",
        "- cohort-stratified Figure 4-style source, correlation, and adjusted-model CSVs using the same stem as each stratum figure",
        "- `figureS_bsi_pairwise_scatter_matrix_source_data.csv`",
        "- `figureS_bsi_pairwise_scatter_matrix_correlations.csv`",
        "- cohort-specific pairwise source and correlation CSVs using the same stem as each cohort figure",
        "- `figure5_stage_descriptives_source.csv`",
        "- `figure5_rem_minus_nrem_source_data.csv`",
        "- `figure6_cognition_source_data.csv`",
        "",
        "Notes:",
        f"- Main Figure 4 uses derived mean BSI (`{BSI_MEAN_SLEEP_COL}`), computed as `{BSI_MEAN_SLEEP_SOURCE_COL}` / {BSI_STABILITY_OUTPUT_FS_HZ:g}; the supplemental median-BSI Figure 4 and cohort supplements use `{BSI_MEDIAN_SLEEP_COL}`.",
        f"- Figure 4-style plots exclude MGH-cog because arousals are not scored in that cohort; smoothed nonlinear overlays are shown, x-axes are clipped at the {int(FIGURE4_DISPLAY_QUANTILE * 100)}th percentile by default, and hexbin colors saturate at the {int(FIGURE4_COLOR_QUANTILE * 100)}th percentile of nonzero bin counts by default.",
        f"- Figure 4 y-axis limits default to periodic breathing <= {FIGURE4_DEFAULT_YMAX_SS:g}, AHI <= {FIGURE4_DEFAULT_YMAX_AHI:g}, and hypoxic burden <= {FIGURE4_DEFAULT_YMAX_HYPOXIC_BURDEN:g}; arousal index and BSI window-count panels still use the display quantile unless fixed limits are supplied.",
        "- Figure 4 adjusted beta annotations show standardized BSI coefficients from models adjusted for age, sex, BMI, and cohort; mutually adjusted comparator models are also exported in the adjusted-model CSVs.",
        "- Figure 4 includes an added BSI-derived row for stable and unstable 5-minute window counts; these panels are labeled as BSI window counts and do not receive adjusted external-comparator beta annotations.",
        "- Figure 4 source, correlation, and adjusted-model CSVs are reused when they are newer than the master table; pass `--refresh-figure4-results` to recompute them.",
        "- For faster Figure 4 redraws only, run `python3 REVISION/build_revision_figures.py --figure4-only`; use `--figure4-color-quantile`, `--figure4-axis-quantile`, and fixed axis flags such as `--figure4-ymax-arousal-index 40` or `--figure4-xmax-bsi-mean 25` to try display alternatives.",
        f"- The supplemental pairwise matrices mirror the submitted scatter-matrix style, exclude MGH-cog for consistency with comparator physiology, fix AHI display axes at <= {FIGURE4_DEFAULT_YMAX_AHI:g} and hypoxic burden at <= {FIGURE4_DEFAULT_YMAX_HYPOXIC_BURDEN:g}, and clip other display axes at the {int(PAIRWISE_DISPLAY_QUANTILE * 100)}th percentile; correlations are Spearman rho, with Pearson r also exported.",
        "- Figure 5 integrates stage summaries, cohort REM-NREM contrasts, and the REM-predominant threshold distribution.",
        "- Figure 6 is cognition-only and emphasizes attenuation across adjustment models.",
        "- These are first-pass revision figures and can be restyled later without rerunning upstream analyses.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.figure4_only:
        master = pd.read_csv(args.master_csv, usecols=figure4_master_columns(), low_memory=False)
    else:
        master = pd.read_csv(args.master_csv, low_memory=False)
    master = add_derived_bsi_metrics(master)
    fixed_y_limits = figure4_fixed_y_limits(args)
    build_figure4(
        master,
        args.physiology_dir,
        args.output_dir,
        args.master_csv,
        args.refresh_figure4_results,
        args.figure4_color_quantile,
        args.figure4_axis_quantile,
        args.figure4_xmax_bsi_mean,
        fixed_y_limits,
        x_col=BSI_MEAN_SLEEP_COL,
        x_label="BSI mean",
        title=None,
    )
    build_figure4_median_bsi(
        master,
        args.physiology_dir,
        args.output_dir,
        args.master_csv,
        args.refresh_figure4_results,
        args.figure4_color_quantile,
        args.figure4_axis_quantile,
        args.figure4_xmax_bsi,
        fixed_y_limits,
    )
    build_figure4_by_cohort(
        master,
        args.output_dir,
        args.master_csv,
        args.refresh_figure4_results,
        args.figure4_color_quantile,
        args.figure4_axis_quantile,
        args.figure4_xmax_bsi,
        fixed_y_limits,
    )
    build_figure4_cohort_strata(
        master,
        args.output_dir,
        args.master_csv,
        args.refresh_figure4_results,
        args.figure4_color_quantile,
        args.figure4_axis_quantile,
        args.figure4_xmax_bsi,
        fixed_y_limits,
    )
    if args.figure4_only:
        write_summary(args.output_dir)
        print(f"Wrote Figure 4 figures to {args.output_dir}")
        return 0
    comparator_master = comparator_analysis_data(master)
    build_pairwise_scatter_matrix(comparator_master, args.output_dir)
    build_pairwise_scatter_matrices_by_cohort(comparator_master, args.output_dir)
    build_figure5(master, args.physiology_dir, args.output_dir)
    build_figure6(args.phase1_dir, args.output_dir)
    write_summary(args.output_dir)
    print(f"Wrote revision figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
