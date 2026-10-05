#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
DEFAULT_SELECTION = REPO_ROOT / "REVISION" / "mortality_v2" / "selection_one_row_per_subject.csv"
DEFAULT_OUTDIR = REPO_ROOT / "REVISION" / "mortality_bsi_score_cv_v1"

DEFAULT_SELECTION_MODE = "strict_primary"
COHORTS = {"I0002": "BIDMC", "mros": "MrOS"}
RIDGE_ALPHA = 0.01

COMPARATORS = {
    "ss_percent_main": "SS",
    "ahi": "AHI",
    "hypoxic_burden": "HB",
    "arousal_index": "ArI",
}

MODEL_SPECS = {
    "m1_unadjusted": {
        "label": "Model 1: unadjusted",
        "short_label": "M1",
        "covariates": [],
    },
    "m2_age_sex": {
        "label": "Model 2: age + sex",
        "short_label": "M2",
        "covariates": ["age", "sex"],
    },
    "m3_age_sex_ahi": {
        "label": "Model 3: age + sex + AHI",
        "short_label": "M3",
        "covariates": ["age", "sex", "ahi"],
    },
    "m4_age_sex_comparators": {
        "label": "Model 4: age + sex + SS + AHI + HB + ArI",
        "short_label": "M4",
        "covariates": ["age", "sex", *COMPARATORS.keys()],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-validated BSI mortality scores by cohort.")
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument(
        "--feature-set",
        action="append",
        dest="feature_sets",
        default=None,
        help="Feature set to run. Can be repeated. Defaults to all pre-specified sets.",
    )
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return (values - values.mean()) / std


def stage_bsi_features(columns: list[str], stages: list[str], include_auc_median: bool) -> list[str]:
    features: list[str] = []
    for stage in stages:
        if include_auc_median:
            features.extend(
                [
                    col
                    for col in [
                        f"bsi_robust_mean_w2_ov0p9_{stage}_stability_auc",
                        f"bsi_robust_mean_w2_ov0p9_{stage}_median_stability",
                    ]
                    if col in columns
                ]
            )
        features.extend(
            [col for col in columns if col.startswith(f"bsi_robust_mean_w2_ov0p9_{stage}_quantile_")]
        )
        features.extend(
            [col for col in columns if col.startswith(f"bsi_robust_mean_w2_ov0p9_{stage}_n_5min_windows_")]
        )
    return list(dict.fromkeys(features))


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    return {
        "sleep_summary": stage_bsi_features(columns, ["sleep"], include_auc_median=True),
        "sleep_nrem_rem_summary": stage_bsi_features(columns, ["sleep", "nrem", "rem"], include_auc_median=True),
        "original_style_quantile_windows": stage_bsi_features(
            columns, ["sleep", "nrem", "rem"], include_auc_median=False
        ),
    }


def make_folds(n: int, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(max(2, int(n_splits)), n)
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        val_idx = np.sort(indices[start:stop])
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
        folds.append((train_idx, val_idx))
        start = stop
    return folds


def standardize_train_val(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: list[str],
    *,
    impute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    kept: list[str] = []
    train_out = pd.DataFrame(index=train.index)
    val_out = pd.DataFrame(index=val.index)
    for col in columns:
        tr = safe_numeric(train[col])
        va = safe_numeric(val[col])
        mean = tr.mean()
        std = tr.std(ddof=0)
        if pd.isna(std) or math.isclose(float(std), 0.0):
            continue
        if impute:
            tr = tr.fillna(mean)
            va = va.fillna(mean)
        train_out[col] = (tr - mean) / std
        val_out[col] = (va - mean) / std
        kept.append(col)
    return train_out, val_out, kept


def prepare_covariates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_out = pd.DataFrame(index=train.index)
    val_out = pd.DataFrame(index=val.index)
    used: list[str] = []
    dropped_constant: list[str] = []
    for covariate in covariates:
        tr = safe_numeric(train[covariate])
        va = safe_numeric(val[covariate])
        if tr.dropna().nunique() <= 1:
            dropped_constant.append(covariate)
            continue
        if covariate == "sex":
            train_out[covariate] = tr
            val_out[covariate] = va
        else:
            mean = tr.mean()
            std = tr.std(ddof=0)
            if pd.isna(std) or math.isclose(float(std), 0.0):
                dropped_constant.append(covariate)
                continue
            train_out[covariate] = (tr - mean) / std
            val_out[covariate] = (va - mean) / std
        used.append(covariate)
    return train_out, val_out, used, dropped_constant


def fit_cox_ridge(X: np.ndarray, time: np.ndarray, event: np.ndarray, alpha: float) -> tuple[np.ndarray, bool, str, int]:
    order = np.argsort(-time, kind="mergesort")
    X = np.asarray(X, dtype="float64")[order]
    time = np.asarray(time, dtype="float64")[order]
    event = np.asarray(event, dtype="float64")[order]
    _, starts = np.unique(time, return_index=True)
    starts = np.sort(starts)
    ends = np.r_[starts[1:], len(time)]
    event_groups = []
    for start, end in zip(starts, ends):
        ev_mask = event[start:end].astype(bool)
        n_events = float(event[start:end].sum())
        if n_events > 0:
            event_groups.append((start, end, ev_mask, n_events))

    p = X.shape[1]
    total_events = max(float(event.sum()), 1.0)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(X @ beta, -50.0, 50.0)
        weights = np.exp(eta)
        cum_weights = np.cumsum(weights)
        cum_weighted_x = np.cumsum(weights[:, None] * X, axis=0)
        loglik = 0.0
        grad = np.zeros(p)
        for start, end, ev_mask, n_events in event_groups:
            risk_sum = cum_weights[end - 1]
            risk_mean = cum_weighted_x[end - 1] / risk_sum
            loglik += eta[start:end][ev_mask].sum() - n_events * np.log(risk_sum)
            grad += X[start:end][ev_mask].sum(axis=0) - n_events * risk_mean
        neg = -loglik / total_events + 0.5 * alpha * float(beta @ beta)
        gradient = -grad / total_events + alpha * beta
        return neg, gradient

    result = minimize(
        lambda beta: objective(beta),
        np.zeros(p),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-7, "gtol": 1e-5, "maxls": 30},
    )
    return result.x, bool(result.success), str(result.message), int(result.nit)


def build_oof_score(
    frame: pd.DataFrame,
    feature_cols: list[str],
    covariates: list[str],
    n_splits: int,
    random_state: int,
    ridge_alpha: float,
) -> tuple[pd.Series, pd.DataFrame]:
    oof = pd.Series(np.nan, index=frame.index, dtype="float64")
    coef_rows: list[dict[str, object]] = []
    folds = make_folds(len(frame), n_splits, random_state)
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train = frame.iloc[train_idx].copy()
        val = frame.iloc[val_idx].copy()

        x_cov_tr, x_cov_va, used_covariates, dropped_constant = prepare_covariates(train, val, covariates)
        x_bsi_tr, x_bsi_va, kept_features = standardize_train_val(train, val, feature_cols, impute=True)
        x_tr = pd.concat([x_cov_tr, x_bsi_tr], axis=1)
        x_va = pd.concat([x_cov_va, x_bsi_va], axis=1)

        keep_train = train[["followup_days", "event"]].notna().all(axis=1) & x_tr.notna().all(axis=1)
        x_tr = x_tr.loc[keep_train]
        train_fit = train.loc[keep_train]
        beta, ok, message, n_iter = fit_cox_ridge(
            x_tr.to_numpy(),
            train_fit["followup_days"].to_numpy(),
            train_fit["event"].to_numpy(),
            alpha=ridge_alpha,
        )
        beta_series = pd.Series(beta, index=x_tr.columns)
        bsi_beta = beta_series.loc[kept_features]
        score = x_bsi_va[kept_features].to_numpy() @ bsi_beta.to_numpy()
        oof.iloc[val_idx] = score
        for feature, value in bsi_beta.items():
            coef_rows.append(
                {
                    "fold": fold_id,
                    "feature": feature,
                    "coef": float(value),
                    "fit_success": ok,
                    "fit_message": message,
                    "n_iter": n_iter,
                    "n_train": int(len(train_fit)),
                    "n_train_events": int(train_fit["event"].sum()),
                    "used_covariates": ", ".join(used_covariates),
                    "dropped_constant_covariates": ", ".join(dropped_constant),
                }
            )

    oof_mean = oof.mean()
    oof_std = oof.std(ddof=0)
    if pd.notna(oof_std) and not math.isclose(float(oof_std), 0.0):
        oof = (oof - oof_mean) / oof_std
    return oof, pd.DataFrame(coef_rows)


def final_phreg(frame: pd.DataFrame, covariates: list[str]) -> dict[str, object]:
    dat = frame[["followup_days", "event", "bsi_score_oof", *covariates]].copy()
    dropped_constant: list[str] = []
    exog = pd.DataFrame(index=dat.index)
    exog["bsi_score_oof"] = dat["bsi_score_oof"]
    used_covariates: list[str] = []
    for covariate in covariates:
        values = safe_numeric(dat[covariate])
        if values.dropna().nunique() <= 1:
            dropped_constant.append(covariate)
            continue
        if covariate == "sex":
            exog[covariate] = values
        else:
            exog[covariate] = zscore(values)
        used_covariates.append(covariate)

    keep = dat[["followup_days", "event"]].notna().all(axis=1) & exog.notna().all(axis=1)
    dat = dat.loc[keep]
    exog = exog.loc[keep]
    result = {
        "n": int(len(dat)),
        "n_events": int(dat["event"].sum()) if len(dat) else 0,
        "used_covariates": ", ".join(used_covariates),
        "dropped_constant_covariates": ", ".join(dropped_constant),
        "status": "not_run",
        "status_detail": "",
        "hr_per_sd": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "ph_spearman_rho_log_time": np.nan,
        "ph_p_value": np.nan,
        "concordance_score_only": np.nan,
    }
    if len(dat) < 50 or int(dat["event"].sum()) < 10:
        result["status_detail"] = "Too few rows/events."
        return result
    try:
        fit = PHReg(dat["followup_days"], exog, status=dat["event"], ties="breslow").fit(disp=0)
        idx = list(exog.columns).index("bsi_score_oof")
        ci = fit.conf_int()[idx]
        result["status"] = "ok"
        result["hr_per_sd"] = float(np.exp(fit.params[idx]))
        result["ci_lower"] = float(np.exp(ci[0]))
        result["ci_upper"] = float(np.exp(ci[1]))
        result["p_value"] = float(fit.pvalues[idx])

        residuals = np.asarray(fit.schoenfeld_residuals)
        resid = pd.Series(residuals[:, idx], index=dat.index)
        mask = dat["event"].eq(1) & resid.notna() & dat["followup_days"].gt(0)
        if int(mask.sum()) >= 10:
            rho, ph_p = stats.spearmanr(np.log(dat.loc[mask, "followup_days"]), resid.loc[mask])
            result["ph_spearman_rho_log_time"] = float(rho)
            result["ph_p_value"] = float(ph_p)
        result["concordance_score_only"] = harrell_c(dat["followup_days"], dat["event"], dat["bsi_score_oof"])
    except Exception as exc:
        result["status"] = "fit_error"
        result["status_detail"] = f"{type(exc).__name__}: {exc}"
    return result


def harrell_c(time: pd.Series, event: pd.Series, score: pd.Series) -> float:
    t = time.to_numpy(dtype="float64")
    e = event.to_numpy(dtype="float64")
    s = score.to_numpy(dtype="float64")
    concordant = 0.0
    comparable = 0.0
    n = len(t)
    for i in range(n):
        if not e[i]:
            continue
        mask = t[i] < t
        if not np.any(mask):
            continue
        comparable += float(mask.sum())
        concordant += float((s[i] > s[mask]).sum())
        concordant += 0.5 * float((s[i] == s[mask]).sum())
    return float(concordant / comparable) if comparable > 0 else np.nan


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def fmt_hr(row: pd.Series) -> str:
    if row["status"] != "ok":
        return "not estimable"
    return f"{row['hr_per_sd']:.2f} ({row['ci_lower']:.2f}-{row['ci_upper']:.2f}), p={fmt_p(row['p_value'])}"


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_markdown(outdir: Path, results: pd.DataFrame, original_note: list[str]) -> None:
    lines = [
        "# Cross-Validated BSI Mortality Score v1",
        "",
        *original_note,
        "",
        "Current revision results use the requested one-row-per-subject selection. HRs are per 1 SD of the out-of-fold BSI score.",
        "",
    ]
    for feature_set, fs_df in results.groupby("feature_set", sort=False):
        lines.extend([f"## {feature_set}", ""])
        for cohort_label, cohort_df in fs_df.groupby("cohort_label", sort=False):
            row = {"Cohort": cohort_label, "n": int(cohort_df["n"].iloc[0]), "events": int(cohort_df["n_events"].iloc[0])}
            for model_name in MODEL_SPECS:
                model_row = cohort_df[cohort_df["model_name"].eq(model_name)].iloc[0]
                row[MODEL_SPECS[model_name]["short_label"]] = fmt_hr(model_row)
            lines.extend([markdown_table(pd.DataFrame([row])), ""])
    (outdir / "bsi_score_cv_results.md").write_text("\n".join(lines).rstrip() + "\n")


def write_note(outdir: Path, args: argparse.Namespace, results: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    lines = [
        "# BSI Mortality Score CV v1 Note",
        "",
        "## Original manuscript reference",
        "",
        "- Original Figure 7 caption reported unadjusted BSI-score HRs: MrOS 1.13 (1.08-1.18), MGH 1.60 (1.44-1.76), BIDMC 1.20 (1.11-1.30).",
        "- Original age-adjusted BSI-score HRs: MrOS 1.04 (1.00-1.09), MGH 1.23 (1.12-1.36), BIDMC 1.16 (1.07-1.25).",
        "- The original code used a two-stage out-of-fold Cox score from BSI quantile and 5-minute window features.",
        "",
        "## Current implementation",
        "",
        f"- Master: `{args.master_csv}`",
        f"- Selection: `{args.selection_csv}`",
        f"- Selection mode: `{args.selection_mode}`",
        f"- Folds: `{args.n_splits}`",
        f"- First-stage Cox ridge alpha: `{args.ridge_alpha}`",
        "- Fold-wise imputation uses training-fold BSI feature means. Continuous features/covariates are standardized in training folds before applying to validation folds.",
        "- The first-stage model includes the same adjustment covariates as the final model, but the out-of-fold score is calculated from BSI feature coefficients only.",
        "- MrOS has no sex variation; sex is dropped when constant.",
        "",
        "## Feature sets",
    ]
    for name, cols in feature_sets.items():
        lines.append(f"- `{name}`: `{len(cols)}` BSI features")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `bsi_score_cv_results.csv`: numeric model output.",
            "- `bsi_score_cv_results.md`: compact result tables.",
            "- `bsi_score_oof_predictions.csv`: subject-level out-of-fold BSI scores.",
            "- `bsi_score_fold_coefficients.csv`: first-stage fold coefficients.",
        ]
    )
    non_ok = results[results["status"].ne("ok")]
    if not non_ok.empty:
        lines.extend(["", "## Non-ok final fits", ""])
        for _, row in non_ok.iterrows():
            lines.append(
                f"- {row['feature_set']} {row['cohort_label']} {row['model_name']}: {row['status']} {row['status_detail']}"
            )
    (outdir / "bsi_score_cv_note.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.master_csv, nrows=0)
    feature_sets = build_feature_sets(list(header.columns))
    if args.feature_sets:
        unknown = sorted(set(args.feature_sets) - set(feature_sets))
        if unknown:
            raise ValueError(f"Unknown feature set(s): {unknown}. Available: {sorted(feature_sets)}")
        feature_sets = {name: feature_sets[name] for name in args.feature_sets}
    all_features = list(dict.fromkeys(col for cols in feature_sets.values() for col in cols))
    base_cols = [
        "fileid",
        "sid",
        "cohort",
        "followup_days",
        "vital_status",
        "age",
        "sex",
        *COMPARATORS.keys(),
    ]
    master = pd.read_csv(args.master_csv, usecols=list(dict.fromkeys(base_cols + all_features)), low_memory=False)
    selected = pd.read_csv(
        args.selection_csv, usecols=["selection_mode", "fileid", "sid", "cohort"], low_memory=False
    )
    selected = selected[selected["selection_mode"].eq(args.selection_mode) & selected["cohort"].isin(COHORTS)].copy()
    data = master.merge(
        selected[["selection_mode", "fileid", "sid", "cohort"]],
        on=["fileid", "sid", "cohort"],
        how="inner",
    )
    data["cohort_label"] = data["cohort"].map(COHORTS)
    data["event"] = data["vital_status"].astype(str).str.lower().eq("dead").astype(float)
    for col in ["followup_days", "event", "age", "sex", *COMPARATORS.keys(), *all_features]:
        data[col] = safe_numeric(data[col])

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []

    for feature_set, feature_cols in feature_sets.items():
        for cohort, cohort_label in COHORTS.items():
            cohort_frame = data[data["cohort"].eq(cohort)].copy().reset_index(drop=True)
            for model_name, model_spec in MODEL_SPECS.items():
                covariates = list(model_spec["covariates"])
                required = ["followup_days", "event", "age", *[c for c in covariates if c != "sex"]]
                if "sex" in covariates and cohort_frame["sex"].dropna().nunique() > 1:
                    required.append("sex")
                frame = cohort_frame.dropna(subset=list(dict.fromkeys(required))).copy().reset_index(drop=True)
                print(f"{feature_set} {cohort_label} {model_name}: n={len(frame)}", flush=True)
                score, coefs = build_oof_score(
                    frame,
                    feature_cols,
                    covariates,
                    args.n_splits,
                    args.random_state,
                    args.ridge_alpha,
                )
                frame["bsi_score_oof"] = score
                final = final_phreg(frame, covariates)
                result_rows.append(
                    {
                        "selection_mode": args.selection_mode,
                        "feature_set": feature_set,
                        "n_bsi_features": len(feature_cols),
                        "cohort": cohort,
                        "cohort_label": cohort_label,
                        "model_name": model_name,
                        "model_label": model_spec["label"],
                        **final,
                    }
                )
                pred = frame[["fileid", "sid", "cohort", "cohort_label", "followup_days", "event"]].copy()
                pred["feature_set"] = feature_set
                pred["model_name"] = model_name
                pred["bsi_score_oof"] = frame["bsi_score_oof"]
                prediction_rows.append(pred)
                coefs["feature_set"] = feature_set
                coefs["cohort"] = cohort
                coefs["cohort_label"] = cohort_label
                coefs["model_name"] = model_name
                coefficient_rows.append(coefs)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "bsi_score_cv_results.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(args.output_dir / "bsi_score_oof_predictions.csv", index=False)
    pd.concat(coefficient_rows, ignore_index=True).to_csv(
        args.output_dir / "bsi_score_fold_coefficients.csv", index=False
    )

    original_note = [
        "Original manuscript Figure 7 reported stronger BSI-score mortality associations: unadjusted HRs were MrOS 1.13, MGH 1.60, BIDMC 1.20; age-adjusted HRs were MrOS 1.04, MGH 1.23, BIDMC 1.16.",
        f"The current analysis intentionally rebuilds that idea in the revision `{args.selection_mode}` BIDMC/MrOS cohorts using out-of-fold BSI feature scores.",
    ]
    write_markdown(args.output_dir, results, original_note)
    write_note(args.output_dir, args, results, feature_sets)
    print(f"Wrote {args.output_dir / 'bsi_score_cv_results.csv'}")


if __name__ == "__main__":
    main()
