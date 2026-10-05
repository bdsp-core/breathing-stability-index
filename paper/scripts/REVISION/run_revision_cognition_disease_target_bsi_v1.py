#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER = REPO_ROOT / "REVISION" / "master_analysis_table_primary_cohorts_v4_paper.csv"
TABLE_PHI = REPO_ROOT / "table_phi.csv"
OUTDIR = REPO_ROOT / "REVISION" / "cognition_disease_target_bsi_v1"

COGNITION_TARGETS = [
    ("cog_fluid", "Fluid cognition"),
    ("cog_crystallized", "Crystallized cognition"),
    ("cog_total", "Total cognition"),
]

DISEASE_SPECS = [
    ("dementia", "Dementia", "dx-tm-dementia"),
    ("mci", "MCI", "dx-tm-mci"),
    ("symptomatic", "Symptomatic", "dx-tm-symptomatic"),
    ("atrial_fibrillation", "Atrial fibrillation", "dx-tm-atrial_fibrillation"),
    ("myocardial_infarction", "Myocardial infarction", "dx-tm-myocardial_infarction"),
    ("diabetesii", "Diabetes II", "dx-tm-diabetesii"),
    ("hypertension", "Hypertension", "dx-tm-hypertension"),
    ("bipolar_disorder", "Bipolar disorder", "dx-tm-bipolar_disorder"),
    ("depression", "Depression", "dx-tm-depression"),
]

COMPARATORS = {
    "ss_percent_main": "SS",
    "ahi": "AHI",
    "hypoxic_burden": "HB",
    "arousal_index": "ArI",
}

EXPOSURE_LABELS = {
    "bsi_score": "BSI feature score",
    **COMPARATORS,
}

ALPHA_GRID = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=float)


@dataclass
class ScoreResult:
    score_raw: np.ndarray
    score_z: np.ndarray
    metric: float
    ci_lower: float
    ci_upper: float
    selected_alphas: list[float]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_fileid(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = Path(str(value).strip()).stem
    if not text or text.lower() == "nan":
        return None
    return text.replace("sub-S", "sub-s").lower()


def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.full(arr.shape, np.nan)
    return (arr - mean) / sd


def bsi_feature_columns(columns: list[str]) -> list[str]:
    out = []
    stages = ("sleep", "nrem", "rem")
    for col in columns:
        if not col.startswith("bsi_robust_mean_w2_ov0p9_"):
            continue
        if not any(col.startswith(f"bsi_robust_mean_w2_ov0p9_{stage}_") for stage in stages):
            continue
        if "quantile" in col or "n_5min_windows_stable" in col or "n_5min_windows_unstable" in col:
            out.append(col)
    return out


def prepare_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train = train.apply(safe_numeric).to_numpy(dtype=float)
    x_test = test.apply(safe_numeric).to_numpy(dtype=float)
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, means)
    x_test = np.where(np.isfinite(x_test), x_test, means)
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (x_train - mu) / sd, (x_test - mu) / sd


def group_folds(groups: pd.Series | np.ndarray, n_splits: int = 10, seed: int = 42) -> list[np.ndarray]:
    group_values = pd.Series(groups).astype(str).fillna("missing").to_numpy()
    unique_groups, counts = np.unique(group_values, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique_groups))
    rng.shuffle(order)
    fold_sizes = np.zeros(n_splits, dtype=int)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    for idx in order[np.argsort(counts[order])[::-1]]:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(unique_groups[idx])
        fold_sizes[fold] += int(counts[idx])
    folds = []
    for vals in fold_groups:
        folds.append(np.flatnonzero(np.isin(group_values, vals)))
    return [fold for fold in folds if len(fold) > 0]


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s, method="average")
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_metric(y_true: np.ndarray, scores: np.ndarray, metric: str, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(123)
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) < 20:
        return float("nan"), float("nan")
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if metric == "r":
            if np.nanstd(y[sample]) == 0 or np.nanstd(s[sample]) == 0:
                continue
            vals.append(float(stats.pearsonr(y[sample], s[sample]).statistic))
        elif metric == "auc":
            if len(np.unique(y[sample])) < 2:
                continue
            vals.append(rank_auc(y[sample], s[sample]))
    if not vals:
        return float("nan"), float("nan")
    low, high = np.percentile(vals, [2.5, 97.5])
    return float(low), float(high)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return np.column_stack([np.ones(len(x_test)), x_test]) @ beta


def choose_ridge_alpha_regression(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | np.ndarray,
    seed: int,
    alphas: np.ndarray = ALPHA_GRID,
) -> float:
    folds = group_folds(groups, n_splits=min(5, max(2, len(np.unique(groups)))), seed=seed)
    scores = []
    for alpha in alphas:
        mse = []
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
            xtr, xte = prepare_matrix(x.iloc[train_idx], x.iloc[test_idx])
            pred = ridge_predict(xtr, y[train_idx], xte, alpha)
            mse.append(float(np.mean((y[test_idx] - pred) ** 2)))
        scores.append(np.mean(mse))
    return float(alphas[int(np.argmin(scores))])


def oof_ridge_score(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | np.ndarray,
    n_splits: int = 10,
    seed: int = 42,
) -> ScoreResult:
    y_z = zscore(y)
    folds = group_folds(groups, n_splits=n_splits, seed=seed)
    pred = np.full(len(y_z), np.nan)
    selected = []
    for fold_num, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(y_z)), test_idx, assume_unique=False)
        inner_groups = pd.Series(groups).iloc[train_idx].to_numpy()
        alpha = choose_ridge_alpha_regression(x.iloc[train_idx], y_z[train_idx], inner_groups, seed + fold_num)
        xtr, xte = prepare_matrix(x.iloc[train_idx], x.iloc[test_idx])
        pred[test_idx] = ridge_predict(xtr, y_z[train_idx], xte, alpha)
        selected.append(alpha)
    score_z = zscore(pred)
    r = float(stats.pearsonr(y_z[np.isfinite(score_z)], score_z[np.isfinite(score_z)]).statistic)
    low, high = bootstrap_metric(y_z, score_z, "r")
    return ScoreResult(pred, score_z, r, low, high, selected)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    y = y.astype(float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x_aug @ beta
        p = sigmoid(eta)
        nll = -np.sum(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))
        nll += 0.5 * alpha * np.sum(beta[1:] ** 2)
        grad = x_aug.T @ (p - y)
        grad[1:] += alpha * beta[1:]
        return float(nll), grad

    start = np.zeros(x_aug.shape[1])
    prevalence = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
    start[0] = math.log(prevalence / (1.0 - prevalence))
    res = optimize.minimize(lambda b: objective(b)[0], start, jac=lambda b: objective(b)[1], method="L-BFGS-B")
    if not res.success:
        warnings.warn(f"logistic ridge optimizer did not fully converge: {res.message}")
    return np.asarray(res.x, dtype=float)


def logistic_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    beta = fit_logistic_ridge(x_train, y_train, alpha)
    return np.column_stack([np.ones(len(x_test)), x_test]) @ beta


def log_loss(y: np.ndarray, eta: np.ndarray) -> float:
    p = sigmoid(eta)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12)))


def choose_ridge_alpha_logistic(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | np.ndarray,
    seed: int,
    alphas: np.ndarray = ALPHA_GRID,
) -> float:
    folds = group_folds(groups, n_splits=min(5, max(2, len(np.unique(groups)))), seed=seed)
    scores = []
    for alpha in alphas:
        losses = []
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            xtr, xte = prepare_matrix(x.iloc[train_idx], x.iloc[test_idx])
            eta = logistic_linear_predict(xtr, y[train_idx], xte, alpha)
            losses.append(log_loss(y[test_idx], eta))
        scores.append(np.mean(losses) if losses else np.inf)
    return float(alphas[int(np.argmin(scores))])


def oof_logistic_ridge_score(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | np.ndarray,
    n_splits: int = 10,
    seed: int = 42,
) -> ScoreResult:
    yv = safe_numeric(y).astype(int).to_numpy()
    folds = group_folds(groups, n_splits=n_splits, seed=seed)
    eta = np.full(len(yv), np.nan)
    selected = []
    for fold_num, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(yv)), test_idx, assume_unique=False)
        inner_groups = pd.Series(groups).iloc[train_idx].to_numpy()
        alpha = choose_ridge_alpha_logistic(x.iloc[train_idx], yv[train_idx], inner_groups, seed + fold_num)
        xtr, xte = prepare_matrix(x.iloc[train_idx], x.iloc[test_idx])
        eta[test_idx] = logistic_linear_predict(xtr, yv[train_idx], xte, alpha)
        selected.append(alpha)
    score_z = zscore(eta)
    if np.isfinite(score_z).sum() == 0:
        score_z = np.zeros(len(eta), dtype=float)
    auc = rank_auc(yv, eta)
    low, high = bootstrap_metric(yv, eta, "auc")
    return ScoreResult(eta, score_z, auc, low, high, selected)


def design_matrix(df: pd.DataFrame, columns: list[str], categorical: set[str] | None = None) -> pd.DataFrame:
    categorical = categorical or set()
    parts = []
    for col in columns:
        if col in categorical:
            parts.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
        else:
            vals = safe_numeric(df[col])
            if col != "sex":
                vals = pd.Series(zscore(vals), index=df.index)
            parts.append(vals.rename(col).to_frame())
    if not parts:
        return pd.DataFrame(index=df.index)
    x = pd.concat(parts, axis=1)
    keep_cols = []
    for col in x.columns:
        vals = x[col].to_numpy(dtype=float)
        if np.nanstd(vals) > 0:
            keep_cols.append(col)
    return x[keep_cols]


def prediction_matrix(df: pd.DataFrame, columns: list[str], categorical: set[str] | None = None) -> pd.DataFrame:
    categorical = categorical or set()
    parts = []
    for col in columns:
        if col in categorical:
            parts.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
        else:
            parts.append(safe_numeric(df[col]).rename(col).to_frame())
    x = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return x.loc[:, x.notna().any(axis=0)]


def add_delta_vs_m0(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        performance["delta_vs_m0"] = np.nan
        return performance
    out = performance.copy()
    m0 = (
        out[out["model_set"].eq("M0_demographics")]
        [["endpoint_type", "target", "estimate"]]
        .rename(columns={"estimate": "m0_estimate"})
    )
    out = out.merge(m0, on=["endpoint_type", "target"], how="left")
    out["delta_vs_m0"] = out["estimate"] - out["m0_estimate"]
    return out.drop(columns=["m0_estimate"])


def cognition_model(
    df: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: list[str],
    model_name: str,
) -> dict[str, object] | None:
    cols = [outcome, exposure, *covariates]
    dat = df[cols].copy().dropna()
    if len(dat) < 50 or dat[exposure].nunique() < 2:
        return None
    y = pd.Series(zscore(dat[outcome]), index=dat.index)
    dat = dat.assign(**{exposure: zscore(dat[exposure])})
    categorical = {"cohort"} if "cohort" in covariates else set()
    x_full = design_matrix(dat, [exposure, *covariates], categorical=categorical)
    if exposure not in x_full.columns:
        return None
    x_full = sm.add_constant(x_full, has_constant="add")
    x_reduced = design_matrix(dat, covariates, categorical=categorical)
    x_reduced = sm.add_constant(x_reduced, has_constant="add")
    full_fit = sm.OLS(y, x_full).fit()
    robust = full_fit.get_robustcov_results(cov_type="HC3")
    names = list(full_fit.params.index)
    idx = names.index(exposure)
    rss_full = float(np.sum(full_fit.resid**2))
    reduced_fit = sm.OLS(y, x_reduced).fit()
    rss_reduced = float(np.sum(reduced_fit.resid**2))
    partial_r2 = max(0.0, (rss_reduced - rss_full) / rss_reduced) if rss_reduced > 0 else np.nan
    return {
        "outcome": outcome,
        "exposure": exposure,
        "model": model_name,
        "n": int(len(dat)),
        "beta_std": float(robust.params[idx]),
        "ci_lower": float(robust.conf_int()[idx, 0]),
        "ci_upper": float(robust.conf_int()[idx, 1]),
        "p_value": float(robust.pvalues[idx]),
        "partial_r2": float(partial_r2),
        "r_squared": float(full_fit.rsquared),
    }


def disease_model(
    df: pd.DataFrame,
    disease: str,
    exposure: str,
    covariates: list[str],
    model_name: str,
) -> dict[str, object] | None:
    cols = ["disease_label", "pair_id", exposure, *covariates]
    dat = df[cols].copy().dropna()
    if len(dat) < 50 or dat["disease_label"].nunique() < 2 or dat[exposure].nunique() < 2:
        return None
    dat = dat.assign(**{exposure: zscore(dat[exposure])})
    x = design_matrix(dat, [exposure, *covariates])
    if exposure not in x.columns:
        return None
    x = sm.add_constant(x, has_constant="add")
    y = safe_numeric(dat["disease_label"]).astype(int)
    try:
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": dat["pair_id"].astype(str).to_numpy()},
            maxiter=100,
        )
    except Exception:
        fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100)
    beta = float(fit.params[exposure])
    ci = fit.conf_int().loc[exposure]
    return {
        "disease": disease,
        "exposure": exposure,
        "model": model_name,
        "n": int(len(dat)),
        "n_cases": int(y.sum()),
        "n_controls": int((1 - y).sum()),
        "odds_ratio": float(np.exp(beta)),
        "ci_lower": float(np.exp(ci.iloc[0])),
        "ci_upper": float(np.exp(ci.iloc[1])),
        "p_value": float(fit.pvalues[exposure]),
    }


def model_covariates(exposure: str, endpoint_type: str, model: str) -> list[str]:
    cohort = ["cohort"] if endpoint_type == "cognition" else []
    base = ["age", "sex"] + cohort
    if model == "M1":
        return []
    if model == "M2":
        return base
    if model == "M3":
        return base + ([] if exposure == "ahi" else ["ahi"])
    if model == "M4":
        comparators = list(COMPARATORS)
        if exposure == "bsi_score":
            return base + comparators
        return base + ["bsi_score"] + [col for col in comparators if col != exposure]
    raise ValueError(model)


def load_master() -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(MASTER, nrows=0).columns.tolist()
    bsi_cols = bsi_feature_columns(header)
    required = sorted(
        set(
            [
                "fileid",
                "sid",
                "cohort",
                "age",
                "sex",
                "visitno",
                *[x[0] for x in COGNITION_TARGETS],
                *COMPARATORS.keys(),
                *bsi_cols,
            ]
        )
        & set(header)
    )
    master = pd.read_csv(MASTER, usecols=required, low_memory=False)
    for col in ["age", "sex", *COMPARATORS.keys(), *bsi_cols]:
        if col in master.columns:
            master[col] = safe_numeric(master[col])
    return master, bsi_cols


def cognition_analysis(master: pd.DataFrame, bsi_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = master[master[[x[0] for x in COGNITION_TARGETS]].notna().any(axis=1)].copy()
    base["analysis_subject_id"] = np.where(
        base["cohort"].astype(str).eq("mgh-cog"),
        base["fileid"].astype(str),
        base["cohort"].astype(str) + ":" + base["sid"].astype(str),
    )

    model_rows = []
    score_rows = []
    corr_rows = []
    performance_rows = []
    for outcome, outcome_label in COGNITION_TARGETS:
        dat = base.dropna(subset=[outcome]).copy()
        dat = dat[dat[bsi_cols].notna().any(axis=1)].copy()
        score = oof_ridge_score(dat[bsi_cols], dat[outcome], dat["analysis_subject_id"], seed=101)
        m0_x = prediction_matrix(dat, ["age", "sex", "cohort"], categorical={"cohort"})
        m0_score = oof_ridge_score(m0_x, dat[outcome], dat["analysis_subject_id"], seed=151)
        demo_bsi_x = pd.concat([m0_x, dat[bsi_cols]], axis=1)
        demo_bsi_score = oof_ridge_score(demo_bsi_x, dat[outcome], dat["analysis_subject_id"], seed=171)
        dat["bsi_score"] = score.score_z
        performance_rows.extend(
            [
                {
                    "endpoint_type": "cognition",
                    "target": outcome,
                    "target_label": outcome_label,
                    "model_set": "M0_demographics",
                    "n": int(len(dat)),
                    "n_cases": np.nan,
                    "n_controls": np.nan,
                    "metric": "pearson_r",
                    "estimate": m0_score.metric,
                    "ci_lower": m0_score.ci_lower,
                    "ci_upper": m0_score.ci_upper,
                    "note": "OOF age + sex + cohort benchmark.",
                },
                {
                    "endpoint_type": "cognition",
                    "target": outcome,
                    "target_label": outcome_label,
                    "model_set": "BSI_features",
                    "n": int(len(dat)),
                    "n_cases": np.nan,
                    "n_controls": np.nan,
                    "metric": "pearson_r",
                    "estimate": score.metric,
                    "ci_lower": score.ci_lower,
                    "ci_upper": score.ci_upper,
                    "note": "OOF target-specific BSI feature score.",
                },
                {
                    "endpoint_type": "cognition",
                    "target": outcome,
                    "target_label": outcome_label,
                    "model_set": "M0_plus_BSI_features",
                    "n": int(len(dat)),
                    "n_cases": np.nan,
                    "n_controls": np.nan,
                    "metric": "pearson_r",
                    "estimate": demo_bsi_score.metric,
                    "ci_lower": demo_bsi_score.ci_lower,
                    "ci_upper": demo_bsi_score.ci_upper,
                    "note": "OOF demographics plus target-specific BSI features.",
                },
            ]
        )
        score_rows.append(
            dat[["fileid", "sid", "cohort", "analysis_subject_id", outcome, "bsi_score"]]
            .assign(outcome=outcome, oof_r=score.metric, oof_ci_lower=score.ci_lower, oof_ci_upper=score.ci_upper)
            .rename(columns={outcome: "outcome_value"})
        )
        corr_vars = ["bsi_score", *COMPARATORS.keys()]
        corr = dat[corr_vars].corr(method="spearman")
        for a in corr_vars:
            for b in corr_vars:
                corr_rows.append({"outcome": outcome, "var1": a, "var2": b, "spearman_rho": corr.loc[a, b]})
        for exposure in ["bsi_score", *COMPARATORS.keys()]:
            for model in ["M1", "M2", "M3", "M4"]:
                covs = model_covariates(exposure, "cognition", model)
                row = cognition_model(dat, outcome, exposure, covs, model)
                if row is not None:
                    row["outcome_label"] = outcome_label
                    row["exposure_label"] = EXPOSURE_LABELS[exposure]
                    row["oof_prediction_r"] = score.metric if exposure == "bsi_score" else np.nan
                    row["oof_prediction_ci_lower"] = score.ci_lower if exposure == "bsi_score" else np.nan
                    row["oof_prediction_ci_upper"] = score.ci_upper if exposure == "bsi_score" else np.nan
                    model_rows.append(row)
        print(f"finished cognition target: {outcome}", flush=True)
    return (
        pd.DataFrame(model_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(corr_rows),
        add_delta_vs_m0(pd.DataFrame(performance_rows)),
    )


def session_number(fileid: pd.Series) -> pd.Series:
    return fileid.astype(str).str.extract(r"_ses-(\d+)")[0].astype(float)


def load_disease_subject_rows(master: pd.DataFrame, bsi_cols: list[str]) -> pd.DataFrame:
    master_features = master[
        ["fileid", "cohort", "age", "sex", *COMPARATORS.keys(), *bsi_cols]
    ].copy()
    master_features["analysis_fileid"] = master_features["fileid"].map(normalize_fileid)
    master_features = master_features[master_features["cohort"].eq("S0001")].copy()
    master_features = master_features.dropna(subset=["analysis_fileid"]).drop_duplicates("analysis_fileid", keep="first")
    rename = {
        "fileid": "master_fileid",
        "cohort": "master_cohort",
        "age": "master_age",
        "sex": "master_sex",
    }
    master_features = master_features.rename(columns=rename)

    dx_cols = [x[2] for x in DISEASE_SPECS]
    phi_cols = ["sid", "cohort", "fileid", "visitno", "age", "sex", *dx_cols]
    phi = pd.read_csv(TABLE_PHI, usecols=phi_cols, dtype={"sid": str}, low_memory=False)
    phi = phi[phi["cohort"].eq("mgh-dx")].copy()
    phi["source_fileid"] = phi["fileid"]
    phi["analysis_fileid"] = phi["fileid"].map(normalize_fileid)
    phi = phi.merge(master_features.drop(columns=["master_cohort"]), on="analysis_fileid", how="left")
    phi["master_feature_linked"] = phi["master_fileid"].notna()
    phi["age"] = safe_numeric(phi["age"]).fillna(safe_numeric(phi["master_age"]))
    phi["sex"] = safe_numeric(phi["sex"]).fillna(safe_numeric(phi["master_sex"]))
    phi["visit_sort"] = safe_numeric(phi["visitno"]).fillna(session_number(phi["source_fileid"])).fillna(999.0)
    phi["bsi_feature_nonmissing"] = phi[bsi_cols].notna().sum(axis=1)
    phi = phi[phi["master_feature_linked"] & phi["age"].notna() & phi["sex"].notna() & (phi["bsi_feature_nonmissing"] > 0)].copy()
    phi = phi.sort_values(["sid", "visit_sort", "analysis_fileid"], kind="mergesort")
    one = phi.drop_duplicates("sid", keep="first").copy()
    for col in [*dx_cols, *COMPARATORS.keys(), *bsi_cols]:
        if col in one.columns:
            one[col] = safe_numeric(one[col])
    return one


def build_matched_disease_rows(
    subject_rows: pd.DataFrame,
    bsi_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dx_cols = [x[2] for x in DISEASE_SPECS]
    any_dx = subject_rows[dx_cols].apply(safe_numeric).eq(1).any(axis=1)
    healthy = subject_rows[~any_dx].copy()
    matched_frames = []
    balance_rows = []
    coverage_rows = []
    for slug, label, dx_col in DISEASE_SPECS:
        cases = subject_rows[safe_numeric(subject_rows[dx_col]).eq(1)].copy()
        selected_controls = []
        pair_records = []
        unmatched = 0
        for sex_value, cases_for_sex in cases.groupby("sex", sort=True):
            available = set(healthy[healthy["sex"].eq(sex_value)].index.tolist())
            cases_for_sex = cases_for_sex.sort_values(["age", "analysis_fileid"], ascending=[False, True])
            for case_idx, case_row in cases_for_sex.iterrows():
                if not available:
                    unmatched += 1
                    continue
                available_idx = list(available)
                age_diffs = (subject_rows.loc[available_idx, "age"] - case_row["age"]).abs().sort_values(kind="mergesort")
                control_idx = age_diffs.index[0]
                pair_id = f"{slug}_{len(pair_records):05d}"
                selected_controls.append(control_idx)
                pair_records.append((pair_id, case_idx, control_idx))
                available.remove(control_idx)

        rows = []
        for pair_id, case_idx, control_idx in pair_records:
            case = subject_rows.loc[[case_idx]].copy()
            control = subject_rows.loc[[control_idx]].copy()
            case["disease_label"] = 1
            control["disease_label"] = 0
            case["pair_id"] = pair_id
            control["pair_id"] = pair_id
            rows.extend([case, control])
        matched = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        matched["disease"] = slug
        matched["disease_label_name"] = label
        matched["target_column"] = dx_col
        matched_frames.append(matched)

        case_age = cases["age"].dropna()
        control_age = subject_rows.loc[selected_controls, "age"].dropna()
        pooled = np.nan
        if len(case_age) > 1 and len(control_age) > 1:
            pooled = math.sqrt(
                ((len(case_age) - 1) * case_age.var(ddof=1) + (len(control_age) - 1) * control_age.var(ddof=1))
                / (len(case_age) + len(control_age) - 2)
            )
        balance_rows.append(
            {
                "disease": slug,
                "disease_label": label,
                "cases": int(len(cases)),
                "matched_controls": int(len(selected_controls)),
                "matched_n": int(2 * len(selected_controls)),
                "unmatched_cases": int(unmatched),
                "case_age_mean": float(case_age.mean()) if len(case_age) else np.nan,
                "control_age_mean": float(control_age.mean()) if len(control_age) else np.nan,
                "age_smd": float((case_age.mean() - control_age.mean()) / pooled)
                if pooled and np.isfinite(pooled) and pooled > 0
                else np.nan,
            }
        )

        cov_cols = ["age", "sex", *COMPARATORS.keys()]
        coverage_rows.append(
            {
                "disease": slug,
                "disease_label": label,
                "matched_n": int(len(matched)),
                "complete_m4": int(matched[cov_cols].notna().all(axis=1).sum()),
                "bsi_feature_any": int(matched[bsi_cols].notna().any(axis=1).sum()),
                "bsi_feature_all": int(matched[bsi_cols].notna().all(axis=1).sum()),
            }
        )

    return pd.concat(matched_frames, ignore_index=True), pd.DataFrame(balance_rows), pd.DataFrame(coverage_rows)


def disease_analysis(
    master: pd.DataFrame,
    bsi_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows = load_disease_subject_rows(master, bsi_cols)
    matched_all, balance, coverage = build_matched_disease_rows(subject_rows, bsi_cols)
    model_rows = []
    score_rows = []
    corr_rows = []
    performance_rows = []
    for slug, label, _ in DISEASE_SPECS:
        dat = matched_all[matched_all["disease"].eq(slug)].copy()
        dat = dat[dat[bsi_cols].notna().any(axis=1)].copy()
        score = oof_logistic_ridge_score(dat[bsi_cols], dat["disease_label"], dat["pair_id"], seed=201)
        m0_x = prediction_matrix(dat, ["age", "sex"])
        m0_score = oof_logistic_ridge_score(m0_x, dat["disease_label"], dat["pair_id"], seed=251)
        demo_bsi_x = pd.concat([m0_x, dat[bsi_cols]], axis=1)
        demo_bsi_score = oof_logistic_ridge_score(demo_bsi_x, dat["disease_label"], dat["pair_id"], seed=271)
        dat["bsi_score"] = score.score_z
        dat["bsi_score_raw"] = score.score_raw
        n_cases = int(safe_numeric(dat["disease_label"]).sum())
        n_controls = int(len(dat) - n_cases)
        performance_rows.extend(
            [
                {
                    "endpoint_type": "disease",
                    "target": slug,
                    "target_label": label,
                    "model_set": "M0_demographics",
                    "n": int(len(dat)),
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "metric": "auroc",
                    "estimate": m0_score.metric,
                    "ci_lower": m0_score.ci_lower,
                    "ci_upper": m0_score.ci_upper,
                    "note": "OOF age + sex matched-design sanity benchmark.",
                },
                {
                    "endpoint_type": "disease",
                    "target": slug,
                    "target_label": label,
                    "model_set": "BSI_features",
                    "n": int(len(dat)),
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "metric": "auroc",
                    "estimate": score.metric,
                    "ci_lower": score.ci_lower,
                    "ci_upper": score.ci_upper,
                    "note": "OOF target-specific BSI feature score.",
                },
                {
                    "endpoint_type": "disease",
                    "target": slug,
                    "target_label": label,
                    "model_set": "M0_plus_BSI_features",
                    "n": int(len(dat)),
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "metric": "auroc",
                    "estimate": demo_bsi_score.metric,
                    "ci_lower": demo_bsi_score.ci_lower,
                    "ci_upper": demo_bsi_score.ci_upper,
                    "note": "OOF demographics plus target-specific BSI features.",
                },
            ]
        )
        score_rows.append(
            dat[
                [
                    "source_fileid",
                    "master_fileid",
                    "sid",
                    "analysis_fileid",
                    "disease",
                    "disease_label_name",
                    "pair_id",
                    "disease_label",
                    "bsi_score",
                    "bsi_score_raw",
                ]
            ].assign(oof_auc=score.metric, oof_ci_lower=score.ci_lower, oof_ci_upper=score.ci_upper)
        )
        corr_vars = ["bsi_score", *COMPARATORS.keys()]
        corr = dat[corr_vars].corr(method="spearman")
        for a in corr_vars:
            for b in corr_vars:
                corr_rows.append({"disease": slug, "var1": a, "var2": b, "spearman_rho": corr.loc[a, b]})
        for exposure in ["bsi_score", *COMPARATORS.keys()]:
            for model in ["M1", "M2", "M3", "M4"]:
                covs = model_covariates(exposure, "disease", model)
                row = disease_model(dat, label, exposure, covs, model)
                if row is not None:
                    row["disease_slug"] = slug
                    row["exposure_label"] = EXPOSURE_LABELS[exposure]
                    row["oof_prediction_auc"] = score.metric if exposure == "bsi_score" else np.nan
                    row["oof_prediction_ci_lower"] = score.ci_lower if exposure == "bsi_score" else np.nan
                    row["oof_prediction_ci_upper"] = score.ci_upper if exposure == "bsi_score" else np.nan
                    model_rows.append(row)
        print(f"finished disease target: {slug}", flush=True)
    return (
        pd.DataFrame(model_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(corr_rows),
        balance,
        coverage,
        add_delta_vs_m0(pd.DataFrame(performance_rows)),
    )


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def fmt_effect(row: pd.Series, kind: str) -> str:
    if kind == "cognition":
        return f"{row['beta_std']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})"
    return f"{row['odds_ratio']:.2f} ({row['ci_lower']:.2f}, {row['ci_upper']:.2f})"


def write_markdown_tables(
    cognition_results: pd.DataFrame,
    disease_results: pd.DataFrame,
    performance_results: pd.DataFrame,
    balance: pd.DataFrame,
    coverage: pd.DataFrame,
    outdir: Path,
    n_bsi_features: int,
) -> None:
    perf_lines = [
        "# Prediction Performance Benchmark",
        "",
        "M0 is an out-of-fold demographic benchmark only and is not part of the exposure-association model ladder. Disease M0 is expected to be near chance because case/control matching is exact for sex and nearest-age.",
        "",
        "| endpoint type | target | model set | n | cases | controls | metric | estimate (95% CI) | delta vs M0 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    sort_cols = ["endpoint_type", "target", "model_set"]
    for _, row in performance_results.sort_values(sort_cols).iterrows():
        cases = "" if pd.isna(row["n_cases"]) else f"{int(row['n_cases']):,}"
        controls = "" if pd.isna(row["n_controls"]) else f"{int(row['n_controls']):,}"
        perf_lines.append(
            f"| {row['endpoint_type']} | {row['target_label']} | {row['model_set']} | {int(row['n']):,} | "
            f"{cases} | {controls} | {row['metric']} | "
            f"{row['estimate']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f}) | {row['delta_vs_m0']:+.3f} |"
        )
    (outdir / "performance_benchmark_results.md").write_text("\n".join(perf_lines) + "\n")

    cog_lines = [
        "# Cognition Target-Specific BSI Score Results",
        "",
        "BSI scores are target-specific out-of-fold ridge scores trained only on prespecified BSI quantile and 5-minute stable/unstable window features. Cognition outcomes are standardized; beta values are per 1 SD exposure.",
        "",
        "| outcome | exposure | model | n | beta (95% CI) | p | partial R2 | OOF r |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in cognition_results.sort_values(["outcome", "exposure", "model"]).iterrows():
        oof = ""
        if row["exposure"] == "bsi_score":
            oof = f"{row['oof_prediction_r']:.3f}"
        cog_lines.append(
            f"| {row['outcome_label']} | {row['exposure_label']} | {row['model']} | {int(row['n']):,} | "
            f"{fmt_effect(row, 'cognition')} | {fmt_p(row['p_value'])} | {row['partial_r2']:.4f} | {oof} |"
        )
    (outdir / "cognition_model_ladder_results.md").write_text("\n".join(cog_lines) + "\n")

    dis_lines = [
        "# Disease Target-Specific BSI Score Results",
        "",
        "Disease labels are matched cross-sectional MGH-DX/S0001 labels. The primary disease analysis uses one feature-linked PSG per subject before exact-sex nearest-age matching. Odds ratios are per 1 SD exposure from logistic models with matched-pair clustered standard errors.",
        "",
        "## Matching and Coverage",
        "",
        "| disease | cases | matched controls | complete M4 | BSI feature any |",
        "|---|---:|---:|---:|---:|",
    ]
    cov_map = coverage.set_index("disease")
    for _, row in balance.iterrows():
        cov = cov_map.loc[row["disease"]]
        dis_lines.append(
            f"| {row['disease_label']} | {int(row['cases']):,} | {int(row['matched_controls']):,} | "
            f"{int(cov['complete_m4']):,} | {int(cov['bsi_feature_any']):,} |"
        )
    dis_lines.extend(
        [
            "",
            "## Model Results",
            "",
            "| disease | exposure | model | n | OR (95% CI) | p | OOF AUROC |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in disease_results.sort_values(["disease", "exposure", "model"]).iterrows():
        oof = ""
        if row["exposure"] == "bsi_score":
            oof = f"{row['oof_prediction_auc']:.3f}"
        dis_lines.append(
            f"| {row['disease']} | {row['exposure_label']} | {row['model']} | {int(row['n']):,} | "
            f"{fmt_effect(row, 'disease')} | {fmt_p(row['p_value'])} | {oof} |"
        )
    (outdir / "disease_model_ladder_results.md").write_text("\n".join(dis_lines) + "\n")

    summary_lines = [
        "# Cognition and Disease Target-Specific BSI v1",
        "",
        f"Input master: `{MASTER}`",
        f"BSI feature block: `{n_bsi_features}` prespecified quantile/window features; model feature list is saved in `bsi_feature_columns.txt`.",
        "M0 and M0+BSI are used only as prediction-performance benchmarks; the association ladder remains M1-M4.",
        "",
        "## Prediction Performance Benchmark",
        "",
        "| endpoint | target | M0 | BSI features | M0 + BSI | BSI delta | M0 + BSI delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint_type in ["cognition", "disease"]:
        sub = performance_results[performance_results["endpoint_type"].eq(endpoint_type)]
        for target in sub["target"].drop_duplicates():
            rows = sub[sub["target"].eq(target)].set_index("model_set")
            if (
                "M0_demographics" not in rows.index
                or "BSI_features" not in rows.index
                or "M0_plus_BSI_features" not in rows.index
            ):
                continue
            m0 = rows.loc["M0_demographics"]
            bsi = rows.loc["BSI_features"]
            combo = rows.loc["M0_plus_BSI_features"]
            summary_lines.append(
                f"| {endpoint_type} | {bsi['target_label']} | {m0['estimate']:.3f} | "
                f"{bsi['estimate']:.3f} | {combo['estimate']:.3f} | "
                f"{bsi['delta_vs_m0']:+.3f} | {combo['delta_vs_m0']:+.3f} |"
            )
    summary_lines.extend(
        [
        "",
        "## Cognition BSI Score Summary",
        "",
        "| outcome | n | OOF r | M4 beta (95% CI) | M4 p |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    bsi_cog = cognition_results[cognition_results["exposure"].eq("bsi_score")]
    for outcome, label in COGNITION_TARGETS:
        sub = bsi_cog[(bsi_cog["outcome"].eq(outcome)) & (bsi_cog["model"].eq("M4"))]
        if sub.empty:
            continue
        row = sub.iloc[0]
        summary_lines.append(
            f"| {label} | {int(row['n']):,} | {row['oof_prediction_r']:.3f} | "
            f"{fmt_effect(row, 'cognition')} | {fmt_p(row['p_value'])} |"
        )
    summary_lines.extend(
        [
            "",
            "## Disease BSI Score Summary",
            "",
            "| disease | n | OOF AUROC | M4 OR (95% CI) | M4 p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    bsi_dis = disease_results[disease_results["exposure"].eq("bsi_score")]
    for slug, label, _ in DISEASE_SPECS:
        sub = bsi_dis[(bsi_dis["disease_slug"].eq(slug)) & (bsi_dis["model"].eq("M4"))]
        if sub.empty:
            continue
        row = sub.iloc[0]
        summary_lines.append(
            f"| {label} | {int(row['n']):,} | {row['oof_prediction_auc']:.3f} | "
            f"{fmt_effect(row, 'disease')} | {fmt_p(row['p_value'])} |"
        )
    summary_lines.extend(
        [
            "",
            "## Model Ladder",
            "",
            "- M1: unadjusted.",
            "- M2: age + sex, plus cohort for cognition.",
            "- M3: age + sex + AHI, plus cohort for cognition; AHI self-adjustment is omitted for AHI exposure rows.",
            "- M4 BSI score: age + sex + SS + AHI + HB + ArI, plus cohort for cognition.",
            "- M4 comparator rows: age + sex + target-specific BSI score + the other comparator metrics, plus cohort for cognition.",
            "- Disease analyses are cross-sectional matched analyses, not incident disease models.",
        ]
    )
    (outdir / "summary.md").write_text("\n".join(summary_lines) + "\n")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    master, bsi_cols = load_master()
    if len(bsi_cols) != 60:
        warnings.warn(f"Expected 60 mortality-style BSI features, found {len(bsi_cols)}")
    (OUTDIR / "bsi_feature_columns.txt").write_text("\n".join(bsi_cols) + "\n")

    cognition_results, cognition_scores, cognition_corr, cognition_performance = cognition_analysis(master, bsi_cols)
    cognition_results.to_csv(OUTDIR / "cognition_model_ladder_results.csv", index=False)
    cognition_scores.to_csv(OUTDIR / "cognition_target_bsi_oof_scores.csv", index=False)
    cognition_corr.to_csv(OUTDIR / "cognition_bsi_score_comparator_correlations.csv", index=False)

    disease_results, disease_scores, disease_corr, balance, coverage, disease_performance = disease_analysis(master, bsi_cols)
    disease_results.to_csv(OUTDIR / "disease_model_ladder_results.csv", index=False)
    disease_scores.to_csv(OUTDIR / "disease_target_bsi_oof_scores.csv", index=False)
    disease_corr.to_csv(OUTDIR / "disease_bsi_score_comparator_correlations.csv", index=False)
    balance.to_csv(OUTDIR / "disease_matching_balance.csv", index=False)
    coverage.to_csv(OUTDIR / "disease_matching_comparator_coverage.csv", index=False)

    performance_results = add_delta_vs_m0(pd.concat([cognition_performance, disease_performance], ignore_index=True))
    performance_results.to_csv(OUTDIR / "performance_benchmark_results.csv", index=False)

    write_markdown_tables(
        cognition_results,
        disease_results,
        performance_results,
        balance,
        coverage,
        OUTDIR,
        len(bsi_cols),
    )
    print(f"Wrote {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
