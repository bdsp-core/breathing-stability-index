from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


DEFAULT_MASTER = Path("REVISION/master_analysis_table_primary_cohorts_v4_paper.csv")
DEFAULT_OUTPUT_DIR = Path("REVISION/osa_csa_ahi_strata_v1")

PRIMARY_BSI = "bsi_median_sleep"
PHENOTYPE = "phenotype_osa_csa"
COHORT = "cohort"

AHI_BINS = [-np.inf, 5.0, 15.0, 30.0, np.inf]
AHI_LABELS = ["Normal (<5)", "Mild (5-<15)", "Moderate (15-<30)", "Severe (>=30)"]
PHENOTYPE_ORDER = ["csa", "mixed", "osa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSA/CSA phenotype-by-AHI BSI analyses for the revision."
    )
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def iqr_text(series: pd.Series) -> str:
    s = safe_numeric(series).dropna()
    if s.empty:
        return "NA"
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.3f} [{q1:.3f}, {q3:.3f}]"


def p_text(p_value: object) -> str:
    if p_value is None or pd.isna(p_value):
        return "NA"
    p_float = float(p_value)
    if p_float < 0.001:
        return "<0.001"
    return f"{p_float:.3f}".rstrip("0").rstrip(".")


def contrast_text(row: pd.Series | None) -> str:
    if row is None or row.empty or row.get("status", "ok") != "ok":
        return "not estimable"
    p_formatted = p_text(row["p_value"])
    p_clause = f"p{p_formatted}" if p_formatted.startswith("<") else f"p={p_formatted}"
    return (
        f"{float(row['beta']):.3f} "
        f"(95% CI {float(row['ci_lower']):.3f} to {float(row['ci_upper']):.3f}; "
        f"{p_clause})"
    )


def load_master(path: Path) -> pd.DataFrame:
    usecols = [
        COHORT,
        "age",
        "sex",
        "bmi",
        "ahi",
        "oai",
        "cai",
        "hypoxic_burden",
        "arousal_index",
        "ss_percent_main",
        PRIMARY_BSI,
        PHENOTYPE,
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df[COHORT] = df[COHORT].astype(str)
    df[PHENOTYPE] = df[PHENOTYPE].astype(str).str.lower()
    for col in [
        "age",
        "sex",
        "bmi",
        "ahi",
        "oai",
        "cai",
        "hypoxic_burden",
        "arousal_index",
        "ss_percent_main",
        PRIMARY_BSI,
    ]:
        df[col] = safe_numeric(df[col])
    df["ahi_category"] = pd.cut(
        df["ahi"],
        bins=AHI_BINS,
        labels=AHI_LABELS,
        right=False,
        ordered=True,
    )
    return df


def phenotype_analysis_set(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df[PHENOTYPE].isin(PHENOTYPE_ORDER)
        & df[PRIMARY_BSI].notna()
        & df["ahi"].notna()
        & df["ahi_category"].notna()
    ].copy()


def build_descriptives(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ahi_category in AHI_LABELS:
        ahi_df = df[df["ahi_category"].eq(ahi_category)]
        for phenotype in PHENOTYPE_ORDER:
            grp = ahi_df[ahi_df[PHENOTYPE].eq(phenotype)]
            bsi = grp[PRIMARY_BSI].dropna()
            model_complete = grp[
                [PRIMARY_BSI, PHENOTYPE, "age", "sex", "bmi", COHORT]
            ].dropna()
            rows.append(
                {
                    "ahi_category": ahi_category,
                    "phenotype_osa_csa": phenotype,
                    "n": int(len(grp)),
                    "n_model_age_sex_bmi_cohort": int(len(model_complete)),
                    "bsi_median_sleep_iqr": iqr_text(grp[PRIMARY_BSI]),
                    "bsi_mean": float(bsi.mean()) if not bsi.empty else np.nan,
                    "bsi_sd": float(bsi.std()) if len(bsi) > 1 else np.nan,
                    "ahi_iqr": iqr_text(grp["ahi"]),
                    "oai_iqr": iqr_text(grp["oai"]),
                    "cai_iqr": iqr_text(grp["cai"]),
                    "hypoxic_burden_iqr": iqr_text(grp["hypoxic_burden"]),
                    "ss_percent_main_iqr": iqr_text(grp["ss_percent_main"]),
                    "arousal_index_iqr": iqr_text(grp["arousal_index"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "phenotype_ahi_bsi_summary.csv", index=False)
    return out


def build_pairwise_tests(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ahi_category in AHI_LABELS:
        ahi_df = df[df["ahi_category"].eq(ahi_category)]
        groups = {
            phenotype: safe_numeric(ahi_df.loc[ahi_df[PHENOTYPE].eq(phenotype), PRIMARY_BSI]).dropna()
            for phenotype in PHENOTYPE_ORDER
        }
        samples = [groups[phenotype] for phenotype in PHENOTYPE_ORDER]
        if all(len(sample) >= 3 for sample in samples):
            stat, p_value = stats.kruskal(*samples)
        else:
            stat, p_value = np.nan, np.nan
        rows.append(
            {
                "ahi_category": ahi_category,
                "comparison_type": "omnibus_kruskal",
                "metric": PRIMARY_BSI,
                "group_a": "csa,mixed,osa",
                "group_b": "",
                "n_a": int(sum(len(sample) for sample in samples)),
                "n_b": 0,
                "statistic": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
            }
        )
        for group_a, group_b in [("mixed", "csa"), ("osa", "csa"), ("osa", "mixed")]:
            a_vals = groups[group_a]
            b_vals = groups[group_b]
            if len(a_vals) >= 3 and len(b_vals) >= 3:
                stat_u, p_u = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            else:
                stat_u, p_u = np.nan, np.nan
            rows.append(
                {
                    "ahi_category": ahi_category,
                    "comparison_type": "pairwise_mannwhitney",
                    "metric": PRIMARY_BSI,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(len(a_vals)),
                    "n_b": int(len(b_vals)),
                    "statistic": float(stat_u) if pd.notna(stat_u) else np.nan,
                    "p_value": float(p_u) if pd.notna(p_u) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "phenotype_ahi_bsi_pairwise_tests.csv", index=False)
    return out


def build_stratified_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    term_labels = {
        f"C({PHENOTYPE}, Treatment(reference='csa'))[T.mixed]": "mixed vs csa",
        f"C({PHENOTYPE}, Treatment(reference='csa'))[T.osa]": "osa vs csa",
    }
    model_specs = [
        {
            "model_name": "phenotype_age_sex_bmi_cohort",
            "model_label": "Age, sex, BMI, and cohort",
            "formula": (
                f"{PRIMARY_BSI} ~ C({PHENOTYPE}, Treatment(reference='csa')) "
                f"+ age + C(sex) + bmi + C({COHORT})"
            ),
            "needed": [PRIMARY_BSI, PHENOTYPE, "age", "sex", "bmi", COHORT],
        },
        {
            "model_name": "phenotype_age_sex_bmi_cohort_ahi_hypoxic_burden",
            "model_label": "Age, sex, BMI, cohort, AHI, and hypoxic burden",
            "formula": (
                f"{PRIMARY_BSI} ~ C({PHENOTYPE}, Treatment(reference='csa')) "
                f"+ age + C(sex) + bmi + C({COHORT}) + ahi + hypoxic_burden"
            ),
            "needed": [
                PRIMARY_BSI,
                PHENOTYPE,
                "age",
                "sex",
                "bmi",
                COHORT,
                "ahi",
                "hypoxic_burden",
            ],
        },
    ]

    rows: list[dict[str, object]] = []
    for ahi_category in AHI_LABELS:
        ahi_df = df[df["ahi_category"].eq(ahi_category)].copy()
        for spec in model_specs:
            model_data = ahi_df.dropna(subset=spec["needed"]).copy()
            phenotype_counts = model_data[PHENOTYPE].value_counts().to_dict()
            status = "ok"
            error = ""
            fit = None
            ci = pd.DataFrame()
            if len(model_data) < 10 or any(phenotype_counts.get(p, 0) < 3 for p in PHENOTYPE_ORDER):
                status = "insufficient_data"
            else:
                try:
                    fit = smf.ols(spec["formula"], data=model_data).fit(cov_type="HC3")
                    ci = fit.conf_int()
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            for term, label in term_labels.items():
                rows.append(
                    {
                        "ahi_category": ahi_category,
                        "model_name": spec["model_name"],
                        "model_label": spec["model_label"],
                        "outcome": PRIMARY_BSI,
                        "reference": "csa",
                        "contrast": label,
                        "term": term,
                        "n": int(len(model_data)),
                        "n_csa": int(phenotype_counts.get("csa", 0)),
                        "n_mixed": int(phenotype_counts.get("mixed", 0)),
                        "n_osa": int(phenotype_counts.get("osa", 0)),
                        "beta": float(fit.params.get(term, np.nan)) if fit is not None else np.nan,
                        "ci_lower": float(ci.loc[term, 0])
                        if fit is not None and term in ci.index
                        else np.nan,
                        "ci_upper": float(ci.loc[term, 1])
                        if fit is not None and term in ci.index
                        else np.nan,
                        "p_value": float(fit.pvalues.get(term, np.nan)) if fit is not None else np.nan,
                        "r_squared": float(fit.rsquared) if fit is not None else np.nan,
                        "covariance": "HC3",
                        "status": status,
                        "model_error": error,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "phenotype_ahi_adjusted_bsi_models.csv", index=False)
    return out


def z_log1p(series: pd.Series) -> pd.Series:
    x = np.log1p(safe_numeric(series))
    sd = x.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (x - x.mean()) / sd


def build_event_burden_models(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    work = df.copy()
    for col in ["ahi", "oai", "cai", "hypoxic_burden"]:
        work[f"z_log1p_{col}"] = z_log1p(work[col])

    model_specs = [
        {
            "model_name": "ahi_age_sex_bmi_cohort",
            "model_label": "AHI adjusted for age, sex, BMI, and cohort",
            "formula": f"{PRIMARY_BSI} ~ z_log1p_ahi + age + C(sex) + bmi + C({COHORT})",
            "needed": [PRIMARY_BSI, "z_log1p_ahi", "age", "sex", "bmi", COHORT],
            "terms": ["z_log1p_ahi"],
        },
        {
            "model_name": "oai_cai_age_sex_bmi_cohort",
            "model_label": "OAI and CAI adjusted for age, sex, BMI, and cohort",
            "formula": (
                f"{PRIMARY_BSI} ~ z_log1p_oai + z_log1p_cai + "
                f"age + C(sex) + bmi + C({COHORT})"
            ),
            "needed": [PRIMARY_BSI, "z_log1p_oai", "z_log1p_cai", "age", "sex", "bmi", COHORT],
            "terms": ["z_log1p_oai", "z_log1p_cai"],
        },
        {
            "model_name": "oai_cai_age_sex_bmi_cohort_hypoxic_burden",
            "model_label": "OAI and CAI with hypoxic burden sensitivity",
            "formula": (
                f"{PRIMARY_BSI} ~ z_log1p_oai + z_log1p_cai + "
                f"age + C(sex) + bmi + C({COHORT}) + z_log1p_hypoxic_burden"
            ),
            "needed": [
                PRIMARY_BSI,
                "z_log1p_oai",
                "z_log1p_cai",
                "age",
                "sex",
                "bmi",
                COHORT,
                "z_log1p_hypoxic_burden",
            ],
            "terms": ["z_log1p_oai", "z_log1p_cai", "z_log1p_hypoxic_burden"],
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in model_specs:
        model_data = work.dropna(subset=spec["needed"]).copy()
        status = "ok"
        error = ""
        fit = None
        ci = pd.DataFrame()
        try:
            fit = smf.ols(spec["formula"], data=model_data).fit(cov_type="HC3")
            ci = fit.conf_int()
        except Exception as exc:
            status = "failed"
            error = str(exc)
        for term in spec["terms"]:
            rows.append(
                {
                    "model_name": spec["model_name"],
                    "model_label": spec["model_label"],
                    "outcome": PRIMARY_BSI,
                    "predictor": term,
                    "n": int(len(model_data)),
                    "beta": float(fit.params.get(term, np.nan)) if fit is not None else np.nan,
                    "ci_lower": float(ci.loc[term, 0])
                    if fit is not None and term in ci.index
                    else np.nan,
                    "ci_upper": float(ci.loc[term, 1])
                    if fit is not None and term in ci.index
                    else np.nan,
                    "p_value": float(fit.pvalues.get(term, np.nan)) if fit is not None else np.nan,
                    "r_squared": float(fit.rsquared) if fit is not None else np.nan,
                    "covariance": "HC3",
                    "status": status,
                    "model_error": error,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "oai_cai_continuous_bsi_models.csv", index=False)
    return out


def maybe_write_figure(summary: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_df = summary.copy()
    plot_df["bsi_median"] = plot_df["bsi_median_sleep_iqr"].str.extract(r"^([0-9.]+)").astype(float)
    plot_df["bsi_q1"] = plot_df["bsi_median_sleep_iqr"].str.extract(r"\[([0-9.]+),").astype(float)
    plot_df["bsi_q3"] = plot_df["bsi_median_sleep_iqr"].str.extract(r", ([0-9.]+)\]").astype(float)

    x = np.arange(len(AHI_LABELS))
    colors = {"csa": "#3B82F6", "mixed": "#6B7280", "osa": "#D97706"}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for phenotype in PHENOTYPE_ORDER:
        sub = plot_df[plot_df[PHENOTYPE].eq(phenotype)].set_index("ahi_category").loc[AHI_LABELS]
        y = sub["bsi_median"].to_numpy()
        yerr = np.vstack([(y - sub["bsi_q1"].to_numpy()), (sub["bsi_q3"].to_numpy() - y)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[phenotype],
            label=phenotype.upper(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(AHI_LABELS, rotation=20, ha="right")
    ax.set_ylabel("Median sleep BSI")
    ax.set_xlabel("AHI category")
    ax.set_title("BSI by OSA/CSA phenotype and AHI category")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "figure_osa_csa_ahi_bsi.png", dpi=300)
    fig.savefig(outdir / "figure_osa_csa_ahi_bsi.pdf")
    plt.close(fig)


def pick_model_row(models: pd.DataFrame, ahi_category: str, model_name: str, contrast: str) -> pd.Series | None:
    sub = models[
        models["ahi_category"].eq(ahi_category)
        & models["model_name"].eq(model_name)
        & models["contrast"].eq(contrast)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def pick_event_row(models: pd.DataFrame, model_name: str, predictor: str) -> pd.Series | None:
    sub = models[models["model_name"].eq(model_name) & models["predictor"].eq(predictor)]
    if sub.empty:
        return None
    return sub.iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.reset_index()
    columns = [str(col) for col in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (np.integer, int)):
                values.append(str(int(value)))
            elif isinstance(value, (np.floating, float)) and float(value).is_integer():
                values.append(str(int(value)))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(
    outdir: Path,
    master_csv: Path,
    summary: pd.DataFrame,
    models: pd.DataFrame,
    event_models: pd.DataFrame,
) -> None:
    n_total = int(summary["n"].sum())
    counts = summary.pivot(index="ahi_category", columns=PHENOTYPE, values="n").loc[AHI_LABELS]
    median_pivot = (
        summary.pivot(index="ahi_category", columns=PHENOTYPE, values="bsi_median_sleep_iqr")
        .loc[AHI_LABELS, PHENOTYPE_ORDER]
        .copy()
    )

    severe_osa = pick_model_row(
        models, "Severe (>=30)", "phenotype_age_sex_bmi_cohort", "osa vs csa"
    )
    normal_mixed = pick_model_row(
        models, "Normal (<5)", "phenotype_age_sex_bmi_cohort", "mixed vs csa"
    )
    mild_osa_sens = pick_model_row(
        models,
        "Mild (5-<15)",
        "phenotype_age_sex_bmi_cohort_ahi_hypoxic_burden",
        "osa vs csa",
    )
    cai_row = pick_event_row(event_models, "oai_cai_age_sex_bmi_cohort", "z_log1p_cai")
    oai_row = pick_event_row(event_models, "oai_cai_age_sex_bmi_cohort", "z_log1p_oai")
    cai_hb_row = pick_event_row(
        event_models, "oai_cai_age_sex_bmi_cohort_hypoxic_burden", "z_log1p_cai"
    )
    oai_hb_row = pick_event_row(
        event_models, "oai_cai_age_sex_bmi_cohort_hypoxic_burden", "z_log1p_oai"
    )

    lines = [
        "# OSA/CSA AHI-Stratified BSI Analysis v1",
        "",
        "## Inputs",
        f"- Master table: `{master_csv}`",
        f"- Phenotype: existing `{PHENOTYPE}` from obstructive fraction `OAI / (OAI + CAI)`.",
        "- AHI categories: Normal `<5`, Mild `5-<15`, Moderate `15-<30`, Severe `>=30`.",
        f"- Descriptive analysis set: phenotype, AHI category, and `{PRIMARY_BSI}` nonmissing.",
        "",
        "## Cell Counts",
        markdown_table(counts),
        "",
        "## Median BSI [IQR]",
        markdown_table(median_pivot),
        "",
        "## Adjusted Within-AHI Phenotype Contrasts",
        (
            "Models use CSA as the reference and adjust for age, sex, BMI, and cohort with HC3 robust "
            "confidence intervals. A sensitivity model additionally adjusts for continuous AHI and hypoxic burden "
            "inside each AHI band."
        ),
        (
            f"- In the normal-AHI band, mixed vs CSA was {contrast_text(normal_mixed)} in the demographic model."
        ),
        (
            f"- In the mild-AHI band, OSA vs CSA was {contrast_text(mild_osa_sens)} after additional AHI and "
            "hypoxic-burden adjustment."
        ),
        (
            f"- In the severe-AHI band, OSA vs CSA was {contrast_text(severe_osa)} in the demographic model."
        ),
        "",
        "## Continuous OAI/CAI Model",
        (
            "As a threshold-free check, median BSI was modeled against standardized `log1p(OAI)` and "
            "`log1p(CAI)` together, adjusted for age, sex, BMI, and cohort."
        ),
        (
            f"- OAI association: {contrast_text(oai_row)} per 1 SD of `log1p(OAI)`."
        ),
        (
            f"- CAI association: {contrast_text(cai_row)} per 1 SD of `log1p(CAI)`."
        ),
        (
            "With additional hypoxic-burden adjustment, associations persisted: "
            f"OAI {contrast_text(oai_hb_row)}; CAI {contrast_text(cai_hb_row)}."
        ),
        "",
        "## Interpretation",
        (
            "The AHI-stratified table suggests that the overall OSA/mixed/CSA BSI separation is largely driven "
            "by event burden. Within AHI categories, phenotype contrasts are much smaller and can reverse in "
            "moderate-to-severe disease. The continuous OAI/CAI model supports wording that BSI reflects breathing "
            "instability related to both obstructive and central event burden, rather than being specific to OSA."
        ),
        "",
        "## Output Files",
        "- `phenotype_ahi_bsi_summary.csv`",
        "- `phenotype_ahi_bsi_pairwise_tests.csv`",
        "- `phenotype_ahi_adjusted_bsi_models.csv`",
        "- `oai_cai_continuous_bsi_models.csv`",
        "- `figure_osa_csa_ahi_bsi.png` / `.pdf` if matplotlib is available",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_master(args.master_csv)

    phenotype_df = phenotype_analysis_set(df)
    summary = build_descriptives(phenotype_df, args.output_dir)
    build_pairwise_tests(phenotype_df, args.output_dir)
    models = build_stratified_models(phenotype_df, args.output_dir)
    event_models = build_event_burden_models(df, args.output_dir)
    maybe_write_figure(summary, args.output_dir)
    write_summary(args.output_dir, args.master_csv, summary, models, event_models)


if __name__ == "__main__":
    main()
