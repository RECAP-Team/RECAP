"""
Stage 4 of the human-evaluation pipeline -- takes the two TRI annotators'
FILLED-IN copies of <Lang>_human_eval.csv (the blind CSV from stage 2,
columns: source_sentence, direction, translation_1..4, Direct_Assessment_1..4,
Error_type_1..4, Severity_1..4) and reproduces the correlation/agreement
analysis from 2025.findings-emnlp.508.pdf Section 5.1, scaled to this
project's 4 models / 2 languages / 4 directions:

  - De-anonymizes both annotators' scores back to real model names, using
    the KEY file from stage 2 (row position, NOT a row_id column -- the
    distributed CSV never had one -- so this script first verifies the
    returned CSV's source_sentence/direction columns still line up
    position-by-position against the original blank CSV before trusting
    the KEY's row order; it refuses to proceed if they don't).
  - Inter-Annotator Agreement (IAA): Pearson's r between the two
    annotators' Direct Assessment scores, per direction -- same "one
    coefficient per direction" shape as the paper's reported IAA numbers
    (0.60 for hin-bhb, 0.66 for bhb-hin, etc.), computed here across all
    annotated segments (not just a 50-segment calibration subset, since
    both TRIs annotated the full 400 rows).
  - Table 5 equivalent: segment-level Kendall's tau and Pearson's rho
    between each automatic metric (spBLEU, chrF++, from stage 3's
    <Lang>_human_eval_metrics.csv) and the human DA score (mean of the two
    annotators), per direction, pooled across all 4 models' segments.
  - Per-model mean DA score per direction (human quality ranking table).
  - Error-type and severity distributions per model.
  - Plots: mean-DA bar chart per model/direction, automatic-metric-vs-DA
    scatter plots per direction, error-type distribution bar chart. (The
    paper's Figures 7/8 are hand-curated qualitative translation examples,
    not something derivable from these CSVs, so they're out of scope here.)

Severity -> numeric mapping (fixed, as specified): good=1, low=2,
medium=3, high=4, veryhigh=5.

Run (CPU only, after both annotators return their filled CSVs):
    python analyze_human_eval_results.py --lang Mundari \\
        --annotator1_csv /path/to/Mundari_annotator1.csv \\
        --annotator2_csv /path/to/Mundari_annotator2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
N_SLOTS = 4
SEVERITY_MAP = {"good": 1, "low": 2, "medium": 3, "high": 4, "veryhigh": 5,
                 "very high": 5, "very_high": 5}


def normalize_severity(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    key = str(s).strip().lower()
    return SEVERITY_MAP.get(key, np.nan)


def normalize_error_type(e):
    if pd.isna(e) or str(e).strip() == "" or str(e).strip().lower() in ("none", "no error", "na", "n/a"):
        return "None"
    return str(e).strip()


def verify_row_alignment(original_df, annotator_df, annotator_label):
    """The distributed CSV had no row_id -- confirm the returned CSV's
    source_sentence/direction columns still match the original blank CSV
    position-by-position before trusting KEY row order for de-anonymizing."""
    assert len(original_df) == len(annotator_df), \
        f"{annotator_label}: row count {len(annotator_df)} != original {len(original_df)} -- rows added/removed?"
    mismatches = (original_df["source_sentence"].astype(str).str.strip() !=
                  annotator_df["source_sentence"].astype(str).str.strip())
    n_mismatch = mismatches.sum()
    assert n_mismatch == 0, (
        f"{annotator_label}: {n_mismatch} row(s) where source_sentence doesn't match the original "
        f"CSV at the same position -- rows were likely reordered/sorted/deleted. Cannot safely "
        f"de-anonymize. First mismatched row index: {mismatches.idxmax()}"
    )
    dir_mismatches = (original_df["direction"].astype(str).str.strip() !=
                       annotator_df["direction"].astype(str).str.strip())
    assert dir_mismatches.sum() == 0, f"{annotator_label}: direction column also misaligned"


def deanonymize(annotator_df, key_df, metrics_df, annotator_label):
    """Long format: one row per (original_row_index, slot) -> real model,
    DA, error_type, severity, spBLEU, chrF++, annotator."""
    rows = []
    for idx in range(len(annotator_df)):
        key_row = key_df.iloc[idx]
        metrics_row = metrics_df.iloc[idx]
        direction = key_row["direction"]
        row_id = key_row["row_id"]
        for slot in range(1, N_SLOTS + 1):
            model_name = key_row[f"translation_{slot}_model"]
            da = annotator_df.loc[idx, f"Direct_Assessment_{slot}"]
            err = normalize_error_type(annotator_df.loc[idx, f"Error_type_{slot}"])
            sev_raw = annotator_df.loc[idx, f"Severity_{slot}"]
            sev = normalize_severity(sev_raw)
            rows.append({
                "row_id": row_id, "direction": direction, "model": model_name,
                "annotator": annotator_label,
                "DA": pd.to_numeric(da, errors="coerce"),
                "Error_type": err, "Severity": sev,
                "spBLEU": metrics_row[f"spBLEU_{slot}"],
                "chrF++": metrics_row[f"chrF++_{slot}"],
            })
    return pd.DataFrame(rows)


def compute_iaa(long_df, directions):
    """Pearson's r between the two annotators' DA scores, per direction --
    one coefficient per direction, same shape as the paper's reported IAA."""
    rows = []
    a1 = long_df[long_df["annotator"] == "annotator1"]
    a2 = long_df[long_df["annotator"] == "annotator2"]
    merged = a1.merge(a2, on=["row_id", "direction", "model"], suffixes=("_a1", "_a2"))
    for direction in directions:
        sub = merged[merged["direction"] == direction].dropna(subset=["DA_a1", "DA_a2"])
        if len(sub) < 2:
            rows.append({"direction": direction, "n": len(sub), "IAA_pearson_r": np.nan})
            continue
        r, p = pearsonr(sub["DA_a1"], sub["DA_a2"])
        rows.append({"direction": direction, "n": len(sub), "IAA_pearson_r": round(r, 4), "p_value": round(p, 4)})
    overall = merged.dropna(subset=["DA_a1", "DA_a2"])
    if len(overall) >= 2:
        r, p = pearsonr(overall["DA_a1"], overall["DA_a2"])
        rows.append({"direction": "ALL", "n": len(overall), "IAA_pearson_r": round(r, 4), "p_value": round(p, 4)})
    return pd.DataFrame(rows), merged


def compute_metric_correlation(long_df, directions):
    """Table-5 equivalent: segment-level Kendall's tau / Pearson's rho
    between each automatic metric and mean human DA (across annotators),
    pooled across all models' segments within each direction."""
    mean_da = (long_df.groupby(["row_id", "direction", "model"])
               .agg(DA=("DA", "mean"), spBLEU=("spBLEU", "first"), chrF_pp=("chrF++", "first"))
               .reset_index())
    rows = []
    for direction in directions:
        sub = mean_da[mean_da["direction"] == direction].dropna(subset=["DA"])
        if len(sub) < 2:
            continue
        for metric_name, col in [("spBLEU", "spBLEU"), ("chrF++", "chrF_pp")]:
            tau, tau_p = kendalltau(sub[col], sub["DA"])
            rho, rho_p = pearsonr(sub[col], sub["DA"])
            rows.append({"direction": direction, "metric": metric_name, "n": len(sub),
                         "kendall_tau": round(tau, 4), "tau_p": round(tau_p, 4),
                         "pearson_rho": round(rho, 4), "rho_p": round(rho_p, 4)})
    return pd.DataFrame(rows), mean_da


def plot_mean_da_bar(mean_da, out_dir, lang):
    pivot = mean_da.groupby(["model", "direction"])["DA"].mean().unstack("direction")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Mean Direct Assessment (1-5)")
    ax.set_title(f"{lang}: Mean Human DA Score by Model and Direction")
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig(out_dir / f"{lang}_mean_DA_by_model_direction.png", dpi=150)
    plt.close()


def plot_metric_vs_da_scatter(mean_da, out_dir, lang, directions):
    fig, axes = plt.subplots(2, len(directions), figsize=(4 * len(directions), 8))
    for col_i, direction in enumerate(directions):
        sub = mean_da[mean_da["direction"] == direction]
        for row_i, (metric_name, col) in enumerate([("spBLEU", "spBLEU"), ("chrF++", "chrF_pp")]):
            ax = axes[row_i, col_i]
            ax.scatter(sub[col], sub["DA"], alpha=0.4, s=12)
            ax.set_title(f"{direction}: {metric_name} vs DA")
            ax.set_xlabel(metric_name)
            ax.set_ylabel("Human DA")
    plt.tight_layout()
    plt.savefig(out_dir / f"{lang}_metric_vs_DA_scatter.png", dpi=150)
    plt.close()


def plot_error_type_distribution(long_df, out_dir, lang):
    err = long_df[long_df["Error_type"] != "None"]
    if err.empty:
        return
    counts = err.groupby(["model", "Error_type"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_ylabel("Count of annotated errors (both annotators pooled)")
    ax.set_title(f"{lang}: Error Type Distribution by Model")
    plt.tight_layout()
    plt.savefig(out_dir / f"{lang}_error_type_distribution.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--annotator1_csv", required=True)
    ap.add_argument("--annotator2_csv", required=True)
    ap.add_argument("--original_csv", default=None, help="defaults to <lang>_human_eval.csv in this directory")
    ap.add_argument("--key_csv", default=None, help="defaults to <lang>_human_eval_KEY.csv in this directory")
    ap.add_argument("--metrics_csv", default=None, help="defaults to <lang>_human_eval_metrics.csv in this directory")
    ap.add_argument("--out_dir", default=None, help="defaults to <lang>_analysis/ in this directory")
    args = ap.parse_args()

    lang = args.lang
    original_csv = Path(args.original_csv) if args.original_csv else HERE / f"{lang}_human_eval.csv"
    key_csv = Path(args.key_csv) if args.key_csv else HERE / f"{lang}_human_eval_KEY.csv"
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else HERE / f"{lang}_human_eval_metrics.csv"
    out_dir = Path(args.out_dir) if args.out_dir else HERE / f"{lang}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(original_csv)
    key_df = pd.read_csv(key_csv)
    metrics_df = pd.read_csv(metrics_csv)
    a1_df = pd.read_csv(args.annotator1_csv)
    a2_df = pd.read_csv(args.annotator2_csv)

    verify_row_alignment(original_df, a1_df, "annotator1")
    verify_row_alignment(original_df, a2_df, "annotator2")
    assert len(key_df) == len(original_df) == len(metrics_df), \
        "original/KEY/metrics row counts don't match -- did stage 2/3 run on the same sample?"
    print(f"[ok] row alignment verified for both annotators ({len(original_df)} rows)")

    directions = list(dict.fromkeys(key_df["direction"].tolist()))  # preserve order, dedupe

    long1 = deanonymize(a1_df, key_df, metrics_df, "annotator1")
    long2 = deanonymize(a2_df, key_df, metrics_df, "annotator2")
    long_df = pd.concat([long1, long2], ignore_index=True)
    long_df.to_csv(out_dir / f"{lang}_deanonymized_long.csv", index=False)
    print(f"-> {out_dir / f'{lang}_deanonymized_long.csv'}")

    # ---- IAA ----
    iaa_df, _ = compute_iaa(long_df, directions)
    iaa_df.to_csv(out_dir / f"{lang}_IAA.csv", index=False)
    print(f"-> {out_dir / f'{lang}_IAA.csv'}")
    print(iaa_df.to_string(index=False))

    # ---- Table 5 equivalent: metric vs human DA correlation ----
    corr_df, mean_da = compute_metric_correlation(long_df, directions)
    corr_df.to_csv(out_dir / f"{lang}_metric_human_correlation.csv", index=False)
    print(f"-> {out_dir / f'{lang}_metric_human_correlation.csv'}")

    # ---- Per-model mean DA per direction ----
    model_table = mean_da.groupby(["model", "direction"])["DA"].mean().unstack("direction").round(3)
    model_table.to_csv(out_dir / f"{lang}_mean_DA_by_model.csv")
    print(f"-> {out_dir / f'{lang}_mean_DA_by_model.csv'}")

    # ---- Error type / severity distributions ----
    err_dist = (long_df[long_df["Error_type"] != "None"]
                .groupby(["model", "Error_type"]).size().unstack(fill_value=0))
    err_dist.to_csv(out_dir / f"{lang}_error_type_distribution.csv")
    sev_dist = long_df.dropna(subset=["Severity"]).groupby(["model"])["Severity"].mean().round(3)
    sev_dist.to_csv(out_dir / f"{lang}_mean_severity_by_model.csv")
    print(f"-> {out_dir / f'{lang}_error_type_distribution.csv'}")
    print(f"-> {out_dir / f'{lang}_mean_severity_by_model.csv'}")

    # ---- Plots ----
    plot_mean_da_bar(mean_da, out_dir, lang)
    plot_metric_vs_da_scatter(mean_da, out_dir, lang, directions)
    plot_error_type_distribution(long_df, out_dir, lang)
    print(f"-> plots saved to {out_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
