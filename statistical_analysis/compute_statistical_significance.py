"""
Paired bootstrap resampling statistical significance testing, replicating
2025.findings-emnlp.508.pdf Appendix A.10.5 "Statistical Significance
Testing" as closely as possible:
  - Method: paired bootstrap resampling, 1,000 iterations, following
    established practice in MT evaluation (Koehn, 2004).
  - Metric: segment-level chrF++ (word_order=2) ONLY -- the paper's own
    choice, since they found chrF++ correlates most closely with human
    judgments (their Section 5.1). spBLEU/BLEU are NOT used here, by design
    -- this mirrors the paper exactly, not an oversight.
  - Split: TEST set only (same as the paper: "we repeatedly resampled the
    test set with replacement").
  - For each (language, direction), every other model is compared against
    ONE baseline model, producing Delta chrF++ [95% CI] and a two-tailed
    p-value per model -- the same table shape as the paper's Tables 11-14
    (one table per language x direction).
  - The paper reports no plots for this analysis (A.10.5 is tables only,
    Tables 11-14) -- so none are produced here either, to match exactly.

Reads predictions from the CORRECTED checkpoints under
correct_checkpoints/<model>/<lang>/ (dataset-corrected finetunes), NOT the
older <model>_finetune/ trees this script originally pointed at. Each
model's finetune/infer script saves its own filename and column-naming
convention there -- captured per-model in MODEL_SPECS/BASELINE_SPEC below,
read directly off each script rather than assumed:
    nllb/mt5/qwen:  <root>/<lang>/test_predictions_<lang>_<direction>.csv
                    columns: prediction_<tgt>, reference_<tgt>
    sarvam_translate: <root>/<lang>/test_predictions_<lang>_<direction>.csv
                    columns: prediction, reference  (no suffix)
    indictrans2:    <root>/<lang>/infer_predictions_<lang>_<direction>.csv
                    columns: prediction (no suffix), reference_<tgt>

Baseline: Sarvam-Translate (BASELINE_SPEC below) -- chosen because it has
the best overall average test spBLEU/chrF++ across all 12 (language,
direction) combinations among the models currently in scope, mirroring the
paper's own approach of using its strongest/main system (NLLB-200) as the
reference every other model is compared against.

Models in scope (5): NLLB, mT5, Qwen2.5-1.5B, IndicTrans2 -- compared
against the Sarvam baseline. Any (language, direction) a model has no data
for is simply skipped with a printed warning, same as any other missing file.

p-value: two-tailed, estimated from the 1,000-sample bootstrap distribution
of (other_model - baseline) differences as 2 * min(P(diff <= 0),
P(diff >= 0)), capped at 1.0 -- the standard percentile-based two-tailed
bootstrap significance test (Koehn, 2004).

Output: one CSV per (language, direction) in this directory (e.g.
Bhili_hi2tgt_significance.csv, matching the paper's Tables 11-14 shape:
model, delta_chrF++, ci_low, ci_high, p_value), plus one combined
all_significance_results.csv with every (language, direction, model) row.

Run (on Pragya -- reuses already-saved test_predictions_*.csv /
infer_predictions_*.csv, no inference rerun needed):
    python compute_statistical_significance.py
    python compute_statistical_significance.py --n_bootstrap 1000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Bhili", "Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]

# Built once, reused for every sentence (see spBLEU_scores' sentence-level
# scripts for the same pattern).
_CHRF_METRIC = CHRF(word_order=2)

N_BOOTSTRAP = 1000
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5

# What the *target* (reference) column is named for each direction, per
# each finetune/infer script's own `tgt_col`/`reference_{tgt_col.lower()}`
# convention (Hindi/English/<Lang> are the literal column names used).
TGT_COL_BY_DIRECTION = {
    "hi2tgt": lambda lang: lang,
    "en2tgt": lambda lang: lang,
    "tgt2hi": lambda lang: "Hindi",
    "tgt2en": lambda lang: "English",
}

# (display name, dir name under correct_checkpoints/, filename prefix,
#  prediction-column style, reference-column style)
#   filename prefix: "test_predictions" (nllb/mt5/qwen/sarvam) or
#                     "infer_predictions" (indictrans2)
#   column style: "suffixed" -> f"{prefix}_{tgt_col.lower()}"
#                 "plain"    -> just "prediction" / "reference"
# (verified directly against each model's finetune_*.py/infer_*.py .to_csv
# call, not assumed -- see module docstring table.)
BASELINE_SPEC = ("Sarvam-Translate", "sarvam_translate", "test_predictions", "plain", "plain")

MODEL_SPECS = [
    ("NLLB",            "nllb",        "test_predictions",  "suffixed", "suffixed"),
    ("mT5",             "mt5",         "test_predictions",  "suffixed", "suffixed"),
    ("Qwen2.5-1.5B",    "qwen",        "test_predictions",  "suffixed", "suffixed"),
    ("IndicTrans2",     "indictrans2", "infer_predictions", "plain",    "suffixed"),
]


def get_preds_path(recap_root, dir_name, filename_prefix, lang, direction):
    return (Path(recap_root) / "correct_checkpoints" / dir_name / lang.lower()
            / f"{filename_prefix}_{lang.lower()}_{direction}.csv")


def load_segment_chrf(recap_root, dir_name, filename_prefix, pred_style, ref_style, lang, direction):
    """Returns a list of per-segment chrF++ scores (one per test-set row),
    or None if the prediction file / reference column isn't there yet."""
    preds_path = get_preds_path(recap_root, dir_name, filename_prefix, lang, direction)
    if not preds_path.exists():
        return None, preds_path
    df = pd.read_csv(preds_path)
    tgt_col = TGT_COL_BY_DIRECTION[direction](lang)
    pred_col = "prediction" if pred_style == "plain" else f"prediction_{tgt_col.lower()}"
    ref_col = "reference" if ref_style == "plain" else f"reference_{tgt_col.lower()}"
    if pred_col not in df.columns or ref_col not in df.columns:
        return None, preds_path
    hyps = df[pred_col].astype(str).tolist()
    refs = df[ref_col].astype(str).tolist()
    scores = [_CHRF_METRIC.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    return scores, preds_path


def paired_bootstrap_test(baseline_scores, other_scores, rng, n_bootstrap=N_BOOTSTRAP):
    """Paired bootstrap resampling (Koehn, 2004): the SAME resampled
    indices are applied to both score arrays on every iteration, since
    they're paired per-segment (same underlying test sentences)."""
    baseline_scores = np.asarray(baseline_scores)
    other_scores = np.asarray(other_scores)
    n = len(baseline_scores)
    assert n == len(other_scores), "baseline/other segment counts must match (same test set)"

    observed_diff = other_scores.mean() - baseline_scores.mean()

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = other_scores[idx].mean() - baseline_scores[idx].mean()

    ci_low, ci_high = np.percentile(diffs, [CI_LOW_PCT, CI_HIGH_PCT])
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return observed_diff, ci_low, ci_high, p_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recap_root",
                     default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP")
    ap.add_argument("--out_dir", default=str(HERE))
    ap.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    baseline_name, baseline_dir, baseline_prefix, baseline_pred_style, baseline_ref_style = BASELINE_SPEC
    all_rows = []

    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            print(f"\n===== {lang}/{direction} (baseline: {baseline_name}) =====")
            baseline_scores, baseline_path = load_segment_chrf(
                args.recap_root, baseline_dir, baseline_prefix, baseline_pred_style, baseline_ref_style, lang, direction)
            if baseline_scores is None:
                print(f"[skip] baseline not found/no reference: {baseline_path}")
                continue

            table_rows = []
            for model_name, dir_name, filename_prefix, pred_style, ref_style in MODEL_SPECS:
                other_scores, preds_path = load_segment_chrf(
                    args.recap_root, dir_name, filename_prefix, pred_style, ref_style, lang, direction)
                if other_scores is None:
                    print(f"[skip] {model_name}: not found/no reference ({preds_path})")
                    continue
                if len(other_scores) != len(baseline_scores):
                    print(f"[skip] {model_name}: segment count mismatch "
                          f"({len(other_scores)} vs baseline {len(baseline_scores)})")
                    continue

                delta, ci_low, ci_high, p = paired_bootstrap_test(baseline_scores, other_scores, rng, args.n_bootstrap)
                print(f"  {model_name}: Delta chrF++ = {delta:+.2f} [{ci_low:+.2f}, {ci_high:+.2f}]  p={p:.4f}")
                row = {"language": lang, "direction": direction, "baseline": baseline_name,
                       "model": model_name, "delta_chrF++": round(delta, 4),
                       "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4), "p_value": round(p, 4),
                       "n_segments": len(baseline_scores)}
                table_rows.append(row)
                all_rows.append(row)

            if table_rows:
                out_path = out_dir / f"{lang}_{direction}_significance.csv"
                pd.DataFrame(table_rows).to_csv(out_path, index=False)
                print(f"  -> {out_path}")

    if all_rows:
        combined_path = out_dir / "all_significance_results.csv"
        pd.DataFrame(all_rows).to_csv(combined_path, index=False)
        print(f"\n-> {combined_path} ({len(all_rows)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
