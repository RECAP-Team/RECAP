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

Baseline: Sarvam-Translate (BASELINE_MODEL below) -- chosen because it has
the best overall average test spBLEU/chrF++ across all 12 (language,
direction) combinations among the models currently in scope, mirroring the
paper's own approach of using its strongest/main system (NLLB-200) as the
reference every other model is compared against.

Models currently in scope (4): NLLB, mT5, Qwen -- compared against the
Sarvam baseline. IndicTrans2 is commented out in MODELS below until its
English-direction checkpoints exist (it also uses a different
file-layout -- see get_preds_path()); uncomment that one line and rerun for
the full 5-model sweep once available. No other code change needed -- a
(language, direction) IndicTrans2 has no data for (en2tgt/tgt2en) is simply
skipped with a printed warning, same as any other missing file.

p-value: two-tailed, estimated from the 1,000-sample bootstrap distribution
of (other_model - baseline) differences as 2 * min(P(diff <= 0),
P(diff >= 0)), capped at 1.0 -- the standard percentile-based two-tailed
bootstrap significance test (Koehn, 2004).

Output: one CSV per (language, direction) in this directory (e.g.
Bhili_hi2tgt_significance.csv, matching the paper's Tables 11-14 shape:
model, delta_chrF++, ci_low, ci_high, p_value), plus one combined
all_significance_results.csv with every (language, direction, model) row.

Run (on Pragya -- reuses already-saved infer_predictions_*.csv, no
inference rerun needed):
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
# infer.py's own `tgt_col`/`reference_{tgt_col.lower()}` convention.
TGT_COL_BY_DIRECTION = {
    "hi2tgt": lambda lang: lang,
    "en2tgt": lambda lang: lang,
    "tgt2hi": lambda lang: "Hindi",
    "tgt2en": lambda lang: "English",
}

# (display name, <root>/ directory name, path style)
# "nested_lang_split": <root>/<Lang>/infer_predictions_<lang>_<direction>_test.csv
# "flat_no_split":     <root>/infer_predictions_<lang>_<direction>.csv (IndicTrans2 --
#                       no per-language subdir, no val/test suffix: it only
#                       ever runs inference on test.csv)
BASELINE_MODEL = ("Sarvam-Translate", "sarvam_finetune", "nested_lang_split")

MODELS = [
    ("NLLB",          "NLLB-finetune", "nested_lang_split"),
    ("mT5",           "mt5_finetune",  "nested_lang_split"),
    ("Qwen2.5-0.5B",  "qwen_finetune", "nested_lang_split"),
    # Uncomment once IndicTrans2's English-direction checkpoints exist
    # (hi2tgt/tgt2hi only -- en2tgt/tgt2en rows are skipped automatically
    # for it, same as any other missing file):
    # ("IndicTrans2", "indictrans2_finetune", "flat_no_split"),
]


def get_preds_path(recap_root, root_name, style, lang, direction):
    if style == "nested_lang_split":
        return Path(recap_root) / root_name / lang / f"infer_predictions_{lang.lower()}_{direction}_test.csv"
    else:  # flat_no_split (IndicTrans2)
        return Path(recap_root) / root_name / f"infer_predictions_{lang.lower()}_{direction}.csv"


def load_segment_chrf(recap_root, root_name, style, lang, direction):
    """Returns a list of per-segment chrF++ scores (one per test-set row),
    or None if the prediction file / reference column isn't there yet."""
    preds_path = get_preds_path(recap_root, root_name, style, lang, direction)
    if not preds_path.exists():
        return None, preds_path
    df = pd.read_csv(preds_path)
    tgt_col = TGT_COL_BY_DIRECTION[direction](lang)
    ref_col = f"reference_{tgt_col.lower()}"
    if ref_col not in df.columns:
        return None, preds_path
    hyps = df["prediction"].astype(str).tolist()
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

    baseline_name, baseline_root, baseline_style = BASELINE_MODEL
    all_rows = []

    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            print(f"\n===== {lang}/{direction} (baseline: {baseline_name}) =====")
            baseline_scores, baseline_path = load_segment_chrf(
                args.recap_root, baseline_root, baseline_style, lang, direction)
            if baseline_scores is None:
                print(f"[skip] baseline not found/no reference: {baseline_path}")
                continue

            table_rows = []
            for model_name, root_name, style in MODELS:
                other_scores, preds_path = load_segment_chrf(
                    args.recap_root, root_name, style, lang, direction)
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
