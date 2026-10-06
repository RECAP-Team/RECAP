"""
Compute SENTENCE-LEVEL spBLEU (sacrebleu tokenize="flores200") and chrF++
(word_order=2) for each model's saved prediction/reference text, then
AVERAGE across sentences -- this is deliberately different from
spBLEU_scores/compute_spbleu_scores.py, which computes a single CORPUS-level
score (the standard, recommended way sacreBLEU itself documents). This
script instead scores every (prediction, reference) pair individually and
takes the mean of those per-sentence scores, for direct comparison against
the corpus-level numbers.

Standalone (no imports from spBLEU_scores/) -- same file-discovery/column
convention duplicated here: all 5 models (mt5, NLLB, Qwen, Llama, Sarvam)
share the same inference-output layout:
    <model_root>/<Lang>/infer_predictions_<lang>_<direction>_<split>.csv
    columns: source_<src_col.lower()>, prediction, reference_<tgt_col.lower()>

Covers Bhili/Mundari/Gondi only (no Marathi), all 4 directions, both val and
test splits. A (language, direction, split) whose prediction file doesn't
exist yet is skipped with a printed warning, not a crash.

The BLEU/CHRF metric objects (and the FLORES-200 SentencePiece tokenizer
they load) are constructed ONCE and reused across every sentence, not
reconstructed per row -- reconstructing per sentence would reload the SPM
model tens of thousands of times and be extremely slow.

Run (on Pragya -- needs real network access once, to auto-download the
FLORES-200 SentencePiece model; cached under ~/.cache after that):
    python compute_sentence_spbleu_scores.py
    python compute_sentence_spbleu_scores.py --recap_root /custom/path
"""

import argparse
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Bhili", "Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
SPLITS = ["val", "test"]

# Built once, reused for every sentence -- see module docstring.
# effective_order=True is sacrebleu's own recommended setting specifically
# for SENTENCE-level BLEU (not needed/used for corpus-level): without it,
# any sentence shorter than 4 tokens gets a hard BLEU=0 regardless of
# quality, since it can't form a 4-gram at all -- effective_order instead
# only considers the n-gram orders that sentence can actually have.
_BLEU_METRIC = BLEU(tokenize="flores200", effective_order=True)
_CHRF_METRIC = CHRF(word_order=2)

# What the *target* (reference) column is named for each direction, per
# infer.py's own `tgt_col`/`reference_{tgt_col.lower()}` convention.
TGT_COL_BY_DIRECTION = {
    "hi2tgt": lambda lang: lang,
    "en2tgt": lambda lang: lang,
    "tgt2hi": lambda lang: "Hindi",
    "tgt2en": lambda lang: "English",
}

# (display name, output csv stem, <model_root>/<Lang>/ directory name)
MODELS = [
    ("mT5",              "mt5_scores",    "mt5_finetune"),
    ("NLLB",             "nllb_scores",   "NLLB-finetune"),
    ("Qwen2.5-0.5B",     "qwen_scores",   "qwen_finetune"),
    ("Llama-3.1-8B",     "llama_scores",  "llama_finetune"),
    ("Sarvam-Translate", "sarvam_scores", "sarvam_finetune"),
]


def score_pair_sentence_avg(hyps, refs):
    """Score every (hyp, ref) pair individually with the SAME metric
    objects, then average -- NOT a corpus-level aggregate score."""
    bleu_scores = [_BLEU_METRIC.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    chrf_scores = [_CHRF_METRIC.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_chrf = sum(chrf_scores) / len(chrf_scores)
    return avg_bleu, avg_chrf


def score_model(recap_root, model_root_name):
    rows = []
    for lang in LANGUAGES:
        lang_dir = Path(recap_root) / model_root_name / lang
        for direction in DIRECTIONS:
            for split in SPLITS:
                job_name = f"{lang.lower()}_{direction}_{split}"
                preds_path = lang_dir / f"infer_predictions_{job_name}.csv"
                if not preds_path.exists():
                    print(f"[skip] {model_root_name}/{lang}/{direction}/{split}: not found ({preds_path})")
                    continue
                df = pd.read_csv(preds_path)
                tgt_col = TGT_COL_BY_DIRECTION[direction](lang)
                ref_col = f"reference_{tgt_col.lower()}"
                if ref_col not in df.columns:
                    print(f"[skip] {model_root_name}/{lang}/{direction}/{split}: no reference column "
                          f"{ref_col!r} in {preds_path} (pure-inference run, no ground truth?)")
                    continue
                hyps = df["prediction"].astype(str).tolist()
                refs = df[ref_col].astype(str).tolist()
                avg_bleu, avg_chrf = score_pair_sentence_avg(hyps, refs)
                rows.append({"language": lang, "direction": direction, "split": split, "rows": len(df),
                             "spBLEU": round(avg_bleu, 4), "chrF++": round(avg_chrf, 4)})
                print(f"[ok] {model_root_name}/{lang}/{direction}/{split}: {len(df)} rows "
                      f"avg spBLEU={avg_bleu:.4f} avg chrF++={avg_chrf:.4f}")
    return rows


def pivot_to_wide(rows, model_name):
    """One row per (language, direction), val and test side by side."""
    by_key = {}
    for r in rows:
        key = (r["language"], r["direction"])
        entry = by_key.setdefault(
            key, {"model": model_name, "language": r["language"], "direction": r["direction"]})
        entry[f"{r['split']}_rows"] = r["rows"]
        entry[f"{r['split']}_spBLEU"] = r["spBLEU"]
        entry[f"{r['split']}_chrF++"] = r["chrF++"]
    return list(by_key.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recap_root",
                     default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP")
    ap.add_argument("--out_dir", default=str(HERE))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["model", "language", "direction",
            "val_rows", "val_spBLEU", "val_chrF++",
            "test_rows", "test_spBLEU", "test_chrF++"]

    for display_name, out_stem, model_root_name in MODELS:
        print(f"\n===== {display_name} =====")
        rows = score_model(args.recap_root, model_root_name)
        wide_rows = pivot_to_wide(rows, display_name)
        out_path = out_dir / f"{out_stem}.csv"
        pd.DataFrame(wide_rows, columns=cols).to_csv(out_path, index=False)
        print(f"-> {out_path} ({len(wide_rows)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
