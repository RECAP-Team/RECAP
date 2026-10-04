"""
Compute spBLEU (sacrebleu tokenize="flores200") and chrF++ (word_order=2)
from each model's saved prediction/reference text, for BOTH val and test
splits. Writes one CSV per model into this directory (one row per
language+direction, val and test side by side: language, direction,
val_rows, val_spBLEU, val_chrF++, test_rows, test_spBLEU, test_chrF++),
covering
Bhili/Mundari/Gondi only (no Marathi -- never in scope here, and Marathi
only ever has hi2tgt/tgt2hi since its data has no English column).

All 5 models (mt5, NLLB, Qwen, Llama, Sarvam) now share an identical
inference-output convention (verified directly against each model's
infer.py / infer_sarvam.py, and the infer_config.json files -- all
repointed at RECAP/datasets/<dir_name>/{val,test}.csv, NOT the older,
now-deleted GRPO_RESEARCH/datasets/inference_data/<lang>.csv predictions
used before):
    <model_root>/<Lang>/infer_predictions_<lang>_<direction>_<split>.csv
    columns: source_<src_col.lower()>, prediction, reference_<tgt_col.lower()>

A (language, direction, split) whose prediction file doesn't exist yet (job
not run, or no reference column -- a pure-inference run with no ground
truth) is skipped with a printed warning, not a crash -- run again later
once more jobs finish and it'll pick up whatever's newly there.

Run (on Pragya -- needs real network access once, to auto-download the
FLORES-200 SentencePiece model sacrebleu's flores200 tokenizer uses; cached
under ~/.cache after that):
    python compute_spbleu_scores.py
    python compute_spbleu_scores.py --recap_root /custom/path
"""

import argparse
from pathlib import Path

import pandas as pd
import sacrebleu

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Bhili", "Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
SPLITS = ["val", "test"]

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
    ("mT5",           "mt5_scores",    "mt5_finetune"),
    ("NLLB",          "nllb_scores",   "NLLB-finetune"),
    ("Qwen2.5-0.5B",  "qwen_scores",   "qwen_finetune"),
    ("Llama-3.1-8B",  "llama_scores",  "llama_finetune"),
    ("Sarvam-Translate", "sarvam_scores", "sarvam_finetune"),
]


def score_pair(hyps, refs):
    sp_bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    return sp_bleu, chrfpp


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
                sp_bleu, chrfpp = score_pair(hyps, refs)
                rows.append({"language": lang, "direction": direction, "split": split, "rows": len(df),
                             "spBLEU": round(sp_bleu, 4), "chrF++": round(chrfpp, 4)})
                print(f"[ok] {model_root_name}/{lang}/{direction}/{split}: {len(df)} rows "
                      f"spBLEU={sp_bleu:.4f} chrF++={chrfpp:.4f}")
    return rows


def pivot_to_wide(rows):
    """One row per (language, direction), val and test side by side --
    e.g. language, direction, val_rows, val_spBLEU, val_chrF++, test_rows,
    test_spBLEU, test_chrF++. Separate val_rows/test_rows rather than one
    shared rows column since the two splits can have different sizes."""
    by_key = {}
    for r in rows:
        key = (r["language"], r["direction"])
        entry = by_key.setdefault(key, {"language": r["language"], "direction": r["direction"]})
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

    cols = ["language", "direction",
            "val_rows", "val_spBLEU", "val_chrF++",
            "test_rows", "test_spBLEU", "test_chrF++"]

    for display_name, out_stem, model_root_name in MODELS:
        print(f"\n===== {display_name} =====")
        rows = score_model(args.recap_root, model_root_name)
        wide_rows = pivot_to_wide(rows)
        out_path = out_dir / f"{out_stem}.csv"
        pd.DataFrame(wide_rows, columns=cols).to_csv(out_path, index=False)
        print(f"-> {out_path} ({len(wide_rows)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
