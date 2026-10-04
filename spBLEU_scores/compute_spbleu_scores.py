"""
Compute spBLEU (sacrebleu tokenize="flores200") and chrF++ (word_order=2)
from each model's ALREADY-SAVED prediction/reference text -- no inference
rerun needed, spBLEU only differs from the sacreBLEU numbers already
computed elsewhere in this project by tokenization at scoring time. Writes
one CSV per model into this directory, covering Bhili/Mundari/Gondi only
(no Marathi -- it was never in scope here, and only ever has hi2tgt/tgt2hi
since its data has no English column):

  - mt5, NLLB, Qwen, Llama: hi2tgt, tgt2hi, en2tgt, tgt2en (4 directions).
    All four share an identical per-language infer.py convention (verified
    directly against mt5_finetune/Bhili/infer.py, NLLB-finetune/Bhili/infer.py,
    qwen_finetune/Bhili/infer.py, llama_finetune/Bhili/infer.py) --
    predictions at <model_root>/<Lang>/infer_predictions_<lang>_<direction>.csv,
    columns source_<src_col.lower()>, prediction, reference_<tgt_col.lower()>.
  - Sarvam: hi2tgt, tgt2hi only (English-direction training not finished at
    the time of writing). Different convention (sarvam_finetune.py's own
    run_job(), not a separate infer.py): predictions at
    sarvam_finetune/<Lang>/test_predictions_<lang>_<direction>.csv, plain
    source/reference/prediction columns.

A (language, direction) pair whose prediction file doesn't exist yet (job
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
FOUR_DIR_DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
SARVAM_DIRECTIONS = ["hi2tgt", "tgt2hi"]

# What the *target* (reference) column is named for each direction, per
# infer.py's own `tgt_col`/`reference_{tgt_col.lower()}` convention.
TGT_COL_BY_DIRECTION = {
    "hi2tgt": lambda lang: lang,
    "en2tgt": lambda lang: lang,
    "tgt2hi": lambda lang: "Hindi",
    "tgt2en": lambda lang: "English",
}

# (display name, output csv stem, <model_root>/<Lang>/infer.py directory name)
INFER_PY_MODELS = [
    ("mT5",           "mt5_scores",   "mt5_finetune"),
    ("NLLB",          "nllb_scores",  "NLLB-finetune"),
    ("Qwen2.5-0.5B",  "qwen_scores",  "qwen_finetune"),
    ("Llama-3.1-8B",  "llama_scores", "llama_finetune"),
]


def score_pair(hyps, refs):
    sp_bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    return sp_bleu, chrfpp


def score_infer_py_model(recap_root, model_root_name):
    rows = []
    for lang in LANGUAGES:
        lang_dir = Path(recap_root) / model_root_name / lang
        for direction in FOUR_DIR_DIRECTIONS:
            job_name = f"{lang.lower()}_{direction}"
            preds_path = lang_dir / f"infer_predictions_{job_name}.csv"
            if not preds_path.exists():
                print(f"[skip] {model_root_name}/{lang}/{direction}: not found ({preds_path})")
                continue
            df = pd.read_csv(preds_path)
            tgt_col = TGT_COL_BY_DIRECTION[direction](lang)
            ref_col = f"reference_{tgt_col.lower()}"
            if ref_col not in df.columns:
                print(f"[skip] {model_root_name}/{lang}/{direction}: no reference column "
                      f"{ref_col!r} in {preds_path} (pure-inference run, no ground truth?)")
                continue
            hyps = df["prediction"].astype(str).tolist()
            refs = df[ref_col].astype(str).tolist()
            sp_bleu, chrfpp = score_pair(hyps, refs)
            rows.append({"language": lang, "direction": direction, "rows": len(df),
                         "spBLEU": round(sp_bleu, 4), "chrF++": round(chrfpp, 4)})
            print(f"[ok] {model_root_name}/{lang}/{direction}: {len(df)} rows "
                  f"spBLEU={sp_bleu:.4f} chrF++={chrfpp:.4f}")
    return rows


def score_sarvam(recap_root):
    rows = []
    for lang in LANGUAGES:
        lang_dir = Path(recap_root) / "sarvam_finetune" / lang
        for direction in SARVAM_DIRECTIONS:
            preds_path = lang_dir / f"test_predictions_{lang.lower()}_{direction}.csv"
            if not preds_path.exists():
                print(f"[skip] sarvam/{lang}/{direction}: not found ({preds_path})")
                continue
            df = pd.read_csv(preds_path)
            hyps = df["prediction"].astype(str).tolist()
            refs = df["reference"].astype(str).tolist()
            sp_bleu, chrfpp = score_pair(hyps, refs)
            rows.append({"language": lang, "direction": direction, "rows": len(df),
                         "spBLEU": round(sp_bleu, 4), "chrF++": round(chrfpp, 4)})
            print(f"[ok] sarvam/{lang}/{direction}: {len(df)} rows "
                  f"spBLEU={sp_bleu:.4f} chrF++={chrfpp:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recap_root",
                     default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP")
    ap.add_argument("--out_dir", default=str(HERE))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["language", "direction", "rows", "spBLEU", "chrF++"]

    for display_name, out_stem, model_root_name in INFER_PY_MODELS:
        print(f"\n===== {display_name} =====")
        rows = score_infer_py_model(args.recap_root, model_root_name)
        out_path = out_dir / f"{out_stem}.csv"
        pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
        print(f"-> {out_path} ({len(rows)} rows)")

    print("\n===== Sarvam-Translate =====")
    rows = score_sarvam(args.recap_root)
    out_path = out_dir / "sarvam_scores.csv"
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    print(f"-> {out_path} ({len(rows)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
