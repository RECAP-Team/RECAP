"""
Sentence-level spBLEU (sacrebleu tokenize="flores200") and chrF++
(word_order=2) for IndicTrans2, averaged across sentences -- same
methodology as compute_sentence_spbleu_scores.py (this directory's sibling
script for mT5/NLLB/Qwen/Llama/Sarvam), but IndicTrans2 uses a DIFFERENT
inference-output convention (per infer_indictrans2.py):
    indictrans2_finetune/infer_predictions_<lang>_<direction>.csv
    columns: source_hindi/source_<lang>, prediction, reference_<lang>/reference_hindi
-- no per-language subdirectory, no val/test split suffix (infer_indictrans2.py
only ever runs on test.csv, there is no val inference), and only 2
directions (hi2tgt, tgt2hi -- IndicTrans2 has no English-pair directions,
and at the time of writing only the Hindi-direction checkpoints exist).

Standalone (no imports from the other scripts). Covers Bhili/Mundari/Gondi
only (no Marathi). A (language, direction) whose prediction file doesn't
exist yet is skipped with a printed warning, not a crash.

The BLEU/CHRF metric objects (and the FLORES-200 SentencePiece tokenizer
they load) are constructed ONCE and reused across every sentence, not
reconstructed per row.

Run (on Pragya -- needs real network access once, to auto-download the
FLORES-200 SentencePiece model; cached under ~/.cache after that):
    python compute_sentence_spbleu_indictrans2.py
    python compute_sentence_spbleu_indictrans2.py --recap_root /custom/path
"""

import argparse
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Bhili", "Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]  # IndicTrans2 only -- see module docstring

# Built once, reused for every sentence -- see module docstring.
# effective_order=True is sacrebleu's own recommended setting specifically
# for SENTENCE-level BLEU: without it, any sentence shorter than 4 tokens
# gets a hard BLEU=0 regardless of quality, since it can't form a 4-gram.
_BLEU_METRIC = BLEU(tokenize="flores200", effective_order=True)
_CHRF_METRIC = CHRF(word_order=2)


def score_pair_sentence_avg(hyps, refs):
    """Score every (hyp, ref) pair individually with the SAME metric
    objects, then average -- NOT a corpus-level aggregate score."""
    bleu_scores = [_BLEU_METRIC.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    chrf_scores = [_CHRF_METRIC.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_chrf = sum(chrf_scores) / len(chrf_scores)
    return avg_bleu, avg_chrf


def score_model(indictrans2_root):
    rows = []
    for lang in LANGUAGES:
        for direction in DIRECTIONS:
            job_name = f"{lang.lower()}_{direction}"
            preds_path = Path(indictrans2_root) / f"infer_predictions_{job_name}.csv"
            if not preds_path.exists():
                print(f"[skip] {lang}/{direction}: not found ({preds_path})")
                continue
            df = pd.read_csv(preds_path)
            # hi2tgt -> reference_<lang>; tgt2hi -> reference_hindi (per
            # infer_indictrans2.py's own src_col_name/ref_col_name convention)
            ref_col = f"reference_{lang.lower()}" if direction == "hi2tgt" else "reference_hindi"
            if ref_col not in df.columns:
                print(f"[skip] {lang}/{direction}: no reference column {ref_col!r} in {preds_path}")
                continue
            hyps = df["prediction"].astype(str).tolist()
            refs = df[ref_col].astype(str).tolist()
            avg_bleu, avg_chrf = score_pair_sentence_avg(hyps, refs)
            rows.append({"language": lang, "direction": direction, "rows": len(df),
                         "spBLEU": round(avg_bleu, 4), "chrF++": round(avg_chrf, 4)})
            print(f"[ok] {lang}/{direction}: {len(df)} rows "
                  f"avg spBLEU={avg_bleu:.4f} avg chrF++={avg_chrf:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recap_root",
                     default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP")
    ap.add_argument("--out_dir", default=str(HERE))
    args = ap.parse_args()
    indictrans2_root = Path(args.recap_root) / "indictrans2_finetune"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== IndicTrans2 =====")
    rows = score_model(indictrans2_root)
    cols = ["model", "language", "direction", "rows", "spBLEU", "chrF++"]
    for r in rows:
        r["model"] = "IndicTrans2"
    out_path = out_dir / "indictrans2_scores.csv"
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    print(f"-> {out_path} ({len(rows)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
