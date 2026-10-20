"""
Shared helpers for the reference_eval_210_marathi/ inference scripts
(run_mt5.py, run_nllb.py, run_sarvam.py, run_indictrans2.py).

Same 210-row reference set as ../reference_eval_210/ (it has a Marathi
column too), but scored against each model's ORIGINAL Marathi-finetuned
checkpoint (mt5_finetune/Marathi/, NLLB-finetune/Marathi/,
sarvam_finetune/Marathi/, indictrans2_finetune/Marathi/) instead of the
corrected-data tribal-language checkpoints -- inference-only, no
retraining. Direction naming differs per model (NLLB/mT5 were trained as
hi2mr/mr2hi; Sarvam/IndicTrans2 as hi2tgt/tgt2hi) but both map to the same
Hindi<->Marathi column pair.

Output, per model, saved INSIDE that model's own existing Marathi
directory (not centrally) -- <output_root>/reference_eval_210/:
    row_scores_<direction>.csv   -- per-row BLEU/spBLEU/chrF++
    summary.csv                  -- one row per direction, averaged
"""

import csv
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU

HERE = Path(__file__).resolve().parent
HIN_COL = "Hindi"
MAR_COL = "Marathi"


def load_config(path=None):
    with open(path or HERE / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_marathi_pairs(test_csv, hi_to_mar):
    """Returns (ids, src_texts, ref_texts) for Hindi<->Marathi, read from
    the single unified reference CSV."""
    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    df = df[["unique_id", HIN_COL, MAR_COL]].dropna()
    for c in (HIN_COL, MAR_COL):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[HIN_COL] != "") & (df[MAR_COL] != "")].reset_index(drop=True)

    src_col, ref_col = (HIN_COL, MAR_COL) if hi_to_mar else (MAR_COL, HIN_COL)
    return df["unique_id"].tolist(), df[src_col].tolist(), df[ref_col].tolist()


def score_and_save(output_root, direction, ids, src_texts, ref_texts, preds):
    """Sentence-level BLEU/spBLEU/chrF++ per row, saved under
    <output_root>/reference_eval_210/, plus the mean of each metric over
    all rows appended as one line to that same dir's summary.csv."""
    spbleu_scorer = BLEU(tokenize="flores200")

    rows = []
    for uid, src, ref, pred in zip(ids, src_texts, ref_texts, preds):
        bleu = sacrebleu.sentence_bleu(pred, [ref]).score
        spbleu = spbleu_scorer.sentence_score(pred, [ref]).score
        chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
        rows.append({
            "unique_id": uid, "source": src, "reference": ref, "prediction": pred,
            "bleu": round(bleu, 4), "spbleu": round(spbleu, 4), "chrf++": round(chrf, 4),
        })

    out_dir = Path(output_root) / "reference_eval_210"
    out_dir.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame(rows)
    row_scores_path = out_dir / f"row_scores_{direction}.csv"
    row_df.to_csv(row_scores_path, index=False, encoding="utf-8-sig")
    print(f"[{direction}] row scores -> {row_scores_path}")

    avg_bleu = row_df["bleu"].mean()
    avg_spbleu = row_df["spbleu"].mean()
    avg_chrf = row_df["chrf++"].mean()

    summary_csv = out_dir / "summary.csv"
    header = ["direction", "rows", "avg_bleu", "avg_spbleu", "avg_chrf++"]
    row = [direction, len(row_df), f"{avg_bleu:.4f}", f"{avg_spbleu:.4f}", f"{avg_chrf:.4f}"]
    is_new = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[{direction}] avg BLEU={avg_bleu:.4f} spBLEU={avg_spbleu:.4f} "
          f"chrF++={avg_chrf:.4f} -> {summary_csv}")
