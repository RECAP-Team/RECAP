"""
Shared helpers for the haq_eval/ inference scripts (run_mt5.py, run_sarvam.py).

Inference-only: scores the ALREADY-FINETUNED corrected Bhili hi2tgt
checkpoints (mt5-bhili-hi2tgt-finetuned, sarvam-bhili-hi2tgt-finetuned --
no retraining here) against the HAQ Bhili test set
(Bhili_test_data_haq.csv, converted from the original xlsx -- columns
Language/ISO Code/Serial no./Hindi/Bhili/Model Output). Hindi -> Bhili
only, since that's the only direction this test file has. num_beams=4.

Output layout (under config.json's "scores_root"):
    <model>/row_scores.csv   -- per-row BLEU/spBLEU/chrF++
    summary.csv               -- one row per model, averaged
"""

import csv
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU

HERE = Path(__file__).resolve().parent


def load_config(path=None):
    with open(path or HERE / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_pairs(cfg):
    """Returns (ids, src_texts, ref_texts) for Hindi -> Bhili, read from
    the HAQ test CSV."""
    df = pd.read_csv(cfg["test_csv"], encoding="utf-8-sig")
    id_col, src_col, ref_col = cfg["id_col"], cfg["src_col"], cfg["ref_col"]
    df = df[[id_col, src_col, ref_col]].dropna()
    for c in (src_col, ref_col):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[src_col] != "") & (df[ref_col] != "")].reset_index(drop=True)
    return df[id_col].tolist(), df[src_col].tolist(), df[ref_col].tolist()


def score_and_save(model_name, ids, src_texts, ref_texts, preds, cfg):
    """Sentence-level BLEU/spBLEU/chrF++ per row, saved to
    scores_root/<model>/row_scores.csv, plus the mean of each metric over
    all rows appended as one line to scores_root/summary.csv."""
    spbleu_scorer = BLEU(tokenize="flores200")

    rows = []
    for uid, src, ref, pred in zip(ids, src_texts, ref_texts, preds):
        bleu = sacrebleu.sentence_bleu(pred, [ref]).score
        spbleu = spbleu_scorer.sentence_score(pred, [ref]).score
        chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
        rows.append({
            "id": uid, "source": src, "reference": ref, "prediction": pred,
            "bleu": round(bleu, 4), "spbleu": round(spbleu, 4), "chrf++": round(chrf, 4),
        })

    out_dir = Path(cfg["scores_root"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame(rows)
    row_scores_path = out_dir / "row_scores.csv"
    row_df.to_csv(row_scores_path, index=False, encoding="utf-8-sig")
    print(f"[{model_name}] row scores -> {row_scores_path}")

    avg_bleu = row_df["bleu"].mean()
    avg_spbleu = row_df["spbleu"].mean()
    avg_chrf = row_df["chrf++"].mean()

    summary_csv = Path(cfg["scores_root"]) / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "direction", "rows", "avg_bleu", "avg_spbleu", "avg_chrf++"]
    row = [model_name, "hi2tgt", len(row_df), f"{avg_bleu:.4f}", f"{avg_spbleu:.4f}", f"{avg_chrf:.4f}"]
    is_new = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[{model_name}] avg BLEU={avg_bleu:.4f} spBLEU={avg_spbleu:.4f} "
          f"chrF++={avg_chrf:.4f} -> {summary_csv}")
