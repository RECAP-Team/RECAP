"""
Shared helpers for the reference_eval_210/ inference scripts (run_mt5.py,
run_qwen.py, run_sarvam.py, run_nllb.py, run_indictrans2.py).

These scripts run inference-ONLY against the five models' ALREADY-FINETUNED
checkpoints under correct_checkpoints/<model>/<language>/ -- no retraining
happens here. They score the shared 210-row reference set at
config.json's "test_csv" (Hindi/Bhili/Mundari/Gondi/Marathi/English columns,
one row per sentence), hi2tgt + tgt2hi only, num_beams=4.

Output layout (under config.json's "scores_root"):
    scores/<language>/<model>/<direction>/row_scores.csv   -- per-row BLEU/spBLEU/chrF++
    scores/summary_scores.csv                               -- one row per (language, model, direction), averaged
"""

import csv
import json
from pathlib import Path

import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU

HERE = Path(__file__).resolve().parent
HIN_COL = "Hindi"


def load_config(path=None):
    with open(path or HERE / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_test_pairs(cfg, lang, direction):
    """Returns (ids, src_texts, ref_texts) for one (lang, direction), read
    straight from the single unified reference CSV (no train/val split --
    this is eval-only data)."""
    df = pd.read_csv(cfg["test_csv"], encoding="utf-8-sig")
    tgt_col = cfg["languages"][lang]["column"]
    df = df[["unique_id", HIN_COL, tgt_col]].dropna()
    for c in (HIN_COL, tgt_col):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[HIN_COL] != "") & (df[tgt_col] != "")].reset_index(drop=True)

    if direction == "hi2tgt":
        src_col, ref_col = HIN_COL, tgt_col
    elif direction == "tgt2hi":
        src_col, ref_col = tgt_col, HIN_COL
    else:
        raise ValueError(f"reference_eval_210 only supports hi2tgt/tgt2hi, got {direction!r}")

    return df["unique_id"].tolist(), df[src_col].tolist(), df[ref_col].tolist()


def score_and_save(model_name, lang, direction, ids, src_texts, ref_texts, preds, cfg):
    """Sentence-level BLEU/spBLEU/chrF++ per row, saved to
    scores/<lang>/<model>/<direction>/row_scores.csv, plus the mean of each
    metric over all rows appended as one line to scores/summary_scores.csv."""
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

    out_dir = Path(cfg["scores_root"]) / lang / model_name / direction
    out_dir.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame(rows)
    row_scores_path = out_dir / "row_scores.csv"
    row_df.to_csv(row_scores_path, index=False, encoding="utf-8-sig")
    print(f"[{model_name}/{lang}/{direction}] row scores -> {row_scores_path}")

    avg_bleu = row_df["bleu"].mean()
    avg_spbleu = row_df["spbleu"].mean()
    avg_chrf = row_df["chrf++"].mean()

    summary_csv = Path(cfg["scores_root"]) / "summary_scores.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["language", "model", "direction", "rows", "avg_bleu", "avg_spbleu", "avg_chrf++"]
    row = [lang, model_name, direction, len(row_df),
           f"{avg_bleu:.4f}", f"{avg_spbleu:.4f}", f"{avg_chrf:.4f}"]
    is_new = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[{model_name}/{lang}/{direction}] avg BLEU={avg_bleu:.4f} "
          f"spBLEU={avg_spbleu:.4f} chrF++={avg_chrf:.4f} -> {summary_csv}")
