"""
Zero-shot Hindi <-> Marathi inference with the BASE mt5 model (no finetuning),
using the exact inference recipe from mt5_finetune/*/infer.py (the
tribal-language mt5 inference):

    - T5 instruction prefix  "translate {src_name} to {tgt_name}: "
    - MAX_LENGTH = 128   (tokenizer truncation AND generate() max_length)
    - BATCH_SIZE = 32
    - NUM_BEAMS  = 4
    - tokenizer.batch_decode(..., skip_special_tokens=True)
    - corpus BLEU  : sacrebleu.corpus_bleu(preds, [refs]).score
    - corpus chrF++: sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

Runs BOTH directions in one pass over a single CSV:
    hi2mr : hindi   -> marathi    prefix "translate Hindi to Marathi: "
    mr2hi : marathi -> hindi      prefix "translate Marathi to Hindi: "

The CSV must contain a 'hindi' and a 'marathi' column (see the screenshot of
IIT-B-Hindi-Marathi-datasets/Marathi/test.csv). Override the names with
--hi_col / --mr_col if needed.

Outputs (written next to this script):
    marathi_infer_predictions_hi2mr.csv   source, prediction, reference
    marathi_infer_predictions_mr2hi.csv
    marathi_infer_scores.csv              job, rows, bleu, chrf++

Usage:
    python marathi_infer.py --csv /path/to/test.csv
    python marathi_infer.py --csv /path/to/test.csv --model /path/to/mt5-base
"""

import os
import csv
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

# --- Inference hyperparameters: identical to mt5_finetune/*/infer.py ---
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_BEAMS  = 4

HERE = Path(__file__).resolve().parent
SCORES_CSV = HERE / "marathi_infer_scores.csv"

# (job_name, src_col, tgt_col, src_name, tgt_name)
DIRECTIONS = [
    ("hi2mr", "hi_col", "mr_col", "Hindi", "Marathi"),
    ("mr2hi", "mr_col", "hi_col", "Marathi", "Hindi"),
]


def append_scores(job_name, n_rows, bleu, chrfpp):
    header = ["job", "rows", "bleu", "chrf++"]
    row = [job_name, n_rows, f"{bleu:.4f}", f"{chrfpp:.4f}"]
    is_new = not SCORES_CSV.exists()
    with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended '{job_name}' to {SCORES_CSV}")


def run_direction(job_name, src_col, tgt_col, src_name, tgt_name, df, tokenizer, model, device):
    prefix = f"translate {src_name} to {tgt_name}: "
    print(f"\n========== {job_name} ==========")
    print(f"  {src_col} -> {tgt_col}  |  prefix: '{prefix}'")

    for col in (src_col, tgt_col):
        assert col in df.columns, f"'{col}' not found in CSV, got {list(df.columns)}"

    sub = df[[src_col, tgt_col]].copy()
    sub[src_col] = sub[src_col].astype(str).str.strip()
    sub[tgt_col] = sub[tgt_col].astype(str).str.strip()
    sub = sub[sub[src_col] != ""].reset_index(drop=True)

    src_texts = sub[src_col].tolist()
    ref_texts = sub[tgt_col].tolist()
    total_rows = len(src_texts)
    print(f"[Data] rows to translate: {total_rows}")

    preds_list = []
    with torch.no_grad():
        for i in range(0, total_rows, BATCH_SIZE):
            raw_chunk = src_texts[i:i + BATCH_SIZE]
            chunk = [prefix + str(t) for t in raw_chunk]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **enc,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
            )
            preds_list.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  decoded {min(i + BATCH_SIZE, total_rows)}/{total_rows}")

    preds_path = HERE / f"marathi_infer_predictions_{job_name}.csv"
    pd.DataFrame(
        {"source": src_texts, "prediction": preds_list, "reference": ref_texts}
    ).to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"[Save] Predictions -> {preds_path}")

    bleu   = sacrebleu.corpus_bleu(preds_list, [ref_texts]).score
    chrfpp = sacrebleu.corpus_chrf(preds_list, [ref_texts], word_order=2).score
    print(f"BLEU   : {bleu:.4f}")
    print(f"chrF++ : {chrfpp:.4f}")
    append_scores(job_name, total_rows, bleu, chrfpp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the Hindi-Marathi test CSV")
    parser.add_argument("--model", default="google/mt5-base",
                        help="Base mt5 model id or local path (default: google/mt5-base)")
    parser.add_argument("--hi_col", default="hindi", help="Hindi column name (default: hindi)")
    parser.add_argument("--mr_col", default="marathi", help="Marathi column name (default: marathi)")
    args = parser.parse_args()

    print("=" * 60)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("model:", args.model)
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.csv)

    print("[Model] Loading base mt5 tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    col_map = {"hi_col": args.hi_col, "mr_col": args.mr_col}
    for job_name, src_key, tgt_key, src_name, tgt_name in DIRECTIONS:
        run_direction(
            job_name, col_map[src_key], col_map[tgt_key], src_name, tgt_name,
            df, tokenizer, model, device,
        )

    print(f"\nAll directions done. Scores: {SCORES_CSV}")


if __name__ == "__main__":
    main()
