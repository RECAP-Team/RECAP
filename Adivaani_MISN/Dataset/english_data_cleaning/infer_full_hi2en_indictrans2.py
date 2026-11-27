"""
Full-corpus Hindi -> English translation using base IndicTrans2
(IndicTrans2-indic-en-1B, not fine-tuned) -- regenerates the English column
for all rows in eng_ready_infer.csv. IndicTrans2 was chosen over
Sarvam-Translate after a side-by-side comparison on the 100 worst-English
rows (fewer degenerate/repeated outputs, more natural output length,
correct place names where Sarvam hallucinated one) -- see
eng_test_predictions.csv in this same directory.

Writes directly into eng_ready_infer.csv, in place, inserting the new
'English' column before 'Hindi' (final column order: Unique_ID, English,
Hindi, Bhili).

Resumable: if the file already has a partially-filled 'English' column
(e.g. from a previous run that ran out of walltime), only the still-empty
rows are translated. Progress is checkpointed back to the same CSV every
SAVE_EVERY batches, so an interrupted run can just be re-started with the
exact same command to continue where it left off.

Run (1 GPU, interactive job):
    python infer_full_hi2en_indictrans2.py
"""

import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

HERE = Path(__file__).resolve().parent
CSV_PATH = str(HERE / "eng_ready_infer.csv")

REPO_ROOT = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP"
INDICTRANS2_BASE_MODEL = f"{REPO_ROOT}/Models/IndicTrans2-indic-en-1B"

HIN_CODE = "hin_Deva"
ENG_CODE = "eng_Latn"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_BEAMS = 4
SAVE_EVERY = 200  # batches (~3200 rows) between checkpoint saves

OUT_COLS = ["Unique_ID", "English", "Hindi", "Bhili"]


def main():
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, na_values=[])
    print(f"[read] {CSV_PATH} ({len(df)} rows)")
    for col in ("Unique_ID", "Hindi", "Bhili"):
        assert col in df.columns, f"expected column '{col}', got {list(df.columns)}"

    if "English" not in df.columns:
        df.insert(1, "English", "")
    else:
        df["English"] = df["English"].fillna("")

    todo_mask = df["English"].str.strip() == ""
    todo_idx = df.index[todo_mask].tolist()
    print(f"[resume] {len(df) - len(todo_idx)} already translated, {len(todo_idx)} remaining")

    if not todo_idx:
        print("[done] nothing left to translate")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} (cuda available: {torch.cuda.is_available()})")

    print(f"[load] {INDICTRANS2_BASE_MODEL}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS2_BASE_MODEL, trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_BASE_MODEL, trust_remote_code=True)
    model.eval()
    processor = IndicProcessor(inference=True)

    hindi_all = df["Hindi"].astype(str).str.strip().tolist()

    start_time = time.time()
    n_done_this_run = 0
    with torch.no_grad():
        for b in range(0, len(todo_idx), BATCH_SIZE):
            batch_idx = todo_idx[b:b + BATCH_SIZE]
            chunk = [hindi_all[i] for i in batch_idx]

            batch = processor.preprocess_batch(chunk, src_lang=HIN_CODE, tgt_lang=ENG_CODE)
            inputs = tokenizer(
                batch, truncation=True, padding="longest",
                return_tensors="pt", max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds = processor.postprocess_batch(decoded, lang=ENG_CODE)

            for i, pred in zip(batch_idx, preds):
                df.at[i, "English"] = pred

            n_done_this_run += len(batch_idx)
            elapsed = time.time() - start_time
            rate = n_done_this_run / elapsed if elapsed > 0 else 0
            remaining = len(todo_idx) - n_done_this_run
            eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
            print(f"[progress] {n_done_this_run}/{len(todo_idx)} this run "
                  f"({rate:.1f} rows/sec, ETA {eta_min:.1f} min)")

            batch_num = b // BATCH_SIZE
            if (batch_num + 1) % SAVE_EVERY == 0:
                df[OUT_COLS].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                print(f"[checkpoint] saved progress -> {CSV_PATH}")

    df[OUT_COLS].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[done] {len(df)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
