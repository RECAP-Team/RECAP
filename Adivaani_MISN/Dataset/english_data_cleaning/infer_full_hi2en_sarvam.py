"""
Hindi -> English translation using base Sarvam-Translate (not fine-tuned).
Same resumable/checkpointed in-place pattern as infer_full_hi2en_indictrans2.py,
but using Sarvam-Translate instead of IndicTrans2 -- for retrying the rows that
IndicTrans2 still got wrong (see needs_correction.csv).

Writes directly into the target CSV, in place, inserting/overwriting the
'English' column before 'Hindi' (final column order: Unique_ID, English,
Hindi, Bhili).

Resumable: if the file already has a partially-filled 'English' column
(e.g. from a previous run that ran out of walltime), only the still-empty
rows are translated. Progress is checkpointed back to the same CSV every
SAVE_EVERY_ROWS (5,000) rows.

Needs the 'sarvam' conda env (Sarvam-Translate's gemma3 base isn't
recognized by the 'mt5' env's older transformers version).

Run (1 GPU, interactive job):
    python infer_full_hi2en_sarvam.py --csv_path needs_correction.csv
"""

import os
import time
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent

REPO_ROOT = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP"
SARVAM_BASE_MODEL = f"{REPO_ROOT}/Models/Sarvam-Translate"

MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_BEAMS = 5
SAVE_EVERY_ROWS = 5000  # checkpoint to disk after at least this many new rows

OUT_COLS = ["Unique_ID", "English", "Hindi", "Bhili"]


def build_messages(src_text):
    return [
        {"role": "system", "content": (
            "Translate the given Hindi text to English. Output ONLY the direct "
            "translation itself -- a single line, no notes, no explanations, no "
            "alternate translations, no commentary of any kind."
        )},
        {"role": "user", "content": str(src_text)},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default=str(HERE / "needs_correction.csv"))
    args = ap.parse_args()
    CSV_PATH = args.csv_path

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

    print(f"[load] {SARVAM_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation
    model = AutoModelForCausalLM.from_pretrained(
        SARVAM_BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

    hindi_all = df["Hindi"].astype(str).str.strip().tolist()

    start_time = time.time()
    n_done_this_run = 0
    rows_since_checkpoint = 0
    with torch.no_grad():
        for b in range(0, len(todo_idx), BATCH_SIZE):
            batch_idx = todo_idx[b:b + BATCH_SIZE]
            chunk = [hindi_all[i] for i in batch_idx]

            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(t), tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc, max_new_tokens=MAX_LENGTH, num_beams=NUM_BEAMS,
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            # Sarvam is chat-tuned and sometimes appends "Note: ..." commentary
            # or loops restating the translation after a blank line -- keep
            # only the first paragraph, which is always the actual translation.
            preds = [p.strip().split("\n\n")[0].strip() for p in decoded]

            for i, pred in zip(batch_idx, preds):
                df.at[i, "English"] = pred

            n_done_this_run += len(batch_idx)
            rows_since_checkpoint += len(batch_idx)
            elapsed = time.time() - start_time
            rate = n_done_this_run / elapsed if elapsed > 0 else 0
            remaining = len(todo_idx) - n_done_this_run
            eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
            print(f"[progress] {n_done_this_run}/{len(todo_idx)} this run "
                  f"({rate:.1f} rows/sec, ETA {eta_min:.1f} min)")

            if rows_since_checkpoint >= SAVE_EVERY_ROWS:
                df[OUT_COLS].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                rows_since_checkpoint = 0
                print(f"[checkpoint] saved progress -> {CSV_PATH}")

    df[OUT_COLS].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[done] {len(df)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
