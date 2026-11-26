"""
Run the already fine-tuned Adivaani_MISN mt5-bhili checkpoints (hi2tgt and
en2tgt) on a NEW test file that has no gold Bhili column -- i.e. pure
inference, not scoring. Input is English + Hindi sentences only; this
generates two independent Bhili translations per row (one from Hindi, one
from English), each using that direction's own fine-tuned checkpoint.

Same generation recipe (prefix format, MAX_LENGTH, num_beams) as the final
manual test-set inference inside mt5_finetune.py, so these predictions are
directly comparable to the ones already scored during training.

Input CSV columns expected: English, Hindi
    (see Adivaani_MISN/Dataset/Test_Data_for_Bhili_English_Hindi.csv)

Run (auto-detects GPU count; uses up to 2, one per direction):
    python infer_new_testset.py
    python infer_new_testset.py --input_csv /path/to/other_test.csv
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = str(
    HERE.parent / "Dataset" / "Test_Data_for_Bhili_English_Hindi.csv"
)

MAX_LENGTH = 128
NUM_BEAMS = 5
BATCH_SIZE = 32

# direction -> (source column in input CSV, source language name for the prefix,
#               checkpoint dir name under Adivaani_MISN/mt5/)
DIRECTIONS = {
    "hi2tgt": {"src_col": "Hindi", "src_name": "Hindi", "ckpt": "mt5-bhili-hi2tgt-finetuned"},
    "en2tgt": {"src_col": "English", "src_name": "English", "ckpt": "mt5-bhili-en2tgt-finetuned"},
}


def run_worker(gpu_id, direction, ckpt_path, worker_input_csv, partial_out):
    """One direction on one GPU. Writes <partial_out> = row_id,<direction>."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    tag = f"[{direction} gpu{gpu_id if use_cuda else '-'}]"

    cfg = DIRECTIONS[direction]
    src_name = cfg["src_name"]
    prefix = f"translate {src_name} to Bhili: "

    df = pd.read_csv(worker_input_csv)
    ids = df["row_id"].astype(str).tolist()
    src = df["src"].fillna("").astype(str).str.strip().tolist()
    n = len(src)

    print(f"{tag} loading {ckpt_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device).eval()

    preds = [""] * n
    todo = [i for i in range(n) if src[i]]
    print(f"{tag} translating {len(todo)}/{n} non-empty rows", flush=True)
    with torch.no_grad():
        for b in range(0, len(todo), BATCH_SIZE):
            idxs = todo[b:b + BATCH_SIZE]
            chunk = [prefix + src[i] for i in idxs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH).to(device)
            out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            for i, d in zip(idxs, dec):
                preds[i] = d
            if (b // BATCH_SIZE) % 10 == 0:
                print(f"{tag} {min(b + BATCH_SIZE, len(todo))}/{len(todo)}", flush=True)

    pd.DataFrame({"row_id": ids, direction: preds}).to_csv(
        partial_out, index=False, encoding="utf-8-sig")
    print(f"{tag} done -> {partial_out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT)
    ap.add_argument("--output_csv", default=str(HERE / "new_testset_predictions.csv"))
    ap.add_argument("--work_dir", default=str(HERE / "_infer_work_newtestset"))
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"[read] {args.input_csv}  ({len(df)} rows, cols={list(df.columns)})")

    for col in ("English", "Hindi"):
        if col not in df.columns:
            sys.exit(f"[FATAL] expected column '{col}' not found in {list(df.columns)}")

    df.insert(0, "row_id", [f"test_{i:05d}" for i in range(1, len(df) + 1)])

    wd = Path(args.work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    jobs = []
    for direction, cfg in DIRECTIONS.items():
        ckpt_path = HERE / cfg["ckpt"]
        if not ckpt_path.exists():
            sys.exit(f"[FATAL] checkpoint not found: {ckpt_path}\n"
                     f"        (did the {direction} training job finish?)")
        win = wd / f"_worker_input_{direction}.csv"
        df[["row_id", cfg["src_col"]]].rename(columns={cfg["src_col"]: "src"}).to_csv(
            win, index=False, encoding="utf-8-sig")
        jobs.append((direction, str(ckpt_path), str(win)))

    import torch
    ngpu = torch.cuda.device_count()
    print(f"[gpu] {ngpu} visible")
    if ngpu >= len(jobs):
        gpu_ids = list(range(len(jobs)))
    elif ngpu >= 1:
        gpu_ids = [i % ngpu for i in range(len(jobs))]
        print(f"[gpu] WARNING: fewer than {len(jobs)} GPUs -- directions will share GPUs {gpu_ids}")
    else:
        gpu_ids = [0] * len(jobs)
        print("[gpu] WARNING: no CUDA -- running on CPU (slow)")

    ctx = mp.get_context("spawn")
    procs, partials = [], {}
    for (direction, ckpt_path, win), gid in zip(jobs, gpu_ids):
        po = str(wd / f"_partial_{direction}.csv")
        partials[direction] = po
        p = ctx.Process(target=run_worker, args=(gid, direction, ckpt_path, win, po))
        p.start()
        procs.append((direction, p))
        print(f"[spawn] {direction} -> cuda:{gid}  ({ckpt_path})")

    failed = []
    for direction, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(direction)
    if failed:
        sys.exit(f"[FATAL] workers failed: {failed}. Final CSV not written -- see logs above.")

    out_cols = ["row_id", "English", "Hindi"]
    for direction in DIRECTIONS:
        pdf = pd.read_csv(partials[direction])
        m = dict(zip(pdf["row_id"].astype(str), pdf[direction].astype(str)))
        col_name = f"Predicted_Bhili_from_{direction}"
        df[col_name] = df["row_id"].astype(str).map(m).fillna("")
        out_cols.append(col_name)

    df[out_cols].to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    filled = {
        f"Predicted_Bhili_from_{d}": int((df[f"Predicted_Bhili_from_{d}"].str.len() > 0).sum())
        for d in DIRECTIONS
    }
    print(f"[done] {len(df)} rows -> {args.output_csv}")
    print(f"       non-empty predictions: {filled}")


if __name__ == "__main__":
    main()
