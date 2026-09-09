"""
Fill the empty Bhili / Mundari / Gondi columns of the multi-parallel test CSV
by running each language's finetuned mT5 hi2tgt checkpoint, one language per
GPU, in parallel (3 GPUs -> 3 languages at once).

Input CSV columns (see Test_data_inference/'test - test.csv'):
    hindi, marathi, Bhili, Mundari, Gondi
'hindi' is the source; 'marathi' is already filled and passed through
untouched; Bhili / Mundari / Gondi are generated here.

What it does:
  1. Adds a 'unique_id' column (test_00001, ... ) if the CSV doesn't have one.
  2. For each of Bhili / Mundari / Gondi, spawns one worker pinned to one GPU
     that loads mt5_finetune/<Lang>/mt5-<lang>-hi2tgt-finetuned and translates
     the Hindi column with the exact recipe from mt5_finetune/*/infer.py:
        prefix "translate Hindi to <Lang>: ", MAX_LENGTH=128, BATCH_SIZE=32,
        NUM_BEAMS=4, batch_decode(skip_special_tokens=True).
  3. Merges the three per-language predictions back on unique_id and writes a
     6-column CSV: unique_id, hindi, marathi, Bhili, Mundari, Gondi.

Outputs (next to this script unless overridden):
    test_with_ids.csv          input + unique_id, targets still blank
    test_with_predictions.csv  final 6-column result
    _infer_work/               worker input + per-language partial CSVs

Run:
    python infer_tribal_mt5_parallel.py
    python infer_tribal_mt5_parallel.py --input_csv "/path/test - test.csv"
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
DEFAULT_INPUT = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/Test_data_inference/test - test.csv"
DEFAULT_REPO_ROOT = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP"

MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_BEAMS  = 2

# CSV column name == prefix name == repo folder name; checkpoint slug is the
# lowercase form: mt5_finetune/<Lang>/mt5-<lang>-hi2tgt-finetuned
LANGS = ["Bhili", "Mundari", "Gondi"]


def run_worker(gpu_id, lang, ckpt_path, worker_input_csv, partial_out):
    """One language on one GPU. Writes <partial_out> = unique_id,<lang>."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    tag = f"[{lang} gpu{gpu_id if use_cuda else '-'}]"

    if Path(partial_out).exists():
        prev = pd.read_csv(partial_out)
        if len(prev) and prev[lang].notna().all():
            print(f"{tag} partial already complete ({len(prev)} rows) -- skipping", flush=True)
            return

    df = pd.read_csv(worker_input_csv)
    ids = df["unique_id"].astype(str).tolist()
    src = df["hindi"].fillna("").astype(str).str.strip().tolist()
    n = len(src)
    print(f"{tag} loading {ckpt_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device).eval()
    prefix = f"translate Hindi to {lang}: "

    preds = [""] * n
    todo = [i for i in range(n) if src[i]]        # skip blank Hindi rows
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
            if (b // BATCH_SIZE) % 20 == 0:
                print(f"{tag} {min(b + BATCH_SIZE, len(todo))}/{len(todo)}", flush=True)

    pd.DataFrame({"unique_id": ids, lang: preds}).to_csv(
        partial_out, index=False, encoding="utf-8-sig")
    print(f"{tag} done -> {partial_out}", flush=True)


def resolve_col(df, name, default=None):
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower(), default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT)
    ap.add_argument("--output_csv", default=str(HERE / "test_with_predictions.csv"))
    ap.add_argument("--repo_root", default=DEFAULT_REPO_ROOT,
                    help="root that holds mt5_finetune/<Lang>/mt5-<lang>-hi2tgt-finetuned")
    ap.add_argument("--src_col", default="hindi")
    ap.add_argument("--id_prefix", default="test_")
    ap.add_argument("--id_pad", type=int, default=5)
    ap.add_argument("--work_dir", default=str(HERE / "_infer_work"))
    ap.add_argument("--bhili_ckpt", default=None)
    ap.add_argument("--mundari_ckpt", default=None)
    ap.add_argument("--gondi_ckpt", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"[read] {args.input_csv}  ({len(df)} rows, cols={list(df.columns)})")

    src = resolve_col(df, args.src_col)
    if src is None:
        sys.exit(f"[FATAL] source column {args.src_col!r} not found in {list(df.columns)}")
    mar = resolve_col(df, "marathi")

    ren = {src: "hindi"}
    if mar:
        ren[mar] = "marathi"
    for lang in LANGS:
        c = resolve_col(df, lang)
        if c:
            ren[c] = lang
    df = df.rename(columns=ren)
    if "marathi" not in df.columns:
        df["marathi"] = ""
    for lang in LANGS:
        if lang not in df.columns:
            df[lang] = ""

    uid = resolve_col(df, "unique_id")
    if uid:
        df = df.rename(columns={uid: "unique_id"})
        print(f"[id] using existing unique_id column")
    else:
        df.insert(0, "unique_id",
                  [f"{args.id_prefix}{i:0{args.id_pad}d}" for i in range(1, len(df) + 1)])
        print(f"[id] added unique_id ({args.id_prefix}{1:0{args.id_pad}d} ... )")

    out_cols = ["unique_id", "hindi", "marathi", "Bhili", "Mundari", "Gondi"]
    ids_out = Path(args.output_csv).with_name("test_with_ids.csv")
    df[out_cols].to_csv(ids_out, index=False, encoding="utf-8-sig")
    print(f"[write] {ids_out}")

    wd = Path(args.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    win = wd / "_worker_input.csv"
    df[["unique_id", "hindi"]].to_csv(win, index=False, encoding="utf-8-sig")

    def ckpt_for(lang):
        override = getattr(args, f"{lang.lower()}_ckpt")
        if override:
            return override
        return str(Path(args.repo_root) / "mt5_finetune" / lang /
                   f"mt5-{lang.lower()}-hi2tgt-finetuned")

    jobs = []
    for lang in LANGS:
        cp = ckpt_for(lang)
        if not Path(cp).exists():
            sys.exit(f"[FATAL] {lang} checkpoint not found: {cp}\n"
                     f"        fix --repo_root or pass --{lang.lower()}_ckpt /abs/path")
        jobs.append((lang, cp))

    import torch
    ngpu = torch.cuda.device_count()
    print(f"[gpu] {ngpu} visible")
    if ngpu >= len(jobs):
        gpu_ids = list(range(len(jobs)))
    elif ngpu >= 1:
        gpu_ids = [i % ngpu for i in range(len(jobs))]
        print(f"[gpu] WARNING: fewer than {len(jobs)} GPUs -- languages will share GPUs {gpu_ids}")
    else:
        gpu_ids = [0] * len(jobs)
        print("[gpu] WARNING: no CUDA -- running on CPU (slow)")

    ctx = mp.get_context("spawn")
    procs, partials = [], {}
    for (lang, cp), gid in zip(jobs, gpu_ids):
        po = str(wd / f"_partial_{lang.lower()}.csv")
        partials[lang] = po
        p = ctx.Process(target=run_worker, args=(gid, lang, cp, str(win), po))
        p.start()
        procs.append((lang, p))
        print(f"[spawn] {lang} -> cuda:{gid}  ({cp})")

    failed = []
    for lang, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(lang)
    if failed:
        sys.exit(f"[FATAL] workers failed: {failed}. Final CSV not written -- see logs above.")

    for lang in LANGS:
        pdf = pd.read_csv(partials[lang])
        m = dict(zip(pdf["unique_id"].astype(str), pdf[lang].astype(str)))
        df[lang] = df["unique_id"].astype(str).map(m).fillna("")

    df[out_cols].to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    filled = {lang: int((df[lang].str.len() > 0).sum()) for lang in LANGS}
    print(f"[done] {len(df)} rows -> {args.output_csv}")
    print(f"       non-empty predictions: {filled}")
    print(f"       ids-only copy        : {ids_out}")


if __name__ == "__main__":
    main()
