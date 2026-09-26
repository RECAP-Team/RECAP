"""
Fill the empty Bhili/Mundari translation columns across the 6 xnli/*.csv
files using finetuned mt5 hi2tgt checkpoints (beam=2, batch=64). Gondi
columns are never touched (out of scope -- no Gondi checkpoint used here).

One job = one (csv_file, language) pair, e.g. "test.csv" + "Bhili". A job
loads that language's mt5 hi2tgt checkpoint ONCE and translates every Hindi
source column in that file into its matching Bhili/Mundari target column --
only the currently-empty cells; already-filled cells are left untouched.
6 files x 2 languages (Bhili, Mundari) = 12 jobs.

Jobs run one-per-GPU, pulled off a shared queue -- same GPU-pool pattern as
indictrans2_finetune/finetune_indictrans2.py: pool_size = min(n_gpus,
n_jobs), and CUDA_VISIBLE_DEVICES is set per-worker in _pool_init(), before
torch is ever imported in that worker process (torch.cuda.set_device() alone
does not isolate a worker to one physical GPU -- confirmed the hard way on
this cluster). GPU count is auto-detected: with 4 GPUs, up to 4 jobs run at
once; with 1 GPU, pool_size=1 and jobs run one at a time automatically (a
single-process Pool serializes pool.map() -- no separate sequential branch
needed).

Each job writes its own partial CSV (row_idx + prediction) per target
column, so progress survives a crash/timeout. After every job finishes, one
merge pass (in the parent process, strictly after the pool has joined --
no concurrent writers, no lock needed) overlays all partial files onto a
fresh copy of each original CSV and writes the finished file to
<output_dir>/<csv_file>.

Run (on Pragya, from this directory):
    python xnli_infer.py
    python xnli_infer.py --dry_run   # sanity-check column/row detection only,
                                      # no model load, no GPU needed -- useful
                                      # to verify against local copies first
"""

import os
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd

HERE = Path(__file__).resolve().parent
RECAP_ROOT = HERE.parent

# ---- Which Hindi source column(s) feed which Bhili/Mundari target
#      column(s), per file (hardcoded from the actual headers -- these files
#      don't share a naming convention consistent enough to detect safely
#      with a generic regex, e.g. "hindi" vs "Hindi" case differs per file). ----
CSV_CONFIG = {
    "datasets_bbc_hindi_articles_labeled - datasets_bbc_hindi_articles_labeled.csv": [
        {"src_col": "Headline_Hindi", "Bhili": "Headline_Bhili", "Mundari": "Headline_Mundari"},
        {"src_col": "Content_Hindi",  "Bhili": "Content_Bhili",  "Mundari": "Content_Mundari"},
    ],
    "hindi_movie_polarity - hindi_movie_polarity.csv": [
        {"src_col": "Hindi", "Bhili": "Bhili", "Mundari": "Mundari"},
    ],
    "Hindi_quora.csv": [
        {"src_col": "question1_hindi", "Bhili": "question1_bhili", "Mundari": "question1_mundari"},
        {"src_col": "question2_hindi", "Bhili": "question2_bhili", "Mundari": "question2_mundari"},
    ],
    "hindi_sentiment_preprocessed_twitter_sentiment - hindi_sentiment_preprocessed_twitter_sentiment.csv": [
        {"src_col": "Hindi", "Bhili": "Bhili", "Mundari": "Mundari"},
    ],
    "test.csv": [
        {"src_col": "Hindi", "Bhili": "Bhili", "Mundari": "Mundari"},
    ],
    "xnli_hindi_test_corrected.csv": [
        {"src_col": "Premise_Hindi",    "Bhili": "Premise_Bhili",    "Mundari": "Premise_Mundari"},
        {"src_col": "Hypothesis_Hindi", "Bhili": "Hypothesis_Bhili", "Mundari": "Hypothesis_Mundari"},
    ],
}

# mt5 hi2tgt checkpoints trained earlier in this project -- see
# mt5_finetune/Bhili/infer_config.json / mt5_finetune/Mundari/infer_config.json
# for the same paths used there.
CHECKPOINTS = {
    "Bhili":   str(RECAP_ROOT / "mt5_finetune" / "Bhili" / "mt5-bhili-hi2tgt-finetuned"),
    "Mundari": str(RECAP_ROOT / "mt5_finetune" / "Mundari" / "mt5-mundari-hi2tgt-finetuned"),
}

DEFAULT_INPUT_DIR = "/home/scai/msr/aiy257590/flash/xnli"
DEFAULT_OUTPUT_DIR = "/home/scai/msr/aiy257590/flash/xnli/inferenced_files"

MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_BEAMS = 2
FLUSH_EVERY_ROWS = 500  # flush partial predictions to disk this often, for resumability


def _is_empty(val):
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s.lower() == "nan"


def _flush_partial(partial_path, row_idx_list, pred_list):
    out = pd.DataFrame({"row_idx": row_idx_list, "prediction": pred_list})
    file_exists = os.path.exists(partial_path)
    out.to_csv(partial_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def run_job(csv_file, language, args):
    job_name = f"{Path(csv_file).stem}__{language}"
    checkpoint_path = CHECKPOINTS[language]
    col_groups = CSV_CONFIG[csv_file]

    print(f"\n========== {job_name} (gpu {_worker_gpu_id}) ==========")
    print(f"  checkpoint: {checkpoint_path}")

    df = pd.read_csv(Path(args.input_dir) / csv_file)
    for group in col_groups:
        assert group["src_col"] in df.columns, f"'{group['src_col']}' missing in {csv_file}"
        assert group[language] in df.columns, f"'{group[language]}' missing in {csv_file}"

    partial_dir = Path(args.output_dir) / "_partial"
    if not args.dry_run:
        partial_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"translate Hindi to {language}: "
    tokenizer = model = device = None  # lazy-load only if there's real work to do

    for group in col_groups:
        src_col = group["src_col"]
        tgt_col = group[language]
        partial_path = partial_dir / f"{Path(csv_file).stem}__{tgt_col}.csv"

        done_idx = set()
        if partial_path.exists():
            done_idx = set(pd.read_csv(partial_path)["row_idx"].tolist())
            print(f"[Resume] {tgt_col}: {len(done_idx)} rows already done this run")

        already_filled = sum(1 for i in df.index if not _is_empty(df.at[i, tgt_col]))
        pending = [
            i for i in df.index
            if i not in done_idx
            and _is_empty(df.at[i, tgt_col])
            and not _is_empty(df.at[i, src_col])
        ]
        print(f"[{tgt_col}] {len(df)} rows total | {already_filled} already filled | "
              f"{len(done_idx)} done this run | {len(pending)} pending")

        if args.dry_run or not pending:
            continue

        if model is None:
            assert os.path.isdir(checkpoint_path), \
                f"checkpoint not found: {checkpoint_path}"
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("[Model] Loading tokenizer & model from checkpoint ...")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
            # CUDA_VISIBLE_DEVICES already restricts this process to one
            # physical GPU (set in _pool_init(), before torch was imported
            # anywhere in this process) -- so "cuda:0" is always that GPU.
            device = torch.device("cuda:0")
            model.to(device)
            model.eval()

        buf_idx, buf_pred = [], []
        import torch
        with torch.no_grad():
            for i in range(0, len(pending), BATCH_SIZE):
                chunk_idx = pending[i:i + BATCH_SIZE]
                chunk_src = [prefix + str(df.at[j, src_col]) for j in chunk_idx]
                enc = tokenizer(
                    chunk_src, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LENGTH,
                ).to(device)
                out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)
                preds = tokenizer.batch_decode(out, skip_special_tokens=True)

                buf_idx.extend(chunk_idx)
                buf_pred.extend(preds)

                if (i // BATCH_SIZE) % 10 == 0:
                    print(f"  [{tgt_col}] decoded {i + len(chunk_idx)}/{len(pending)}")

                if len(buf_idx) >= FLUSH_EVERY_ROWS:
                    _flush_partial(partial_path, buf_idx, buf_pred)
                    buf_idx, buf_pred = [], []

        if buf_idx:
            _flush_partial(partial_path, buf_idx, buf_pred)

        print(f"[Save] {tgt_col}: partial results -> {partial_path}")

    print(f"[Done] {job_name}")


# =============================================================
# GPU-pool orchestration -- same pattern as
# indictrans2_finetune/finetune_indictrans2.py: a fixed pool of
# min(n_gpus, n_jobs) worker processes, each pinned to one GPU for its whole
# lifetime, pulling jobs off a queue. Works whether there are fewer, equal,
# or more jobs than GPUs -- including exactly 1 GPU, where pool_size=1 makes
# pool.map() process jobs one at a time automatically.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    # Must happen here, before torch is ever imported in this process
    # (run_job() imports it lazily) -- torch.cuda.set_device() alone only
    # changes the default device, it doesn't stop transformers from seeing
    # every GPU and spreading one job's compute across all of them.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    csv_file, language, args = job_args
    run_job(csv_file, language, args)


def merge_outputs(args):
    """Overlay every job's partial predictions onto a fresh copy of the
    original CSV and write the finished file to output_dir/<csv_file>. Runs
    once in the parent process, strictly after the worker pool has joined --
    no concurrent writers, so no lock is needed here."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "_partial"

    for csv_file, col_groups in CSV_CONFIG.items():
        df = pd.read_csv(Path(args.input_dir) / csv_file)
        filled = 0
        for group in col_groups:
            for language in ("Bhili", "Mundari"):
                tgt_col = group[language]
                partial_path = partial_dir / f"{Path(csv_file).stem}__{tgt_col}.csv"
                if not partial_path.exists():
                    continue
                part = pd.read_csv(partial_path)
                for _, r in part.iterrows():
                    df.at[int(r["row_idx"]), tgt_col] = r["prediction"]
                    filled += 1
        out_path = out_dir / csv_file
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Merge] {csv_file}: {filled} cells filled -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--dry_run", action="store_true",
                     help="Only report pending-row counts per column, no "
                          "model load / GPU needed -- run this first to "
                          "sanity-check column detection against local CSVs.")
    args = ap.parse_args()

    for csv_file in CSV_CONFIG:
        assert (Path(args.input_dir) / csv_file).exists(), \
            f"expected input CSV not found: {Path(args.input_dir) / csv_file}"

    jobs = [(csv_file, language) for csv_file in CSV_CONFIG for language in ("Bhili", "Mundari")]
    print(f"[jobs] {len(jobs)} total (6 files x 2 languages)")

    if args.dry_run:
        print("[dry_run] Skipping GPU pool -- running all jobs sequentially, no model load.")
        for csv_file, language in jobs:
            run_job(csv_file, language, args)
        return

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[gpu] {n_gpus} visible")
    if n_gpus == 0:
        raise SystemExit("No GPU visible -- request GPUs on the job scheduler before running this.")

    pool_size = min(n_gpus, len(jobs))
    print(f"[pool] {pool_size} worker(s) -- {'parallel' if pool_size > 1 else 'sequential'}")

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu_id in range(pool_size):
        gpu_queue.put(gpu_id)

    job_args = [(csv_file, language, args) for csv_file, language in jobs]
    with ctx.Pool(processes=pool_size, initializer=_pool_init, initargs=(gpu_queue,)) as pool:
        pool.map(_pool_run, job_args)

    print("\n[Merge] All jobs done, merging partial outputs into final CSVs ...")
    merge_outputs(args)
    print(f"\nAll done. Final CSVs in {args.output_dir}")


if __name__ == "__main__":
    main()
