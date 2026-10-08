"""
Variant of generate_human_eval_translations.py (stage 1): samples ONE
shared set of 50 rows PER LANGUAGE (not independently per direction) --
all 4 directions for a given language are built from the SAME 50
underlying test-set rows, so e.g. hi2tgt's source Hindi sentences are
literally the same sentences tgt2hi translates back from (and likewise for
English/en2tgt/tgt2en). Mundari and Gondi each get their own independent
50-row sample -- confirmed their test sets don't share Hindi sentences, so
a single cross-language shared set isn't possible.

Reuses generate_human_eval_translations.py's per-model generation
functions (GEN_FN) and checkpoint-path conventions directly (import, not
duplicated) -- only the sampling strategy differs. Writes its
"<Lang>_<direction>_samples.csv" (source, reference columns) files in
EXACTLY the same shape the original script produces, just derived from the
shared 50-row set instead of independent per-direction sampling -- so
assemble_human_eval_csv.py and compute_human_eval_metrics.py (stages 2/3)
work UNCHANGED against this script's output, just pointed at this
script's --work_dir/--out_dir.

Run (on Pragya, from this directory):
    python generate_human_eval_translations_50.py
    python generate_human_eval_translations_50.py --langs Mundari --n_samples 50
"""

import os
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd

from generate_human_eval_translations import (
    GEN_FN, MODEL_ROOT, MODEL_PREFIX, MODELS, DIR_NAME, NUM_BEAMS,
)

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
N_SAMPLES = 50
SEED = 42


def sample_shared_rows(recap_root, lang, n_samples, seed, work_dir):
    """ONE sample of n_samples rows for this language, requiring Hindi,
    English, AND the target-language column all non-empty simultaneously
    (since every direction for this language will be built from the same
    rows) -- cached to <Lang>_shared50_base.csv so reruns are stable."""
    base_path = work_dir / f"{lang}_shared50_base.csv"
    if base_path.exists():
        df = pd.read_csv(base_path)
        if len(df) >= n_samples:
            return df
    test_csv = Path(recap_root) / "datasets" / DIR_NAME[lang] / "test.csv"
    df = pd.read_csv(test_csv)
    cols = ["Hindi", "English", lang]
    df = df[cols].dropna()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
    for c in cols:
        df = df[df[c] != ""]
    df = df.reset_index(drop=True)
    sampled = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(base_path, index=False, encoding="utf-8-sig")
    return sampled


def build_direction_samples(shared_df, lang, direction, work_dir):
    """Derives this direction's (source, reference) pair from the SAME
    shared 50-row base -- written in the identical format
    generate_human_eval_translations.py's sample_rows() produces, so
    downstream stages need no changes."""
    path = work_dir / f"{lang}_{direction}_samples.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if len(existing) >= len(shared_df):
            return existing
    if direction == "hi2tgt":
        src, ref = shared_df["Hindi"], shared_df[lang]
    elif direction == "tgt2hi":
        src, ref = shared_df[lang], shared_df["Hindi"]
    elif direction == "en2tgt":
        src, ref = shared_df["English"], shared_df[lang]
    else:  # tgt2en
        src, ref = shared_df[lang], shared_df["English"]
    out = pd.DataFrame({"source": src.tolist(), "reference": ref.tolist()})
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


SRC_NAME_BY_DIRECTION = {"hi2tgt": "Hindi", "en2tgt": "English"}
TGT_NAME_BY_DIRECTION = {"tgt2hi": "Hindi", "tgt2en": "English"}


def run_job(lang, direction, model_name, args, device, work_dir):
    tag = f"[{lang}/{direction}/{model_name}]"
    out_path = work_dir / f"{lang}_{direction}_{model_name}_translations.csv"
    if out_path.exists() and len(pd.read_csv(out_path)) >= args.n_samples:
        print(f"{tag} already done -- skipping")
        return

    shared = sample_shared_rows(args.recap_root, lang, args.n_samples, args.seed, work_dir)
    samples = build_direction_samples(shared, lang, direction, work_dir)
    src_texts = samples["source"].astype(str).tolist()

    src_name = SRC_NAME_BY_DIRECTION.get(direction, lang)
    tgt_name = TGT_NAME_BY_DIRECTION.get(direction, lang)

    checkpoint_path = str(Path(args.recap_root) / MODEL_ROOT[model_name] / lang /
                           f"{MODEL_PREFIX[model_name]}-{lang.lower()}-{direction}-finetuned")
    if not Path(checkpoint_path, "config.json").exists():
        print(f"{tag} skip -- no finished checkpoint at {checkpoint_path}")
        return

    print(f"{tag} generating {len(src_texts)} translations from {checkpoint_path} ...")
    preds = GEN_FN[model_name](checkpoint_path, src_texts, src_name, tgt_name, device)

    out_df = pd.DataFrame({"source": src_texts, "prediction": preds})
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{tag} -> {out_path}")


def _pool_init(gpu_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_queue.get())


def _pool_run(job_args):
    job, args, work_dir = job_args
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lang, direction, model_name = job
    run_job(lang, direction, model_name, args, device, work_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recap_root", default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP")
    ap.add_argument("--langs", default=",".join(LANGUAGES))
    ap.add_argument("--directions", default=",".join(DIRECTIONS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--work_dir", default=str(HERE / "_work_50"))
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build the shared 50-row base + all 4 directions' derived sample files
    # up front, sequentially, before any GPU work.
    for lang in langs:
        shared = sample_shared_rows(args.recap_root, lang, args.n_samples, args.seed, work_dir)
        for direction in directions:
            build_direction_samples(shared, lang, direction, work_dir)
    print(f"[sampling] done for {len(langs)} languages (shared across {len(directions)} directions each)")

    jobs = [(lang, direction, model_name) for lang in langs for direction in directions for model_name in models]
    print(f"[jobs] {len(jobs)} total")

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[gpu] {n_gpus} visible")

    if n_gpus <= 1:
        for job in jobs:
            _pool_run((job, args, work_dir))
    else:
        ctx = mp.get_context("spawn")
        gpu_queue = ctx.Queue()
        pool_size = min(n_gpus, len(jobs))
        for gpu_id in range(pool_size):
            gpu_queue.put(gpu_id)
        job_args = [(job, args, work_dir) for job in jobs]
        with ctx.Pool(processes=pool_size, initializer=_pool_init, initargs=(gpu_queue,)) as pool:
            pool.map(_pool_run, job_args)

    print("\nAll generation jobs done. Run assemble_human_eval_csv.py next, pointed at this --work_dir/--out_dir.")


if __name__ == "__main__":
    main()
