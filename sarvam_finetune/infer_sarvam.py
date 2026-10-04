"""
Run inference with the finetuned Sarvam-Translate checkpoints (full
fine-tune, see sarvam_finetune.py -- not LoRA) on BOTH val.csv and test.csv,
across any number of (language, direction) jobs. Sarvam had no standalone
infer.py before this (unlike mt5/NLLB/Qwen/Llama) -- this one mirrors their
exact output convention (infer_predictions_<lang>_<direction>_<split>.csv,
columns source_<src_col.lower()>, prediction, reference_<tgt_col.lower()>)
so compute_spbleu_scores.py can score all 5 models identically, with no
Sarvam-specific special-casing needed.

Reuses sarvam_finetune.py's own load_splits()/build_messages() directly
(import, not duplicate) -- same chat prompt, same src/tgt column resolution
per direction, so predictions are generated exactly the way training itself
would score them.

Same GPU-pool pattern as sarvam_finetune.py/finetune_indictrans2.py: one job
per GPU, min(n_gpus, n_jobs) workers, CUDA_VISIBLE_DEVICES set per-worker
before torch is ever imported in that process.

A (language, direction) whose checkpoint doesn't exist yet (e.g. English
directions still training as of this writing) is skipped with a printed
warning, not a crash -- rerun later once more checkpoints finish.

Run (from this directory):
    python infer_sarvam.py
    python infer_sarvam.py --langs Bhili --directions hi2tgt,tgt2hi
"""

import os
import csv
import json
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import pandas as pd

from sarvam_finetune import load_splits, build_messages, HERE

MAX_LENGTH = 512
MAX_NEW_TOKENS = 256
NUM_BEAMS = 4
BATCH_SIZE = 16
SAVE_EVERY_ROWS = 2000
SPLITS = ["val", "test"]
ALL_DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]

SCORES_CSV = Path("./infer_scores.csv")


def append_scores(job_name, n_rows, bleu, chrfpp, lock=None):
    header = ["job", "rows", "bleu", "chrf++"]
    row = [job_name, n_rows, f"{bleu:.4f}" if bleu is not None else "",
           f"{chrfpp:.4f}" if chrfpp is not None else ""]

    def _write():
        is_new = not SCORES_CSV.exists()
        with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(header)
            w.writerow(row)

    if lock is not None:
        with lock:
            _write()
    else:
        _write()
    print(f"[Scores] Appended '{job_name}' to {SCORES_CSV}")


def _flush(preds_path, src_col, src_list, pred_list, ref_col, ref_list):
    out = pd.DataFrame({src_col: src_list, "prediction": pred_list, ref_col: ref_list})
    file_exists = os.path.exists(preds_path)
    out.to_csv(preds_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


def run_job(lang, direction, split, gpu_id, lang_cfg, output_root, lock=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sacrebleu

    device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tag = f"[{lang}/{direction}/{split} physical-gpu{gpu_id}]"

    cfg = lang_cfg[lang]
    if direction in ("en2tgt", "tgt2en") and "en_col" not in cfg:
        print(f"{tag} skip -- no en_col for {lang} (English directions unsupported)")
        return

    checkpoint_path = str(Path(output_root) / lang / f"sarvam-{lang.lower()}-{direction}-finetuned")
    if not Path(checkpoint_path, "config.json").exists():
        print(f"{tag} skip -- no finished checkpoint at {checkpoint_path}")
        return

    job_name = f"{lang.lower()}_{direction}_{split}"
    preds_path = Path(output_root) / lang / f"infer_predictions_{job_name}.csv"
    print(f"\n===== {tag} =====")
    print(f"  checkpoint: {checkpoint_path}")

    (train_src, train_lbl), (val_src, val_lbl), (test_src, test_lbl), tgt_name = load_splits(cfg, direction)
    src_texts, ref_texts = (val_src, val_lbl) if split == "val" else (test_src, test_lbl)
    total_rows = len(src_texts)

    # Column names matching the other 4 models' infer.py convention
    # (source_<src>, reference_<tgt>) so compute_spbleu_scores.py's existing
    # TGT_COL_BY_DIRECTION logic works for Sarvam unmodified too.
    src_name_col = {"hi2tgt": "Hindi", "en2tgt": "English"}.get(direction, cfg["tgt_col"])
    src_out_col = f"source_{src_name_col.lower()}"
    ref_out_col = f"reference_{tgt_name.lower()}"

    n_done = 0
    if preds_path.exists():
        n_done = len(pd.read_csv(preds_path))
        print(f"{tag} resuming: {n_done}/{total_rows} rows already done")

    if n_done >= total_rows:
        print(f"{tag} already fully done -- skipping generation")
    else:
        print(f"{tag} loading model + tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16).to(device_tag)
        model.eval()
        tokenizer.padding_side = "left"

        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

        remaining_src = src_texts[n_done:]
        remaining_ref = ref_texts[n_done:]
        buf_src, buf_pred, buf_ref = [], [], []

        with torch.no_grad():
            for i in range(0, len(remaining_src), BATCH_SIZE):
                chunk = remaining_src[i:i + BATCH_SIZE]
                chunk_ref = remaining_ref[i:i + BATCH_SIZE]
                prompts = [
                    tokenizer.apply_chat_template(
                        build_messages(tgt_name, t), tokenize=False, add_generation_prompt=True)
                    for t in chunk
                ]
                enc = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=MAX_LENGTH, add_special_tokens=False,
                ).to(device_tag)
                out = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                    eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
                )
                gen_only = out[:, enc["input_ids"].shape[1]:]
                decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                buf_src.extend(chunk)
                buf_pred.extend(p.strip() for p in decoded)
                buf_ref.extend(chunk_ref)

                if (i // BATCH_SIZE) % 20 == 0:
                    print(f"  {tag} decoded {n_done + i + len(chunk)}/{total_rows}")

                if len(buf_src) >= SAVE_EVERY_ROWS:
                    _flush(preds_path, src_out_col, buf_src, buf_pred, ref_out_col, buf_ref)
                    buf_src, buf_pred, buf_ref = [], [], []

        if buf_src:
            _flush(preds_path, src_out_col, buf_src, buf_pred, ref_out_col, buf_ref)
        tokenizer.padding_side = "right"
        print(f"{tag} predictions saved -> {preds_path}")

    full_preds = pd.read_csv(preds_path)
    preds_list = full_preds["prediction"].astype(str).tolist()
    refs_list = full_preds[ref_out_col].astype(str).tolist()
    bleu = sacrebleu.corpus_bleu(preds_list, [refs_list]).score
    chrfpp = sacrebleu.corpus_chrf(preds_list, [refs_list], word_order=2).score
    print(f"{tag} BLEU={bleu:.4f}  chrF++={chrfpp:.4f}")
    append_scores(job_name, len(full_preds), bleu, chrfpp, lock=lock)
    print(f"{tag} done")


# =============================================================
# GPU-pool orchestration -- identical pattern to sarvam_finetune.py.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    lang, direction, split, lang_cfg, output_root = job_args
    run_job(lang, direction, split, _worker_gpu_id, lang_cfg, output_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--langs", default="Bhili,Mundari,Gondi")
    ap.add_argument("--directions", default=",".join(ALL_DIRECTIONS))
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    output_root = cfg["output_root"]
    data_root = cfg["data_root"]

    lang_cfg = {}
    for lang, entry in cfg["languages"].items():
        e = dict(entry)
        e["train_csv"] = str(Path(data_root) / e["dir_name"] / "train.csv")
        e["val_csv"] = str(Path(data_root) / e["dir_name"] / "val.csv")
        e["test_csv"] = str(Path(data_root) / e["dir_name"] / "test.csv")
        lang_cfg[lang] = e

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    jobs = [(lang, direction, split) for lang in langs for direction in directions for split in splits]
    print(f"[jobs] {len(jobs)} total: {jobs}")

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[gpu] {n_gpus} visible")
    if n_gpus == 0:
        raise SystemExit("No GPU visible -- request GPUs on the job scheduler before running this.")

    pool_size = min(n_gpus, len(jobs))
    print(f"[pool] {pool_size} worker(s), {len(jobs)} job(s) queued")

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu_id in range(pool_size):
        gpu_queue.put(gpu_id)

    job_args = [(lang, direction, split, lang_cfg, output_root) for lang, direction, split in jobs]
    with ctx.Pool(processes=pool_size, initializer=_pool_init, initargs=(gpu_queue,)) as pool:
        pool.map(_pool_run, job_args)

    print(f"\nAll jobs done. Scores: {SCORES_CSV}")


if __name__ == "__main__":
    main()
