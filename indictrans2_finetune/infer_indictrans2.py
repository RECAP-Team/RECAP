"""
Run inference with the finetuned IndicTrans2 LoRA checkpoints (hi2tgt/tgt2hi,
all 4 languages) -- works whether a job finished normally (final adapter at
<output_root>/<Lang>/indictrans2-<lang>-<direction>-lora/, no checkpoint-N/
needed) or was killed mid-training by walltime/preemption (only the latest
checkpoint-N/ subdir survives, found via transformers.trainer_utils.
get_last_checkpoint() -- the same mechanism finetune_indictrans2.py's own
resume logic uses).

Reuses finetune_indictrans2.py's own add_bhili_tag()/resolve_lang_cfg()/
build_jobs()/GPU-pool machinery directly (import, not duplicate) -- Bhili's
adapter checkpoint only makes sense loaded on top of a base model that's
already had its encoder embedding table resized the same way training did
(add_bhili_tag warm-starts the new bhb_Deva row from hin_Deva; PEFT's
modules_to_save then overwrites that row with the trained one when the
adapter loads). Gondi/Mundari/Marathi need no such step (see
finetune_indictrans2.py's module docstring).

Real inference (unlike training's IndicProcessor(inference=False), which
only needs preprocessing for tokenization) needs IndicProcessor(inference=
True) for its postprocess_batch() too -- that's what denormalizes/detokenizes
IndicTrans2's output back into normal readable text.

Same GPU-pool pattern as finetune_indictrans2.py: one job per (language,
direction), min(n_gpus, n_jobs) workers, CUDA_VISIBLE_DEVICES set per-worker
in _pool_init() before torch is ever imported in that process.

Run (from this directory):
    python infer_indictrans2.py                       # all 4 langs x 2 directions
    python infer_indictrans2.py --langs Bhili --directions hi2tgt
    python infer_indictrans2.py --server server2
"""

import os
import json
import csv
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd

from finetune_indictrans2 import (
    HERE, HIN_CODE, EN_CODE, DIRECTIONS,
    resolve_lang_cfg, build_jobs, add_bhili_tag,
)

SAVE_EVERY_ROWS = 2000  # flush predictions to disk this often, for resumability
SCORES_CSV_NAME = "infer_scores.csv"


def append_scores(job_name, n_rows, bleu, chrf, output_dir, lock=None):
    scores_csv = Path(output_dir) / SCORES_CSV_NAME
    header = ["job", "rows", "bleu", "chrf++"]
    row = [job_name, n_rows, f"{bleu:.4f}", f"{chrf:.4f}"]

    def _write():
        is_new = not scores_csv.exists()
        with open(scores_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(header)
            w.writerow(row)

    if lock is not None:
        with lock:
            _write()
    else:
        _write()
    print(f"[Scores] Appended '{job_name}' to {scores_csv}")


def run_job(lang, direction, gpu_id, args, lang_cfg, lock=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from IndicTransToolkit import IndicProcessor
    from peft import PeftModel
    import sacrebleu

    device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tag = f"[{lang}/{direction} physical-gpu{gpu_id}]"

    cfg = lang_cfg[lang]
    lang_code = cfg["lang_code"]
    job_name = f"{lang.lower()}_{direction}"
    train_output_dir = str(Path(args.finetune_root) / lang / f"indictrans2-{lang.lower()}-{direction}-lora")

    # Each direction needs its own IndicTrans2 base checkpoint -- indic-indic
    # for Hindi<->tgt, en-indic for English->tgt, indic-en for tgt->English
    # (AI4Bharat ships these as three separate models, not one that covers
    # every direction) -- same selection as finetune_indictrans2.py's run_job().
    base_model = {
        "hi2tgt": args.base_model, "tgt2hi": args.base_model,
        "en2tgt": args.base_model_en_indic, "tgt2en": args.base_model_indic_en,
    }[direction]

    # Prefer a finished top-level adapter (model.save_pretrained ran
    # normally at the end of training); otherwise fall back to the latest
    # checkpoint-N/ (job was killed mid-training -- walltime, preemption).
    if Path(train_output_dir, "adapter_config.json").exists():
        checkpoint_path = train_output_dir
    else:
        checkpoint_path = get_last_checkpoint(train_output_dir) if Path(train_output_dir).is_dir() else None
        assert checkpoint_path is not None, \
            f"{tag} no finished adapter or checkpoint-N/ found under {train_output_dir}"

    print(f"\n===== {tag} =====")
    print(f"  base model: {base_model}")
    print(f"  adapter   : {checkpoint_path}")

    preds_path = Path(args.output_dir) / f"infer_predictions_{job_name}.csv"

    test_csv = str(Path(args.data_root) / cfg["dir_name"] / "test.csv")
    tgt_col = cfg["csv_col"]
    first_col = cfg["en_col"] if direction in ("en2tgt", "tgt2en") else cfg["hi_col"]
    df = pd.read_csv(test_csv)
    df = df[[first_col, tgt_col]].dropna()
    for c in (first_col, tgt_col):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[first_col] != "") & (df[tgt_col] != "")].reset_index(drop=True)

    if direction == "hi2tgt":
        src_lines, ref_lines = df[first_col].tolist(), df[tgt_col].tolist()
        src_code, dst_code = HIN_CODE, lang_code
        src_col_name, ref_col_name = "source_hindi", f"reference_{lang.lower()}"
    elif direction == "tgt2hi":
        src_lines, ref_lines = df[tgt_col].tolist(), df[first_col].tolist()
        src_code, dst_code = lang_code, HIN_CODE
        src_col_name, ref_col_name = f"source_{lang.lower()}", "reference_hindi"
    elif direction == "en2tgt":
        src_lines, ref_lines = df[first_col].tolist(), df[tgt_col].tolist()
        src_code, dst_code = EN_CODE, lang_code
        src_col_name, ref_col_name = "source_english", f"reference_{lang.lower()}"
    else:  # tgt2en
        src_lines, ref_lines = df[tgt_col].tolist(), df[first_col].tolist()
        src_code, dst_code = lang_code, EN_CODE
        src_col_name, ref_col_name = f"source_{lang.lower()}", "reference_english"
    total_rows = len(src_lines)
    print(f"{tag} rows to translate: {total_rows}")

    # ---- Resume check: how much of this job is already done? ----
    n_done = 0
    if preds_path.exists():
        n_done = len(pd.read_csv(preds_path))
        print(f"{tag} resuming: found {n_done}/{total_rows} rows already done in {preds_path}")

    if n_done >= total_rows:
        print(f"{tag} already fully done -- skipping generation.")
    else:
        print(f"{tag} loading base model + tokenizer ...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, trust_remote_code=True, attn_implementation="eager",
        ).to(device_tag)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        if cfg["needs_new_tag"]:
            # Must resize the embedding table the same way training did
            # BEFORE loading the adapter -- PEFT's modules_to_save state
            # for model.encoder.embed_tokens was saved at the resized
            # shape, and needs a matching-shape module already in place to
            # load into.
            add_bhili_tag(model, tokenizer, new_tag=lang_code, donor_tag=HIN_CODE)

        print(f"{tag} loading LoRA adapter from {checkpoint_path} ...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.eval()

        processor = IndicProcessor(inference=True)

        remaining_src = src_lines[n_done:]
        remaining_ref = ref_lines[n_done:]
        buf_src, buf_pred, buf_ref = [], [], []

        with torch.no_grad():
            for i in range(0, len(remaining_src), args.batch_size):
                chunk_src = remaining_src[i:i + args.batch_size]
                chunk_ref = remaining_ref[i:i + args.batch_size]
                batch = processor.preprocess_batch(chunk_src, src_lang=src_code, tgt_lang=dst_code)
                inputs = tokenizer(
                    batch, truncation=True, padding="longest",
                    return_tensors="pt", max_length=args.max_length,
                ).to(device_tag)
                out = model.generate(
                    **inputs, max_length=args.max_length,
                    num_beams=args.num_beams, num_return_sequences=1,
                )
                decoded = tokenizer.batch_decode(
                    out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                preds = processor.postprocess_batch(decoded, lang=dst_code)

                buf_src.extend(chunk_src)
                buf_pred.extend(preds)
                buf_ref.extend(chunk_ref)

                if (i // args.batch_size) % 20 == 0:
                    print(f"  {tag} decoded {n_done + i + len(chunk_src)}/{total_rows}")

                if len(buf_src) >= SAVE_EVERY_ROWS:
                    _flush(preds_path, src_col_name, buf_src, buf_pred, ref_col_name, buf_ref)
                    buf_src, buf_pred, buf_ref = [], [], []

        if buf_src:
            _flush(preds_path, src_col_name, buf_src, buf_pred, ref_col_name, buf_ref)
        print(f"{tag} predictions saved (incrementally) to {preds_path}")

    # ---- Score against the FULL predictions file (resumed + newly generated) ----
    full_preds = pd.read_csv(preds_path)
    preds_list = full_preds["prediction"].astype(str).tolist()
    refs_list = full_preds[ref_col_name].astype(str).tolist()

    bleu = sacrebleu.corpus_bleu(preds_list, [refs_list]).score
    chrf = sacrebleu.corpus_chrf(preds_list, [refs_list], word_order=2).score
    print(f"{tag} BLEU={bleu:.4f}  chrF++={chrf:.4f}")

    append_scores(job_name, len(full_preds), bleu, chrf, args.output_dir, lock=lock)
    print(f"{tag} done")


def _flush(preds_path, src_col_name, src_list, pred_list, ref_col_name, ref_list):
    out = pd.DataFrame({src_col_name: src_list, "prediction": pred_list, ref_col_name: ref_list})
    file_exists = os.path.exists(preds_path)
    out.to_csv(preds_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


# =============================================================
# GPU-pool orchestration -- same pattern as finetune_indictrans2.py, just
# dispatching to this file's own (inference) run_job instead of training's.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    lang, direction, args, lang_cfg = job_args
    run_job(lang, direction, _worker_gpu_id, args, lang_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--server", default="pragya")
    ap.add_argument("--langs", default="Bhili,Mundari,Gondi,Marathi")
    ap.add_argument("--directions", default="hi2tgt,tgt2hi")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--output_dir", default=str(HERE))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    assert args.server in cfg["servers"], \
        f"unknown --server {args.server!r}, choices: {list(cfg['servers'])}"
    server_cfg = cfg["servers"][args.server]
    args.base_model = server_cfg["base_model"]
    args.base_model_en_indic = server_cfg.get("base_model_en_indic")
    args.base_model_indic_en = server_cfg.get("base_model_indic_en")
    args.data_root = server_cfg["data_root"]
    args.finetune_root = server_cfg["output_root"]  # where finetune_indictrans2.py wrote checkpoints
    lang_cfg = resolve_lang_cfg(cfg["languages"], server_cfg)
    print(f"[server] {args.server!r}: base_model={args.base_model}  "
          f"data_root={args.data_root}  finetune_root={args.finetune_root}")

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    if "en2tgt" in directions:
        assert args.base_model_en_indic, \
            f"--directions includes en2tgt but server {args.server!r} has no 'base_model_en_indic' in config.json"
    if "tgt2en" in directions:
        assert args.base_model_indic_en, \
            f"--directions includes tgt2en but server {args.server!r} has no 'base_model_indic_en' in config.json"
    jobs = build_jobs(lang_cfg, langs, directions)
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

    job_args = [(lang, direction, args, lang_cfg) for lang, direction in jobs]
    with ctx.Pool(processes=pool_size, initializer=_pool_init, initargs=(gpu_queue,)) as pool:
        pool.map(_pool_run, job_args)

    print(f"\nAll jobs done. Scores: {Path(args.output_dir) / SCORES_CSV_NAME}")


if __name__ == "__main__":
    main()
