"""
LoRA-finetune ai4bharat/indictrans2-indic-indic-1B for Hindi<->Bhili, on the
CORRECTED Bhili data produced by the dataset-cleanup pass (see
correct_checkpoints/dataset_corrected/bhili/{train,val,test}.csv and
Removed_Rows_Report.pdf for what was fixed).

This is a self-contained copy of indictrans2_finetune/finetune_indictrans2.py,
adapted to read/write under correct_checkpoints/ instead of the original
indictrans2_finetune/ tree (its own config.json here has only "bhili" under
"languages", data_root/output_root both pointed at correct_checkpoints/),
so this corrected-data re-run cannot interfere with anything already
running against the original pipeline. infer_indictrans2.py (same
directory) imports directly from this file, same as the original pair.

The only functional addition vs. the original script: saves a per-job
training-log JSON/CSV + loss-curve PNG alongside the existing LoRA
adapter output (see save_training_logs_and_plot()). Everything else (LoRA
recipe, Bhili's add_bhili_tag() vocab patch, resumability, GPU pool,
hyperparameters) is unchanged from the original -- this follows
AI4Bharat's own official HF finetuning recipe as closely as possible
(huggingface_interface/train_lora.py/.sh in AI4Bharat/IndicTrans2), same
as before.

English directions (en2tgt/tgt2en) use AI4Bharat's separate en-indic/
indic-en 1B checkpoints instead of indic-indic -- see
"base_model_en_indic"/"base_model_indic_en" in config.json and run_job()'s
base_model selection.

Resumable: if a job is killed partway (walltime, preemption, crash), rerun
the exact same command -- run_job() checks for a finished adapter first
(skips outright if found) and otherwise resumes from the latest HF Trainer
checkpoint-N/ dir.

Bhili (bhb_Deva) is not one of IndicTrans2's pretrained languages:
add_bhili_tag() registers "bhb_Deva" as a new source-vocab token
(tokenizer.add_new_language_tags + a new row in
model.model.encoder.embed_tokens, warm-started from hin_Deva's embedding),
and that job's LoRA config additionally marks
model.model.encoder.embed_tokens as modules_to_save so the new row
actually trains alongside the LoRA adapters.

Run (auto-detects GPU count):
    python finetune_indictrans2.py --langs bhili --directions hi2tgt

Config (correct_checkpoints/indictrans2/bhili/config.json, same directory):
    {
      "servers": {
        "pragya": {
          "base_model": ".../RECAP/Models/IndicTrans2-indic-indic-1B",
          "base_model_en_indic": ".../RECAP/Models/IndicTrans2-en-indic-1B",
          "base_model_indic_en": ".../RECAP/Models/IndicTrans2-indic-en-1B",
          "data_root": ".../RECAP/correct_checkpoints/dataset_corrected",
          "output_root": ".../RECAP/correct_checkpoints/indictrans2"
        }
      },
      "languages": {
        "bhili": {"dir_name": "bhili", "hi_col": "Hindi", "csv_col": "Bhili", "en_col": "English", "lang_code": "bhb_Deva", "needs_new_tag": true}
      }
    }
"""

import os
import json
import csv
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

HERE = Path(__file__).resolve().parent
HIN_CODE = "hin_Deva"
EN_CODE = "eng_Latn"
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]


# =============================================================
# 1. Data loading -- same CSVs (and cleaning) as mt5_finetune.py/nllb_finetune.py
# =============================================================
def load_csv_pair(csv_path, cols):
    df = pd.read_csv(csv_path)
    assert all(c in df.columns for c in cols), \
        f"{csv_path}: need columns {cols}, got {list(df.columns)}"
    df = df[cols].dropna()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
        df = df[df[c] != ""]
    return [df[c].tolist() for c in cols]


def load_train_val_lines(cfg, direction):
    """Returns (train_first, train_tgt, val_first, val_tgt) from each
    language's presplit "train_csv" + "val_csv". "first" is the Hindi
    column for hi2tgt/tgt2hi jobs, or the English column ("en_col") for
    en2tgt/tgt2en jobs -- make_dataset() below picks which side is actually
    source vs. label based on `direction`."""
    tgt_col = cfg["csv_col"]
    if direction in ("en2tgt", "tgt2en"):
        first_col = cfg["en_col"]
    else:
        first_col = cfg["hi_col"]
    train_first, train_tgt = load_csv_pair(cfg["train_csv"], [first_col, tgt_col])
    val_first, val_tgt = load_csv_pair(cfg["val_csv"], [first_col, tgt_col])
    return train_first, train_tgt, val_first, val_tgt


def resolve_lang_cfg(languages, server_cfg):
    """Merge the server-independent language shapes (config.json's top-level
    "languages": dir_name/hi_col/csv_col/lang_code/needs_new_tag) with one
    server's data_root, producing the same per-language cfg dict shape
    run_job()/load_train_val_lines() already expect (train_csv/val_csv)."""
    data_root = server_cfg["data_root"]
    resolved = {}
    for lang, entry in languages.items():
        e = dict(entry)
        e["train_csv"] = str(Path(data_root) / e["dir_name"] / "train.csv")
        e["val_csv"] = str(Path(data_root) / e["dir_name"] / "val.csv")
        resolved[lang] = e
    return resolved


def build_jobs(languages_cfg, langs, directions):
    """One job per (language, direction). Order doesn't matter -- the GPU
    pool below pulls jobs off this list as workers free up."""
    jobs = []
    for lang in langs:
        assert lang in languages_cfg, f"unknown language {lang!r}, choices: {list(languages_cfg)}"
        for direction in directions:
            assert direction in DIRECTIONS, f"unknown direction {direction!r}, choices: {DIRECTIONS}"
            jobs.append((lang, direction))
    return jobs


# =============================================================
# 2. Bhili: register the missing source-side language tag.
# =============================================================
def add_bhili_tag(model, tokenizer, new_tag="bhb_Deva", donor_tag=HIN_CODE):
    import torch.nn as nn

    if new_tag in tokenizer.src_encoder:
        return  # already added (e.g. resumed from a checkpoint that has it)

    donor_id = tokenizer.src_encoder[donor_tag]
    new_id = len(tokenizer.src_encoder)
    tokenizer.src_encoder[new_tag] = new_id
    tokenizer.src_decoder[new_id] = new_tag
    tokenizer.add_new_language_tags([new_tag])

    embed = model.model.encoder.embed_tokens
    old_n, dim = embed.weight.shape
    assert new_id == old_n, f"vocab/embedding size mismatch: new_id={new_id} old_n={old_n}"

    new_embed = nn.Embedding(old_n + 1, dim, padding_idx=embed.padding_idx)
    new_embed.weight.data[:old_n] = embed.weight.data
    new_embed.weight.data[old_n] = embed.weight.data[donor_id].clone()
    new_embed = new_embed.to(dtype=embed.weight.dtype, device=embed.weight.device)

    model.model.encoder.embed_tokens = new_embed
    model.config.encoder_vocab_size = old_n + 1
    print(f"  [vocab] added {new_tag!r} as src-vocab id {new_id} "
          f"(warm-started from {donor_tag!r}, id {donor_id}); "
          f"encoder_vocab_size {old_n} -> {old_n + 1}")


# =============================================================
# 3. Preprocess + tokenize one (language, direction) dataset split
# =============================================================
def make_dataset(first_lines, tgt_lines, direction, lang_code, tokenizer, processor, args, seed=42):
    from datasets import Dataset

    # "first_lines" is Hindi for hi2tgt/tgt2hi, English for en2tgt/tgt2en --
    # see load_train_val_lines().
    if direction == "hi2tgt":
        src_lines, label_lines = first_lines, tgt_lines
        src_code, dst_code = HIN_CODE, lang_code
    elif direction == "tgt2hi":
        src_lines, label_lines = tgt_lines, first_lines
        src_code, dst_code = lang_code, HIN_CODE
    elif direction == "en2tgt":
        src_lines, label_lines = first_lines, tgt_lines
        src_code, dst_code = EN_CODE, lang_code
    else:  # tgt2en
        src_lines, label_lines = tgt_lines, first_lines
        src_code, dst_code = lang_code, EN_CODE

    data = {
        "sentence_SRC": processor.preprocess_batch(src_lines, src_lang=src_code, tgt_lang=dst_code, is_target=False),
        "sentence_TGT": processor.preprocess_batch(label_lines, src_lang=dst_code, tgt_lang=src_code, is_target=True),
    }
    ds = Dataset.from_dict(data).shuffle(seed=seed)

    def preprocess_fn(example):
        model_inputs = tokenizer(
            example["sentence_SRC"], truncation=True, padding=False, max_length=args.max_length
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["sentence_TGT"], truncation=True, padding=False, max_length=args.max_length
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(preprocess_fn, batched=True, num_proc=args.num_proc)


def compute_metrics_factory(tokenizer):
    from sacrebleu.metrics import BLEU, CHRF
    bleu_metric, chrf_metric = BLEU(), CHRF()

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id
        with tokenizer.as_target_tokenizer():
            preds = [p.strip() for p in tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
            labels = [l.strip() for l in tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)]
        return {
            "BLEU": bleu_metric.corpus_score(preds, [labels]).score,
            "chrF": chrf_metric.corpus_score(preds, [labels]).score,
        }
    return compute_metrics


def save_training_logs_and_plot(log_history, output_dir, lang, direction):
    """Saves the HF Trainer's per-step log history (train loss + periodic
    eval loss/BLEU/chrF) as JSON and CSV, and renders a loss-curve PNG."""
    out = Path(output_dir)
    tag = f"{lang.lower()}_{direction}"

    log_json_path = out / f"training_log_{tag}.json"
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    train_rows = [r for r in log_history if "loss" in r and "eval_loss" not in r]
    eval_rows = [r for r in log_history if "eval_loss" in r]

    log_csv_path = out / f"training_log_{tag}.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "eval_loss", "eval_BLEU", "eval_chrF", "learning_rate"])
        steps = sorted({r.get("step") for r in train_rows + eval_rows if r.get("step") is not None})
        train_by_step = {r["step"]: r for r in train_rows}
        eval_by_step = {r["step"]: r for r in eval_rows}
        for step in steps:
            tr = train_by_step.get(step, {})
            ev = eval_by_step.get(step, {})
            w.writerow([step, tr.get("loss", ""), ev.get("eval_loss", ""),
                        ev.get("eval_BLEU", ""), ev.get("eval_chrF", ""), tr.get("learning_rate", "")])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        if train_rows:
            ax.plot([r["step"] for r in train_rows], [r["loss"] for r in train_rows],
                     label="train loss", color="#1a3d6d")
        if eval_rows:
            ax.plot([r["step"] for r in eval_rows], [r["eval_loss"] for r in eval_rows],
                     label="eval loss", color="#b5460a", marker="o")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"IndicTrans2 LoRA -- {lang}/{direction} training loss")
        ax.legend()
        fig.tight_layout()
        plot_path = out / f"training_loss_plot_{tag}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  [Logs] training log/plot -> {log_json_path.name}, {log_csv_path.name}, {plot_path.name}")
    except Exception as e:
        print(f"  [Logs] [warn] could not render loss plot ({e}); log JSON/CSV still saved")


# =============================================================
# 4. One finetuning job: load base model+tokenizer fresh, wrap in LoRA,
#    train, save the adapter. Runs inside one GPU-pinned worker process.
# =============================================================
def run_job(lang, direction, gpu_id, args, lang_cfg):
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from IndicTransToolkit import IndicProcessor, IndicDataCollator
    from peft import LoraConfig, get_peft_model

    # CUDA_VISIBLE_DEVICES was set to just this worker's physical GPU in
    # _pool_init(), so exactly one device is visible here and it's always
    # index 0 from this process's point of view -- gpu_id (the real
    # physical GPU number) is kept only for logging below.
    device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tag = f"[{lang}/{direction} physical-gpu{gpu_id}]"

    cfg = lang_cfg[lang]
    lang_code = cfg["lang_code"]
    # Each direction needs its own IndicTrans2 base checkpoint -- indic-indic
    # for Hindi<->tgt, en-indic for English->tgt, indic-en for tgt->English.
    base_model = {
        "hi2tgt": args.base_model, "tgt2hi": args.base_model,
        "en2tgt": args.base_model_en_indic, "tgt2en": args.base_model_indic_en,
    }[direction]
    suffix = "-smoketest" if args.smoke_test else ""
    output_dir = str(Path(args.output_root) / lang / f"indictrans2-{lang.lower()}-{direction}-lora{suffix}")
    print(f"\n===== {tag}{' [SMOKE TEST]' if args.smoke_test else ''} =====")
    print(f"  base model: {base_model}")
    print(f"  output    : {output_dir}")

    # Whole job already finished in a previous run (final adapter present)?
    # Skip it outright -- resuming would just redo an already-complete job.
    if Path(output_dir, "adapter_config.json").exists():
        print(f"{tag} already finished (found {output_dir}/adapter_config.json) -- skipping")
        return

    # Otherwise: did a previous run of THIS job get partway through and die
    # (walltime, preemption, crash)? get_last_checkpoint() finds the latest
    # HF Trainer checkpoint-N/ dir (model + optimizer + scheduler + RNG
    # state -- save_total_limit=1 below keeps exactly one, so this is
    # always the most recent). Passed to trainer.train() further down.
    resume_from_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if resume_from_checkpoint:
        print(f"{tag} resuming training from {resume_from_checkpoint}")

    print(f"{tag} loading base model + tokenizer ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, trust_remote_code=True, attn_implementation="eager",
        dropout=args.dropout,
    ).to(device_tag)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    processor = IndicProcessor(inference=False)

    if cfg["needs_new_tag"]:
        add_bhili_tag(model, tokenizer, new_tag=lang_code, donor_tag=HIN_CODE)

    print(f"{tag} loading data ({cfg['train_csv']}) ...")
    train_first, train_tgt, val_first, val_tgt = load_train_val_lines(cfg, direction)
    if args.smoke_test:
        train_first, train_tgt = train_first[:args.smoke_train_rows], train_tgt[:args.smoke_train_rows]
        val_first, val_tgt = val_first[:args.smoke_val_rows], val_tgt[:args.smoke_val_rows]
    train_ds = make_dataset(train_first, train_tgt, direction, lang_code, tokenizer, processor, args)
    eval_ds = make_dataset(val_first, val_tgt, direction, lang_code, tokenizer, processor, args)
    print(f"{tag} train={len(train_ds)}  val={len(eval_ds)}")

    data_collator = IndicDataCollator(
        tokenizer=tokenizer, model=model, padding="longest",
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    lora_kwargs = {}
    if cfg["needs_new_tag"]:
        # The new tag's embedding row must actually train -- LoRA alone
        # (q_proj/k_proj only) never touches the embedding table. Marking
        # the whole table as modules_to_save looks broad, but embedding
        # gradients are sparse: only rows that appear in this job's
        # batches get nonzero gradient, so every other language's row is
        # left untouched.
        lora_kwargs["modules_to_save"] = ["model.encoder.embed_tokens"]

    lora_config = LoraConfig(
        r=args.lora_r, bias="none", inference_mode=False, task_type="SEQ_2_SEQ_LM",
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        **lora_kwargs,
    )
    model.set_label_smoothing(args.label_smoothing)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Smoke test: same recipe, just fast.
    if args.smoke_test:
        max_steps = args.smoke_max_steps
        save_steps = eval_steps = min(5, max_steps)
        warmup_steps = min(2, max_steps)
        logging_steps = 1
    else:
        max_steps = args.max_steps
        save_steps, eval_steps = args.save_steps, args.eval_steps
        warmup_steps = args.warmup_steps
        logging_steps = 100

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True, do_eval=True,
        fp16=args.fp16,
        logging_strategy="steps", evaluation_strategy="steps", save_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_accumulation_steps=args.grad_accum_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=warmup_steps,
        learning_rate=args.learning_rate,
        save_steps=save_steps,
        eval_steps=eval_steps,
        # Must stay 0, always -- each job runs inside its own daemon Pool
        # worker, and daemon processes cannot spawn their own children.
        dataloader_num_workers=0,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        report_to="none",
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        group_by_length=True,
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args, data_collator=data_collator,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.patience, early_stopping_threshold=args.threshold)],
    )

    print(f"{tag} training ...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print(f"{tag} interrupted -- a checkpoint-N/ dir should still be on "
              f"disk under {output_dir} for the next run to resume from")

    model.save_pretrained(output_dir)           # LoRA adapter (+ modules_to_save, if any)
    tokenizer.save_pretrained(output_dir)        # needed too when a new tag was added
    print(f"{tag} done -> {output_dir}")

    save_training_logs_and_plot(trainer.state.log_history, output_dir, lang, direction)


# =============================================================
# 5. GPU-pool orchestration
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    # Restrict this worker process to exactly one physical GPU -- must be
    # done here, before torch is ever imported in this process (run_job()
    # imports it lazily).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    lang, direction, args, lang_cfg = job_args
    run_job(lang, direction, _worker_gpu_id, args, lang_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--server", default="pragya",
                    help="which config.json['servers'][...] entry to use for "
                         "base_model/data_root/output_root paths")
    ap.add_argument("--langs", default="bhili")
    ap.add_argument("--directions", default="hi2tgt,tgt2hi")
    # Hyperparameters -- defaults are AI4Bharat's own recommended values.
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=1_000_000)  # unbounded; early stopping decides
    ap.add_argument("--warmup_steps", type=int, default=4000)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.98)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--optimizer", default="adamw_torch")
    ap.add_argument("--lr_scheduler", default="inverse_sqrt")
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--metric_for_best_model", default="eval_BLEU")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--lora_target_modules", default="q_proj,k_proj")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--fp16", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--smoke_train_rows", type=int, default=300)
    ap.add_argument("--smoke_val_rows", type=int, default=60)
    ap.add_argument("--smoke_max_steps", type=int, default=10)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    assert args.server in cfg["servers"], \
        f"unknown --server {args.server!r}, choices: {list(cfg['servers'])} (edit config.json to add more)"
    server_cfg = cfg["servers"][args.server]
    args.base_model = server_cfg["base_model"]
    args.base_model_en_indic = server_cfg.get("base_model_en_indic")
    args.base_model_indic_en = server_cfg.get("base_model_indic_en")
    args.output_root = server_cfg["output_root"]
    lang_cfg = resolve_lang_cfg(cfg["languages"], server_cfg)
    print(f"[server] {args.server!r}: base_model={args.base_model}  data_root={server_cfg['data_root']}  output_root={args.output_root}")

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

    print("\nAll jobs done.")


if __name__ == "__main__":
    main()
