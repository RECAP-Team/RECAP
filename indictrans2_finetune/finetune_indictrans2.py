"""
LoRA-finetune ai4bharat/indictrans2-indic-indic-1B for Hindi<->{Bhili,Mundari,
Gondi,Marathi}, both directions (8 jobs total), auto-detecting GPUs and
running one job per GPU in parallel (a fixed-size worker pool pulls from a
job queue -- works whether you request exactly as many GPUs as jobs, fewer,
or more).

Resumable: if a job is killed partway (walltime, preemption, crash), rerun
the exact same command -- run_job() checks for a finished adapter first
(skips outright if found) and otherwise for the latest HF Trainer
checkpoint-N/ dir (save_total_limit=1 keeps exactly one, with full
optimizer/scheduler/RNG state) and resumes training from there via
trainer.train(resume_from_checkpoint=...). Nothing to pass on the CLI --
this is automatic per job.

This follows AI4Bharat's own official HF finetuning recipe as closely as
possible, verified directly from their source (not reconstructed from
memory) -- see huggingface_interface/train_lora.py + train_lora.sh in
https://github.com/AI4Bharat/IndicTrans2:
  - LoRA via peft (AI4Bharat publish no other HF finetuning path for
    IndicTrans2 -- train_lora.py/.sh is the only officially maintained one)
  - IndicTransToolkit's IndicProcessor(inference=False).preprocess_batch()
    for text normalization/tokenization before the model's own tokenizer
    (this is what actually prepends the "<src_lang> <tgt_lang> " tag prefix
    the model was pretrained on)
  - IndicDataCollator (also from IndicTransToolkit)
  - Hyperparameters taken verbatim from train_lora.sh's recommended
    defaults: lr=2e-4, warmup_steps=4000, adamw_torch, inverse_sqrt schedule,
    batch_size=8/grad_accum_steps=16 (effective batch 128, same as
    AI4Bharat's 32/4 -- per-device batch lowered for GPU-memory headroom on
    shared/contended GPUs; override --batch_size/--grad_accum_steps to change),
    adam_beta=(0.9, 0.98), max_grad_norm=1.0, weight_decay=0.01,
    LoRA r=16/alpha=32/dropout=0.1 on q_proj,k_proj, fp16, early stopping
    on eval_BLEU (patience=10), generation beam=5/max_length=256.
  - metrics via sacrebleu.metrics.BLEU/CHRF .corpus_score(), matching the
    official compute_metrics_factory exactly.

One thing AI4Bharat's script does NOT need to handle, that we do: Bhili
(bhb_Deva) is not one of IndicTrans2's languages. Gondi (gon_Deva) and
Mundari (unr_Deva) already have real, pretrained source-side language-tag
embeddings and IndicProcessor entries -- verified directly against the
downloaded checkpoint's dict.SRC.json (gon_Deva id=121516, unr_Deva
id=121515) and IndicTransToolkit's IndicProcessor._flores_codes (both map
to "hi" for normalization) -- so those two need no special handling at all.
Bhili does: add_bhili_tag() registers "bhb_Deva" as a new source-vocab
token (tokenizer.add_new_language_tags + a new row in
model.model.encoder.embed_tokens, warm-started from hin_Deva's embedding --
Bhili is the closest existing tag linguistically), and that job's LoRA
config additionally marks model.model.encoder.embed_tokens as
modules_to_save so the new row (and only the new row -- embedding
gradients are sparse, so hin_Deva/other-language rows that don't appear in
Bhili's batches get zero gradient and stay untouched) actually trains
alongside the LoRA adapters instead of staying frozen and random.
IndicProcessor itself needs no Bhili-specific entry: its _flores_codes
lookup already falls back to "hi" for any unrecognized tag.

Language tags used are only ever on the SOURCE side (both src_lang and
tgt_lang get prepended to the *source* sequence -- see
IndicTransTokenizer._src_tokenize); the target vocab carries no language
tags at all, so no target-side vocab changes are ever needed.

Data: read directly from <data_root>/<Lang>/{train,val}.csv for all four
languages (see load_train_val_lines()) -- Bhili/Mundari/Gondi have columns
English,Hindi,<Lang> or Unique_ID,Hindi,<Lang>,English; Marathi has
unique_id,Hindi,Marathi. Only Hindi/<Lang> are read in every case. NOT
AI4Bharat's own one-sentence-per-line file layout.

Multi-server: config.json has a top-level "servers": {name: {base_model,
data_root, output_root}} map plus one shared "languages" map (dir_name/
hi_col/csv_col/lang_code/needs_new_tag -- these don't change across
servers, only where the data/model/output actually live does).
--server picks which one (default "pragya"); resolve_lang_cfg() joins the
shared per-language dir_name onto that server's data_root at runtime, so
adding a third server is just one more entry under "servers".

Run (auto-detects GPU count):
    python finetune_indictrans2.py                       # --server pragya (default)
    python finetune_indictrans2.py --server server2
    python finetune_indictrans2.py --langs Bhili --directions hi2tgt

Smoke test first (same job list, same code paths, ~300 train / ~60 val
rows per job, ~10 steps -- finishes in minutes, writes to a
"-smoketest" suffixed output dir that never collides with a real run):
    python finetune_indictrans2.py --langs Bhili,Mundari,Gondi,Marathi \
        --directions hi2tgt,tgt2hi --smoke_test
Check every job printed "... done -> .../indictrans2-<lang>-<direction>
-lora-smoketest" with no traceback, then rerun the same command without
--smoke_test (or qsub run_indictrans2_finetune.pbs) for the real run.

Config (indictrans2_finetune/config.json, same directory) -- each language
entry is fully self-contained (own CSV paths), since Marathi's data lives
in a different directory tree than the other three. See config.json.
"""

import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import pandas as pd

HERE = Path(__file__).resolve().parent
HIN_CODE = "hin_Deva"
DIRECTIONS = ["hi2tgt", "tgt2hi"]


# =============================================================
# 1. Data loading -- same CSVs (and cleaning) as mt5_finetune.py/nllb_finetune.py
# =============================================================
def load_csv_pair(csv_path, hi_col, tgt_col):
    df = pd.read_csv(csv_path)
    assert hi_col in df.columns and tgt_col in df.columns, \
        f"{csv_path}: need columns {hi_col!r},{tgt_col!r}, got {list(df.columns)}"
    df = df[[hi_col, tgt_col]].dropna()
    for c in (hi_col, tgt_col):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[hi_col] != "") & (df[tgt_col] != "")]
    return df[hi_col].tolist(), df[tgt_col].tolist()


def load_train_val_lines(cfg):
    """Returns (train_hi, train_tgt, val_hi, val_tgt) from each language's
    presplit "train_csv" + "val_csv" (RECAP/datasets/<Lang>/*.csv for all
    four languages, incl. Marathi now that it's laid out the same way)."""
    hi_col, tgt_col = cfg["hi_col"], cfg["csv_col"]
    train_hi, train_tgt = load_csv_pair(cfg["train_csv"], hi_col, tgt_col)
    val_hi, val_tgt = load_csv_pair(cfg["val_csv"], hi_col, tgt_col)
    return train_hi, train_tgt, val_hi, val_tgt


def resolve_lang_cfg(languages, server_cfg):
    """Merge the server-independent language shapes (config.json's top-level
    "languages": dir_name/hi_col/csv_col/lang_code/needs_new_tag -- these
    don't change across servers) with one server's data_root, producing the
    same per-language cfg dict shape run_job()/load_train_val_lines()
    already expect (train_csv/val_csv), just pointed at wherever this
    server's copy of the data lives."""
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
#    (Gondi/Mundari need none of this -- see module docstring.)
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
# 3. Preprocess + tokenize one (language, direction) dataset split,
#    mirroring AI4Bharat's load_and_process_translation_dataset() /
#    preprocess_fn() exactly, just sourced from CSV-loaded line lists
#    (via load_train_val_lines() above) instead of one-sentence-per-line
#    files.
# =============================================================
def make_dataset(hi_lines, tgt_lines, direction, lang_code, tokenizer, processor, args, seed=42):
    from datasets import Dataset

    if direction == "hi2tgt":
        src_lines, label_lines = hi_lines, tgt_lines
        src_code, dst_code = HIN_CODE, lang_code
    else:  # tgt2hi
        src_lines, label_lines = tgt_lines, hi_lines
        src_code, dst_code = lang_code, HIN_CODE

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
    suffix = "-smoketest" if args.smoke_test else ""
    output_dir = str(Path(args.output_root) / lang / f"indictrans2-{lang.lower()}-{direction}-lora{suffix}")
    print(f"\n===== {tag}{' [SMOKE TEST]' if args.smoke_test else ''} =====")
    print(f"  base model: {args.base_model}")
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
        args.base_model, trust_remote_code=True, attn_implementation="eager",
        dropout=args.dropout,
    ).to(device_tag)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    processor = IndicProcessor(inference=False)

    if cfg["needs_new_tag"]:
        add_bhili_tag(model, tokenizer, new_tag=lang_code, donor_tag=HIN_CODE)

    print(f"{tag} loading data ({cfg['train_csv']}) ...")
    train_hi, train_tgt, val_hi, val_tgt = load_train_val_lines(cfg)
    if args.smoke_test:
        train_hi, train_tgt = train_hi[:args.smoke_train_rows], train_tgt[:args.smoke_train_rows]
        val_hi, val_tgt = val_hi[:args.smoke_val_rows], val_tgt[:args.smoke_val_rows]
    train_ds = make_dataset(train_hi, train_tgt, direction, lang_code, tokenizer, processor, args)
    eval_ds = make_dataset(val_hi, val_tgt, direction, lang_code, tokenizer, processor, args)
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
        # batches (hin_Deva, bhb_Deva, ordinary subwords) get nonzero
        # gradient, so every other language's row is left untouched.
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

    # Smoke test: same recipe, just fast -- a handful of steps with
    # eval/save/logging tight enough to actually happen (so the
    # eval -> checkpoint -> save_pretrained path all get exercised) instead
    # of the real 1000-step intervals never being reached.
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
        # Must stay 0, always: each job already runs inside its own
        # multiprocessing.Pool worker (see the GPU pool below), and Pool
        # workers are daemon processes -- Python forbids a daemon process
        # from spawning children, so any dataloader_num_workers > 0 here
        # crashes with "daemonic processes are not allowed to have
        # children" the moment Trainer tries to spawn its own dataloader
        # workers. Parallelism already comes from the outer GPU pool
        # (multiple jobs at once), not from per-job dataloader workers.
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


# =============================================================
# 5. GPU-pool orchestration -- same pattern as
#    IE_logit_lens/infer_hidden_states.py: a fixed pool of
#    min(n_gpus, n_jobs) worker processes, each pinned to one GPU for its
#    whole lifetime, pulling jobs off a queue -- works whether there are
#    fewer, equal, or more jobs than GPUs.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    # Restrict this worker process to exactly one physical GPU -- must be
    # done here, before torch is ever imported in this process (run_job()
    # imports it lazily), and via CUDA_VISIBLE_DEVICES rather than just
    # torch.cuda.set_device() later. set_device() only changes the default
    # device; it doesn't stop transformers/accelerate from seeing every
    # GPU and spreading a single job's compute across all of them, which
    # is exactly what happened without this -- each worker ended up using
    # both physical GPUs instead of just its own, doubling memory/compute
    # contention. With this restriction the assigned GPU is always index 0
    # from this process's point of view (see run_job()).
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
    ap.add_argument("--langs", default="Bhili,Mundari,Gondi")
    ap.add_argument("--directions", default="hi2tgt,tgt2hi")
    # Hyperparameters -- defaults are AI4Bharat's own recommended values
    # from huggingface_interface/train_lora.sh (verified against the repo,
    # not guessed).
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    # AI4Bharat's own default is batch_size=32/grad_accum_steps=4 (effective
    # 128). Lowered per-device batch here (server2 hit real GPU contention
    # from another process, not a bug in this script) while keeping the
    # same effective batch size by raising grad_accum_steps proportionally
    # -- override either independently with these flags if needed.
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
    # NOTE: no --num_workers flag -- dataloader_num_workers is hardcoded to
    # 0 in run_job() and must stay that way (see the comment there): each
    # job runs inside a daemon Pool worker, which cannot itself spawn the
    # dataloader's worker subprocesses.
    ap.add_argument("--num_proc", type=int, default=4)
    # Smoke test: same job list, same code paths (incl. add_bhili_tag and
    # the GPU pool), just a tiny slice of data and a handful of steps so it
    # finishes in minutes instead of hours. Writes to
    # <output_root>/<Lang>/indictrans2-<lang>-<direction>-lora-smoketest/
    # -- a separate dir from the real run, so it can never be mistaken for
    # (or block resume of) a real job.
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
    args.output_root = server_cfg["output_root"]
    lang_cfg = resolve_lang_cfg(cfg["languages"], server_cfg)
    print(f"[server] {args.server!r}: base_model={args.base_model}  data_root={server_cfg['data_root']}  output_root={args.output_root}")

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
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
