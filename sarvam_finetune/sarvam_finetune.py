"""
Full fine-tune sarvamai/sarvam-translate for Hindi<->{Bhili,Mundari,Gondi,
Marathi}, both directions (8 jobs total), auto-detecting GPUs and running
one job per GPU in parallel (a fixed-size worker pool pulls from a job
queue -- works whether you request fewer, equal, or more GPUs than jobs).
Pragya only -- no multi-server config, unlike indictrans2_finetune/.

Unlike indictrans2_finetune/ (LoRA on an encoder-decoder model), this is
FULL fine-tuning of a decoder-only causal LM, one independent job per GPU
(no DDP/DeepSpeed -- each job trains standalone on a single GPU, matching
this repo's existing qwen_finetune.py/llama_finetune.py SFT recipe, not
their torchrun+ZeRO orchestration). Sarvam-Translate (~4B params, built on
Gemma3-4B-it) is architecturally the same class of model as those two:

  - No forced_bos_token_id / task-prefix trick (that's the mT5/NLLB/
    IndicTrans2 encoder-decoder mechanism). Direction is a chat prompt.
  - Loss is computed ONLY on the assistant's response tokens (the
    prompt is masked with label id -100) -- same response-only SFT
    masking as qwen_finetune.py/llama_finetune.py.

Prompt format matches Sarvam-Translate's own documented usage exactly
(verified against the model's real README quickstart + chat_template.jinja
-- NOT copied from the Qwen/Llama tribal scripts' "Translate from X to Y"
convention, which is a different, unrelated format):

    system: "Translate the text below to {tgt_name}."
    user:   <source text, verbatim -- no instruction, no source-language name>
    assistant: <target text>

Sarvam's tokenizer ships a real chat_template (tokenizer.apply_chat_template
works directly, unlike Llama-3.1-8B-base) and a real pad token, so neither
of those workarounds from llama_finetune.py are needed here.

Full-finetuning ~4B params on a SINGLE GPU (no ZeRO sharding, since each
job is independent and alone on its GPU) is memory-tight -- gradient
checkpointing + 8-bit AdamW (bitsandbytes) are enabled to make it fit;
these don't change training correctness, just memory footprint.

Resumable: same pattern as indictrans2_finetune/ -- a job already fully
saved is skipped outright; a job that got partway through resumes from its
last HF Trainer checkpoint (single-process-per-GPU here, so, unlike
llama_finetune.py's documented DDP resume bug, resume_from_checkpoint is
safe to use).

Data: RECAP/datasets/<Lang>/{train,val,test}.csv (same files/columns the
mt5/nllb/indictrans2 tribal finetunes use).

Run (auto-detects GPU count):
    python sarvam_finetune.py
    python sarvam_finetune.py --langs Bhili --directions hi2tgt
    python sarvam_finetune.py --smoke_test   # tiny slice, few steps, fast

Config (sarvam_finetune/config.json, same directory):
    {
      "base_model": ".../RECAP/Models/Sarvam-Translate",
      "data_root": ".../RECAP/datasets",
      "output_root": ".../RECAP/sarvam_finetune",
      "languages": {
        "Bhili":   {"dir_name": "bhilli",  "tgt_col": "Bhili"},
        "Mundari": {"dir_name": "Mundari", "tgt_col": "Mundari"},
        "Gondi":   {"dir_name": "Gondi",   "tgt_col": "Gondi"},
        "Marathi": {"dir_name": "Marathi", "tgt_col": "Marathi"}
      }
    }
"""

import os
import csv
import json
import argparse
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd

HERE = Path(__file__).resolve().parent
HIN_NAME = "Hindi"
DIRECTIONS = ["hi2tgt", "tgt2hi"]


# =============================================================
# 1. Data loading -- same CSVs (and cleaning) as the other tribal finetunes
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


def load_splits(cfg):
    hi_col, tgt_col = "Hindi", cfg["tgt_col"]
    train_hi, train_tgt = load_csv_pair(cfg["train_csv"], hi_col, tgt_col)
    val_hi, val_tgt = load_csv_pair(cfg["val_csv"], hi_col, tgt_col)
    test_hi, test_tgt = load_csv_pair(cfg["test_csv"], hi_col, tgt_col)
    return (train_hi, train_tgt), (val_hi, val_tgt), (test_hi, test_tgt)


def build_jobs(languages_cfg, langs, directions):
    jobs = []
    for lang in langs:
        assert lang in languages_cfg, f"unknown language {lang!r}, choices: {list(languages_cfg)}"
        for direction in directions:
            assert direction in DIRECTIONS, f"unknown direction {direction!r}, choices: {DIRECTIONS}"
            jobs.append((lang, direction))
    return jobs


# =============================================================
# 2. Prompt formatting + preprocessing (response-only loss masking)
#    Matches Sarvam-Translate's own documented chat format exactly.
# =============================================================
def build_messages(tgt_name, src_text, tgt_text=None):
    messages = [
        {"role": "system", "content": f"Translate the text below to {tgt_name}."},
        {"role": "user", "content": str(src_text)},
    ]
    if tgt_text is not None:
        messages.append({"role": "assistant", "content": str(tgt_text)})
    return messages


def make_preprocess_fn(tokenizer, tgt_name, max_length):
    def preprocess_function(examples):
        all_input_ids, all_labels, all_attention_mask = [], [], []
        for src_text, tgt_text in zip(examples["src"], examples["tgt"]):
            prompt_text = tokenizer.apply_chat_template(
                build_messages(tgt_name, src_text), tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                build_messages(tgt_name, src_text, tgt_text), tokenize=False, add_generation_prompt=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                labels = labels[overflow:]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention_mask.append([1] * len(full_ids))
        return {"input_ids": all_input_ids, "labels": all_labels, "attention_mask": all_attention_mask}
    return preprocess_function


class CausalLMPaddingCollator:
    """Pads pre-tokenized input_ids/labels/attention_mask to the batch max
    length. Labels are already response-only-masked (-100 on prompt
    tokens) by make_preprocess_fn -- this collator only pads."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        import torch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =============================================================
# 3. Generation-based scoring (real model.generate(), BLEU + chrF++)
# =============================================================
def generate_and_score(src_texts, ref_texts, tgt_name, model, tokenizer, device,
                        max_new_tokens, num_beams=4, batch_size=16):
    import torch
    model.eval()
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), batch_size):
            chunk = src_texts[i:i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(tgt_name, t), tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, num_beams=num_beams,
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            all_preds.extend(p.strip() for p in decoded)
            if (i // batch_size) % 20 == 0:
                print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    tokenizer.padding_side = "right"  # restore training-time default

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(all_preds, [ref_texts]).score
    chrfpp = sacrebleu.corpus_chrf(all_preds, [ref_texts], word_order=2).score
    return all_preds, bleu, chrfpp


def append_master_scores(scores_csv, lang, direction, val_loss, val_bleu, val_chrf, test_bleu, test_chrf):
    header = ["language", "direction", "val_loss", "val_bleu_beam4", "val_chrf++_beam4",
              "test_bleu_beam4", "test_chrf++_beam4"]
    row = [lang, direction, f"{val_loss:.4f}", f"{val_bleu:.4f}", f"{val_chrf:.4f}",
           f"{test_bleu:.4f}", f"{test_chrf:.4f}"]
    is_new = not Path(scores_csv).exists()
    with open(scores_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended '{lang}/{direction}' to {scores_csv}")


# =============================================================
# 4. One finetuning job: load model fresh, train, score. Runs inside one
#    GPU-pinned worker process (see the pool orchestration below).
# =============================================================
def run_job(lang, direction, gpu_id, args, lang_cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from datasets import Dataset

    device_tag = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    tag = f"[{lang}/{direction} physical-gpu{gpu_id}]"

    cfg = lang_cfg[lang]
    tgt_col = cfg["tgt_col"]
    suffix = "-smoketest" if args.smoke_test else ""
    output_dir = str(Path(args.output_root) / lang / f"sarvam-{lang.lower()}-{direction}-finetuned{suffix}")
    print(f"\n===== {tag}{' [SMOKE TEST]' if args.smoke_test else ''} =====")
    print(f"  base model: {args.base_model}")
    print(f"  output    : {output_dir}")

    if Path(output_dir, "config.json").exists() and (
        Path(output_dir, "model.safetensors").exists()
        or Path(output_dir, "model.safetensors.index.json").exists()
    ):
        print(f"{tag} already finished -- skipping")
        return

    resume_from_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if resume_from_checkpoint:
        print(f"{tag} resuming training from {resume_from_checkpoint}")

    print(f"{tag} loading data ...")
    (train_hi, train_tgt), (val_hi, val_tgt), (test_hi, test_tgt) = load_splits(cfg)
    if direction == "hi2tgt":
        train_src, train_lbl, val_src, val_lbl, test_src, test_lbl = train_hi, train_tgt, val_hi, val_tgt, test_hi, test_tgt
        tgt_name = tgt_col
    else:  # tgt2hi
        train_src, train_lbl, val_src, val_lbl, test_src, test_lbl = train_tgt, train_hi, val_tgt, val_hi, test_tgt, test_hi
        tgt_name = HIN_NAME

    if args.smoke_test:
        train_src, train_lbl = train_src[:args.smoke_train_rows], train_lbl[:args.smoke_train_rows]
        val_src, val_lbl = val_src[:args.smoke_val_rows], val_lbl[:args.smoke_val_rows]
        test_src, test_lbl = test_src[:args.smoke_val_rows], test_lbl[:args.smoke_val_rows]

    print(f"{tag} train={len(train_src)}  val={len(val_src)}  test={len(test_src)}  (-> {tgt_name})")

    print(f"{tag} loading base model + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device_tag)

    pre_fn = make_preprocess_fn(tokenizer, tgt_name, args.max_length)
    print(f"{tag} tokenizing (response-only loss masking) ...")
    train_ds = Dataset.from_dict({"src": train_src, "tgt": train_lbl}).map(
        pre_fn, batched=True, remove_columns=["src", "tgt"], num_proc=args.num_proc, desc="train")
    val_ds = Dataset.from_dict({"src": val_src, "tgt": val_lbl}).map(
        pre_fn, batched=True, remove_columns=["src", "tgt"], num_proc=args.num_proc, desc="val")

    data_collator = CausalLMPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    # === TRAINING ARGS: full-parameter LLM SFT recipe (AdamW, small LR,
    #     short warmup, few epochs -- same shape as qwen_finetune.py/
    #     llama_finetune.py), plus gradient checkpointing + 8-bit AdamW
    #     (bitsandbytes) since this job is alone on its GPU with no ZeRO
    #     sharding to fall back on -- full-finetuning ~4B params needs the
    #     memory headroom these buy back. ===
    if args.smoke_test:
        max_steps, save_steps, eval_steps, warmup_steps, logging_steps = args.smoke_max_steps, 5, 5, 2, 1
    else:
        max_steps, save_steps, eval_steps, warmup_steps, logging_steps = -1, args.save_steps, args.eval_steps, None, 100

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03 if warmup_steps is None else 0.0,
        warmup_steps=warmup_steps or 0,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        dataloader_num_workers=0,  # this job runs inside a daemon Pool worker -- see finetune_indictrans2.py's identical note
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model, args=training_args, data_collator=data_collator,
        train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
    )

    print(f"{tag} training ...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"{tag} model saved -> {output_dir}")

    val_metrics = trainer.evaluate()
    val_loss = val_metrics.get("eval_loss", float("nan"))
    print(f"{tag} VAL loss = {val_loss:.4f}")

    print(f"{tag} generating on validation set (beam={args.num_beams}) ...")
    _, val_bleu, val_chrf = generate_and_score(
        val_src, val_lbl, tgt_name, model, tokenizer, device_tag, args.max_new_tokens, args.num_beams)
    print(f"{tag} VAL  BLEU={val_bleu:.4f}  chrF++={val_chrf:.4f}")

    print(f"{tag} generating on test set (beam={args.num_beams}) ...")
    test_preds, test_bleu, test_chrf = generate_and_score(
        test_src, test_lbl, tgt_name, model, tokenizer, device_tag, args.max_new_tokens, args.num_beams)
    print(f"{tag} TEST BLEU={test_bleu:.4f}  chrF++={test_chrf:.4f}")

    preds_path = Path(args.output_root) / lang / f"test_predictions_{lang.lower()}_{direction}{suffix}.csv"
    pd.DataFrame({"source": test_src, "reference": test_lbl, "prediction": test_preds}).to_csv(
        preds_path, index=False, encoding="utf-8-sig")
    print(f"{tag} predictions -> {preds_path}")

    scores_csv = Path(args.output_root) / ("all_sarvam_scores_smoketest.csv" if args.smoke_test else "all_sarvam_scores.csv")
    append_master_scores(scores_csv, lang, direction, val_loss, val_bleu, val_chrf, test_bleu, test_chrf)
    print(f"{tag} done -> {output_dir}")


# =============================================================
# 5. GPU-pool orchestration -- identical pattern to
#    indictrans2_finetune/finetune_indictrans2.py.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    # Restrict this worker to exactly one physical GPU -- see the identical,
    # more detailed comment in finetune_indictrans2.py's _pool_init(). Must
    # happen before torch is ever imported in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    lang, direction, args, lang_cfg = job_args
    run_job(lang, direction, _worker_gpu_id, args, lang_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--langs", default="Bhili,Mundari,Gondi,Marathi")
    ap.add_argument("--directions", default="hi2tgt,tgt2hi")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--save_steps", type=int, default=900)
    ap.add_argument("--eval_steps", type=int, default=900)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--smoke_train_rows", type=int, default=300)
    ap.add_argument("--smoke_val_rows", type=int, default=60)
    ap.add_argument("--smoke_max_steps", type=int, default=10)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    args.base_model = cfg["base_model"]
    args.output_root = cfg["output_root"]
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
