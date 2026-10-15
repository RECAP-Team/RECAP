"""
Fine-tune facebook/nllb-200-distilled-600M for English/Hindi <-> Bhili, on
the CORRECTED Bhili data produced by the dataset-cleanup pass (see
correct_checkpoints/dataset_corrected/bhili/{train,val,test}.csv and
Removed_Rows_Report.pdf for what was fixed).

This runs all 4 directions (hi2tgt, tgt2hi, en2tgt, tgt2en) IN PARALLEL on
one 4-GPU PBS job, one direction per GPU -- same GPU-pool pattern as
sarvam_finetune.py/correct_checkpoints/mt5's mt5_finetune.py (a fixed-size
worker pool pulls from a job queue, each worker pinned to exactly one
physical GPU via CUDA_VISIBLE_DEVICES). Each direction is a fully
independent single-GPU fine-tune with no cross-GPU communication, so this
produces IDENTICAL per-direction training/results to running them as 4
separate single-GPU jobs.

Why this needed restructuring (vs. the original hb_pja_nllb_finetune.py,
which processes config.json's direction list sequentially in ONE
process): that script imports torch/transformers at module level and
calls torch.manual_seed()/prints CUDA diagnostics immediately on import.
In a multiprocessing pool, a spawned worker re-imports the module before
the pool's initializer gets to pin CUDA_VISIBLE_DEVICES, so torch would
already see every GPU by the time it's actually used. Here all
CUDA-touching imports (torch, transformers, datasets, numpy, sacrebleu)
are deferred into _lazy_imports(), called as the first thing inside
run_direction() -- i.e. only after _pool_init() has already set
CUDA_VISIBLE_DEVICES for that worker. Same deferred-import pattern used
in correct_checkpoints/mt5/bhili/mt5_finetune.py's equivalent rewrite.

Output filenames are already direction-qualified (nllb-bhili-hi2tgt-
finetuned/, test_predictions_bhili_hi2tgt.csv, training_log_bhili_hi2tgt.*,
etc.) so all 4 workers writing into the same working directory in parallel
do not collide -- the only shared file is all_languages_scores.csv, same
acceptable-risk shared-append pattern the other corrected-data scripts use.

The two functional additions vs. the original script: spBLEU (FLORES-200
tokenizer) alongside the existing BLEU everywhere BLEU is reported, and a
saved training-log JSON/CSV + loss-curve PNG per job. Everything else
(NLLB forced_bos_token_id direction mechanism, new bhb_Deva token
registration via tokenizer.add_special_tokens + resize_token_embeddings,
training recipe) is unchanged.

Config (nllb/bhili/config.json, same directory):
    {
      "language": "bhili",
      "direction": ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"],
      "epochs": 10,
      "train_csv": "...", "val_csv": "...", "test_csv": "..."
    }

Run (on a PBS job with 4 GPUs requested):
    python hb_pja_nllb_finetune.py
"""

import os
import json
import csv
import multiprocessing as mp
from pathlib import Path

# --- silence wandb / tokenizer warnings BEFORE importing torch/transformers ---
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # safe -- doesn't touch CUDA

# =============================================================
# CONFIG
# =============================================================
MODEL_NAME = "/home/scai/msr/aiy257590/flash/GRPO_RESEARCH/Models/nllb-200-distilled-600M"   # same base model as the original pipeline
MAX_LENGTH = 128
SEED       = 42

ENG_CODE = "eng_Latn"
HIN_CODE = "hin_Deva"

# Per-language config (csv_file kept for parity with the original script,
# though unused -- train/val/test paths come from config.json instead).
LANGUAGES = {
    "bhili": {"csv_file": "/home/scai/msr/aiy257590/flash/180k_script_mill_gyi/dataset/Bhi_Hin_Eng.csv",
              "column": "Bhili", "lang_code": "bhb_Deva"},
}

SCORES_CSV = Path("./all_languages_scores.csv")

# CUDA-touching imports/objects, deferred until after CUDA_VISIBLE_DEVICES
# is pinned in each GPU-pool worker -- see _lazy_imports() and the module
# docstring above.
np = torch = sacrebleu = BLEU = None
Dataset = DatasetDict = None
AutoTokenizer = AutoModelForSeq2SeqLM = None
Seq2SeqTrainer = Seq2SeqTrainingArguments = DataCollatorForSeq2Seq = EarlyStoppingCallback = None


def _lazy_imports():
    """Imports everything CUDA-touching, and only on first call inside a
    worker process -- after _pool_init() has already set
    CUDA_VISIBLE_DEVICES for that worker (see module docstring)."""
    global np, torch, sacrebleu, BLEU, Dataset, DatasetDict
    global AutoTokenizer, AutoModelForSeq2SeqLM
    global Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

    if torch is not None:
        return  # already done in this process

    import numpy as _np
    import torch as _torch
    import sacrebleu as _sacrebleu
    from sacrebleu.metrics import BLEU as _BLEU
    from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict
    from transformers import (
        AutoTokenizer as _AutoTokenizer,
        AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM,
        Seq2SeqTrainer as _Seq2SeqTrainer,
        Seq2SeqTrainingArguments as _Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq,
        EarlyStoppingCallback as _EarlyStoppingCallback,
    )

    np, torch, sacrebleu, BLEU = _np, _torch, _sacrebleu, _BLEU
    Dataset, DatasetDict = _Dataset, _DatasetDict
    AutoTokenizer, AutoModelForSeq2SeqLM = _AutoTokenizer, _AutoModelForSeq2SeqLM
    Seq2SeqTrainer = _Seq2SeqTrainer
    Seq2SeqTrainingArguments = _Seq2SeqTrainingArguments
    DataCollatorForSeq2Seq = _DataCollatorForSeq2Seq
    EarlyStoppingCallback = _EarlyStoppingCallback

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 60)


# =============================================================
# 1. LOAD + CLEAN DATA (train/val/test paths given explicitly)
# =============================================================
def _load_and_clean(csv_file: str, tgt_column: str) -> pd.DataFrame:
    print(f"[Data] Loading {csv_file} ...")
    df_raw = pd.read_csv(csv_file)

    needed = {"English", "Hindi", tgt_column}
    assert needed.issubset(df_raw.columns), \
        f"Expected columns {needed} in {csv_file}, got {list(df_raw.columns)}"

    df = df_raw[["English", "Hindi", tgt_column]].dropna()
    for col in ["English", "Hindi", tgt_column]:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["English"] != "") & (df["Hindi"] != "") & (df[tgt_column] != "")].reset_index(drop=True)
    print(f"[Data] Usable parallel rows (all 3 langs present): {len(df)}")
    return df


def load_splits(train_csv: str, val_csv: str, test_csv: str, tgt_column: str):
    train_df = _load_and_clean(train_csv, tgt_column)
    val_df   = _load_and_clean(val_csv,   tgt_column)
    test_df  = _load_and_clean(test_csv,  tgt_column)
    print(f"[Data] Loaded splits | "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return train_df, val_df, test_df


# =============================================================
# 2. TOKENIZER + MODEL  (register new lang token for tribal lang)
# =============================================================
def load_tokenizer_and_model(src_lang_code: str, tribal_code: str):
    print("[Model] Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_lang_code)

    existing_specials = list(tokenizer.additional_special_tokens)
    if tribal_code not in existing_specials:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing_specials + [tribal_code]}
        )

    new_lang_id = tokenizer.convert_tokens_to_ids(tribal_code)
    if hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id[tribal_code] = new_lang_id
    if hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code[new_lang_id] = tribal_code
    if hasattr(tokenizer, "fairseq_tokens_to_ids"):
        tokenizer.fairseq_tokens_to_ids[tribal_code] = new_lang_id
    if hasattr(tokenizer, "fairseq_ids_to_tokens"):
        tokenizer.fairseq_ids_to_tokens[new_lang_id] = tribal_code

    print(f"[Tokenizer] {tribal_code} token id = {new_lang_id}, vocab size = {len(tokenizer)}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    print("Inside load_tokenizer_and_model()")
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Using device: {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    model.to(device)
    return tokenizer, model, device


# =============================================================
# 3. PREPROCESSING
# =============================================================
def make_preprocess_fn(tokenizer, src_col, tgt_col, src_lang_code, tgt_lang_code):
    def preprocess_function(examples):
        inputs  = examples[src_col]
        targets = examples[tgt_col]
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        return model_inputs
    return preprocess_function


# =============================================================
# 4. METRICS  (BLEU + spBLEU + chrF + chrF++)
# =============================================================
def make_compute_metrics(tokenizer):
    def _postprocess(preds, labels):
        return [p.strip() for p in preds], [l.strip() for l in labels]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels      = np.where(labels      != -100, labels,      tokenizer.pad_token_id)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)
        decoded_preds, decoded_labels = _postprocess(decoded_preds, decoded_labels)

        bleu   = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        # spBLEU: same BLEU computation with the FLORES-200 SentencePiece
        # tokenizer instead of the default 13a tokenizer -- the standard
        # choice for Indic-script text used throughout this project's other
        # spBLEU scripts.
        spbleu = BLEU(tokenize="flores200").corpus_score(decoded_preds, [decoded_labels]).score
        chrf   = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        # "chrf" key (no ++) is used for metric_for_best_model per the training args
        return {"bleu": bleu, "spbleu": spbleu, "chrf": chrf, "chrf++": chrfpp}
    return compute_metrics


# =============================================================
# 5. APPEND TO MASTER SCORES CSV
# =============================================================
def append_master_scores(language, direction, val_bleu, val_spbleu, val_chrf,
                         test_bleu, test_spbleu, test_chrf,
                         manual_bleu, manual_spbleu, manual_chrf):
    header = ["language", "direction",
              "val_bleu", "val_spbleu", "val_chrf++",
              "test_bleu", "test_spbleu", "test_chrf++",
              "test_bleu_beam5", "test_spbleu_beam5", "test_chrf++_beam5"]
    row = [language, direction,
           f"{val_bleu:.4f}",   f"{val_spbleu:.4f}",   f"{val_chrf:.4f}",
           f"{test_bleu:.4f}",  f"{test_spbleu:.4f}",  f"{test_chrf:.4f}",
           f"{manual_bleu:.4f}", f"{manual_spbleu:.4f}", f"{manual_chrf:.4f}"]

    is_new = not SCORES_CSV.exists()
    with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended row '{language}/{direction}' to {SCORES_CSV}")


def save_training_logs_and_plot(log_history, language, direction):
    """Saves the HF Trainer's per-step log history (train loss + periodic
    eval loss/bleu/chrf) as JSON and CSV, and renders a loss-curve PNG."""
    tag = f"{language}_{direction}"

    log_json_path = Path(f"training_log_{tag}.json")
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    train_rows = [r for r in log_history if "loss" in r and "eval_loss" not in r]
    eval_rows = [r for r in log_history if "eval_loss" in r]

    log_csv_path = Path(f"training_log_{tag}.csv")
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "eval_loss", "eval_chrf", "learning_rate"])
        steps = sorted({r.get("step") for r in train_rows + eval_rows if r.get("step") is not None})
        train_by_step = {r["step"]: r for r in train_rows}
        eval_by_step = {r["step"]: r for r in eval_rows}
        for step in steps:
            tr = train_by_step.get(step, {})
            ev = eval_by_step.get(step, {})
            w.writerow([step, tr.get("loss", ""), ev.get("eval_loss", ""),
                        ev.get("eval_chrf", ""), tr.get("learning_rate", "")])

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
        ax.set_title(f"NLLB -- {language}/{direction} training loss")
        ax.legend()
        fig.tight_layout()
        plot_path = Path(f"training_loss_plot_{tag}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[Logs] training log/plot -> {log_json_path.name}, {log_csv_path.name}, {plot_path.name}")
    except Exception as e:
        print(f"[Logs] [warn] could not render loss plot ({e}); log JSON/CSV still saved")


# =============================================================
# 6. One (language, direction) job -- runs inside one GPU-pinned worker
# =============================================================
def run_direction(language, direction, train_csv, val_csv, test_csv, epochs):
    _lazy_imports()

    assert language in LANGUAGES, \
        f"Unknown language '{language}', choices: {list(LANGUAGES.keys())}"
    assert direction in ["en2tgt", "hi2tgt", "tgt2en", "tgt2hi"], \
        f"Unknown direction '{direction}', choices: en2tgt, hi2tgt, tgt2en, tgt2hi"

    cfg       = LANGUAGES[language]
    tgt_column  = cfg["column"]
    tribal_code = cfg["lang_code"]

    # Resolve direction -> (src_col, tgt_col, src_code, tgt_code)
    if direction == "en2tgt":
        src_col, tgt_col, src_code, tgt_code = "English", tgt_column, ENG_CODE, tribal_code
    elif direction == "hi2tgt":
        src_col, tgt_col, src_code, tgt_code = "Hindi", tgt_column, HIN_CODE, tribal_code
    elif direction == "tgt2en":
        src_col, tgt_col, src_code, tgt_code = tgt_column, "English", tribal_code, ENG_CODE
    elif direction == "tgt2hi":
        src_col, tgt_col, src_code, tgt_code = tgt_column, "Hindi", tribal_code, HIN_CODE

    output_dir = f"./nllb-{language}-{direction}-finetuned"
    print(f"\n========== Language: {language} | Direction: {direction} ==========")
    print(f"  {src_col} [{src_code}]  ->  {tgt_col} [{tgt_code}]")
    print(f"[Output] {output_dir}")

    # ---- Data ----
    train_df, val_df, test_df = load_splits(train_csv, val_csv, test_csv, tgt_column)
    train_set = Dataset.from_pandas(train_df, preserve_index=False)
    val_set   = Dataset.from_pandas(val_df,   preserve_index=False)
    test_set  = Dataset.from_pandas(test_df,  preserve_index=False)
    dataset_dict = DatasetDict({"train": train_set, "validation": val_set, "test": test_set})
    print(f"[Data] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # ---- Tokenizer + model ----
    tokenizer, model, device = load_tokenizer_and_model(src_code, tribal_code)

    # ---- Tokenize ----
    pre_fn = make_preprocess_fn(tokenizer, src_col, tgt_col, src_code, tgt_code)
    candidate_cols = ["English", "Hindi", tgt_column, "Unique_ID"]
    cols_to_remove = [c for c in candidate_cols if c in train_set.column_names]

    print("[Tokenize] Mapping splits ...")
    tokenized_train = dataset_dict["train"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="train")
    tokenized_val = dataset_dict["validation"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="val")
    tokenized_test = dataset_dict["test"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="test")

    # ---- Trainer ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
    model.config.forced_bos_token_id = forced_bos_id

    # Use bf16 if GPU supports it; otherwise no mixed precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[Train] bf16={use_bf16}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=150,
        eval_strategy="steps",
        eval_steps=900,
        save_strategy="steps",
        save_steps=900,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        generation_max_length=128,
        bf16=use_bf16,
        fp16=False,
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1000,
        dataloader_num_workers=0,  # this job runs inside a daemon Pool worker, same reason as mt5_finetune.py's equivalent note
        report_to="none",
        predict_with_generate=True,
        eval_accumulation_steps=4,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print(f"[Train] epochs={epochs}  steps/epoch~{len(tokenized_train)//(16*2)}")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Save] Best model written to {output_dir}")

    save_training_logs_and_plot(trainer.state.log_history, language, direction)

    # ---- Final eval ----
    print("\n=== Validation set (Trainer.evaluate, beam search) ===")
    val_results = trainer.evaluate(metric_key_prefix="val")
    val_bleu   = val_results.get("val_bleu",   float("nan"))
    val_spbleu = val_results.get("val_spbleu", float("nan"))
    val_chrf   = val_results.get("val_chrf++", float("nan"))
    print(f"VAL  BLEU   = {val_bleu:.4f}")
    print(f"VAL  spBLEU = {val_spbleu:.4f}")
    print(f"VAL  chrF++ = {val_chrf:.4f}")

    print("\n=== Test set (Trainer.evaluate, beam search) ===")
    test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    test_bleu   = test_results.get("test_bleu",   float("nan"))
    test_spbleu = test_results.get("test_spbleu", float("nan"))
    test_chrf   = test_results.get("test_chrf++", float("nan"))
    print(f"TEST BLEU   = {test_bleu:.4f}")
    print(f"TEST spBLEU = {test_spbleu:.4f}")
    print(f"TEST chrF++ = {test_chrf:.4f}")

    # ---- Manual sentence-level inference (beam=5) for inspection ----
    print("\n=== Manual inference on test set (beam=5) ===")
    model.eval()
    test_records = list(dataset_dict["test"])
    src_texts = [r[src_col] for r in test_records]
    ref_texts = [r[tgt_col] for r in test_records]

    all_preds = []
    batch = 16
    tokenizer.src_lang = src_code
    with torch.no_grad():
        for i in range(0, len(src_texts), batch):
            chunk = src_texts[i:i + batch]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
            ).to(device)
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                max_length=MAX_LENGTH,
                num_beams=5,
            )
            all_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (i // batch) % 20 == 0:
                print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    preds_path = f"test_predictions_{language}_{direction}.csv"
    pd.DataFrame({
        f"source_{src_col.lower()}":     src_texts,
        f"reference_{tgt_col.lower()}":  ref_texts,
        f"prediction_{tgt_col.lower()}": all_preds,
    }).to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"[Save] Predictions: {preds_path}")

    final_bleu   = sacrebleu.corpus_bleu(all_preds, [ref_texts]).score
    final_spbleu = BLEU(tokenize="flores200").corpus_score(all_preds, [ref_texts]).score
    final_chrfpp = sacrebleu.corpus_chrf(all_preds, [ref_texts], word_order=2).score

    print(f"\n=== FINAL TEST SCORES [{language}/{direction}] (manual beam=5) ===")
    print(f"BLEU   : {final_bleu:.4f}")
    print(f"spBLEU : {final_spbleu:.4f}")
    print(f"chrF++ : {final_chrfpp:.4f}")

    with open(f"final_scores_{language}_{direction}.txt", "w", encoding="utf-8") as f:
        f.write(f"Language: {language}  Direction: {direction}  ({src_col} -> {tgt_col})\n")
        f.write(f"VAL  BLEU   : {val_bleu:.4f}\n")
        f.write(f"VAL  spBLEU : {val_spbleu:.4f}\n")
        f.write(f"VAL  chrF++ : {val_chrf:.4f}\n")
        f.write(f"TEST BLEU   : {test_bleu:.4f}\n")
        f.write(f"TEST spBLEU : {test_spbleu:.4f}\n")
        f.write(f"TEST chrF++ : {test_chrf:.4f}\n")
        f.write(f"TEST BLEU (beam=5 manual)   : {final_bleu:.4f}\n")
        f.write(f"TEST spBLEU (beam=5 manual) : {final_spbleu:.4f}\n")
        f.write(f"TEST chrF++ (beam=5 manual) : {final_chrfpp:.4f}\n")

    append_master_scores(language, direction, val_bleu, val_spbleu, val_chrf,
                         test_bleu, test_spbleu, test_chrf,
                         final_bleu, final_spbleu, final_chrfpp)
    print(f"\nDone with {language}/{direction}. Master scores: {SCORES_CSV}")


# =============================================================
# 7. GPU-pool orchestration -- identical pattern to
#    correct_checkpoints/mt5/bhili/mt5_finetune.py / sarvam_finetune.py.
# =============================================================
_worker_gpu_id = None


def _pool_init(gpu_queue):
    global _worker_gpu_id
    _worker_gpu_id = gpu_queue.get()
    # Restrict this worker to exactly one physical GPU -- must happen
    # before torch is ever imported in this process, which is why all
    # torch/transformers imports are deferred into _lazy_imports() instead
    # of sitting at module level.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_worker_gpu_id)


def _pool_run(job_args):
    language, direction, train_csv, val_csv, test_csv, epochs = job_args
    run_direction(language, direction, train_csv, val_csv, test_csv, epochs)


def main():
    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    language   = run_cfg["language"]
    directions = run_cfg["direction"]
    train_csv  = run_cfg["train_csv"]
    val_csv    = run_cfg["val_csv"]
    test_csv   = run_cfg["test_csv"]
    epochs     = run_cfg["epochs"]

    # "direction" can be a single string or a list of the four directions
    if isinstance(directions, str):
        directions = [directions]

    import torch  # just to query device count in the launcher process -- harmless, this process doesn't train
    n_gpus = torch.cuda.device_count()
    print(f"[gpu] {n_gpus} visible")
    if n_gpus == 0:
        raise SystemExit("No GPU visible -- request GPUs on the job scheduler before running this.")

    pool_size = min(n_gpus, len(directions))
    print(f"[pool] {pool_size} worker(s), {len(directions)} job(s) queued: {directions}")

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu_id in range(pool_size):
        gpu_queue.put(gpu_id)

    job_args = [(language, direction, train_csv, val_csv, test_csv, epochs) for direction in directions]
    with ctx.Pool(processes=pool_size, initializer=_pool_init, initargs=(gpu_queue,)) as pool:
        pool.map(_pool_run, job_args)

    print("\nAll jobs done.")


if __name__ == "__main__":
    main()
