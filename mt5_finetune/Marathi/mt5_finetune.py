"""
Fine-tune mT5-base for Hindi <-> Marathi (both directions).

Same recipe as mt5_finetune/<tribal>/mt5_finetune.py (Adafactor, constant LR
with warmup, bf16, beam-search eval), adapted for the IIT-B Hindi-Marathi
data, which differs from the tribal setup in three ways:

  1. The CSV has just two columns -- 'hindi' and 'marathi' (no English).
  2. There is a single train.csv; val is carved out of it in-script at a
     97.5 / 2.5 ratio (shuffled with SEED). test.csv is separate and passed
     in via config.json (optional -- omit it to skip test evaluation).
  3. Only two directions: hi2mr (Hindi->Marathi) and mr2hi (Marathi->Hindi).

mT5 has no NLLB-style language-code tokens -- direction is a natural-language
instruction prefix on the source text, e.g. "translate Hindi to Marathi: ".

config.json (same directory as this script):
    {
      "direction": ["hi2mr", "mr2hi"],
      "epochs": 10,
      "train_csv": "/abs/path/Marathi/train.csv",
      "test_csv":  "/abs/path/Marathi/test.csv",
      "val_ratio": 0.025,
      "hi_col": "hindi",
      "mr_col": "marathi"
    }
"""

import os
import json
import csv
from pathlib import Path

# --- silence wandb / tokenizer warnings BEFORE importing torch/transformers ---
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd
import torch

print("=" * 60)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import sacrebleu

# =============================================================
# CONFIG
# =============================================================
MODEL_NAME = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/Models/mt5-base"
MAX_LENGTH = 128
SEED       = 42

HIN_NAME = "Hindi"
MR_NAME  = "Marathi"
LANGUAGE = "marathi"

DIRECTIONS = ["hi2mr", "mr2hi"]

# Master scores file -- one shared CSV across both directions
SCORES_CSV = Path("./all_marathi_scores.csv")

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================
# 1. LOAD + CLEAN DATA
# =============================================================
def _load_and_clean(csv_file: str, hi_col: str, mr_col: str) -> pd.DataFrame:
    print(f"[Data] Loading {csv_file} ...")
    df_raw = pd.read_csv(csv_file)

    needed = {hi_col, mr_col}
    assert needed.issubset(df_raw.columns), \
        f"Expected columns {needed} in {csv_file}, got {list(df_raw.columns)}"

    df = df_raw[[hi_col, mr_col]].dropna()
    for col in (hi_col, mr_col):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df[hi_col] != "") & (df[mr_col] != "")].reset_index(drop=True)
    print(f"[Data] Usable parallel rows: {len(df)}")
    return df


def load_splits(train_csv, test_csv, val_ratio, hi_col, mr_col):
    """train.csv -> shuffled 97.5 / 2.5 train/val split. test.csv is
    separate and optional (None -> no test set)."""
    full = _load_and_clean(train_csv, hi_col, mr_col)
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(round(len(full) * val_ratio))
    val_df   = full.iloc[:n_val].reset_index(drop=True)
    train_df = full.iloc[n_val:].reset_index(drop=True)

    test_df = _load_and_clean(test_csv, hi_col, mr_col) if test_csv else None
    print(f"[Data] Loaded splits | train={len(train_df)} val={len(val_df)} "
          f"test={len(test_df) if test_df is not None else 0} "
          f"(val_ratio={val_ratio})")
    return train_df, val_df, test_df


# =============================================================
# 2. TOKENIZER + MODEL
# =============================================================
def load_tokenizer_and_model():
    print("[Model] Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model] Using device: {device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    model.to(device)
    return tokenizer, model, device


# =============================================================
# 3. PREPROCESSING  (T5-style instruction prefix encodes the direction)
# =============================================================
def make_preprocess_fn(tokenizer, src_col, tgt_col, prefix):
    def preprocess_function(examples):
        inputs  = [prefix + str(t) for t in examples[src_col]]
        targets = examples[tgt_col]

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
# 4. METRICS  (BLEU + chrF + chrF++)
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
        chrf   = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        # "chrf" key (no ++) is used for metric_for_best_model per the training args
        return {"bleu": bleu, "chrf": chrf, "chrf++": chrfpp}
    return compute_metrics


# =============================================================
# 5. APPEND TO MASTER SCORES CSV
# =============================================================
def append_master_scores(language, direction, val_bleu, val_chrf, test_bleu, test_chrf,
                         manual_bleu, manual_chrf):
    header = ["language", "direction",
              "val_bleu", "val_chrf++", "test_bleu", "test_chrf++",
              "test_bleu_beam5", "test_chrf++_beam5"]
    row = [language, direction,
           f"{val_bleu:.4f}",   f"{val_chrf:.4f}",
           f"{test_bleu:.4f}",  f"{test_chrf:.4f}",
           f"{manual_bleu:.4f}", f"{manual_chrf:.4f}"]

    is_new = not SCORES_CSV.exists()
    with open(SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)
    print(f"[Scores] Appended row '{language}/{direction}' to {SCORES_CSV}")


# =============================================================
# 6. MAIN
# =============================================================
def run_direction(direction, train_df, val_df, test_df, epochs, hi_col, mr_col):
    assert direction in DIRECTIONS, \
        f"Unknown direction '{direction}', choices: {DIRECTIONS}"

    # Resolve direction -> (src_col, tgt_col, src_name, tgt_name)
    if direction == "hi2mr":
        src_col, tgt_col, src_name, tgt_name = hi_col, mr_col, HIN_NAME, MR_NAME
    else:  # mr2hi
        src_col, tgt_col, src_name, tgt_name = mr_col, hi_col, MR_NAME, HIN_NAME

    prefix = f"translate {src_name} to {tgt_name}: "

    output_dir = f"./mt5-{LANGUAGE}-{direction}-finetuned"
    print(f"\n========== {LANGUAGE} | Direction: {direction} ==========")
    print(f"  {src_col} -> {tgt_col}  |  prefix: '{prefix}'")
    print(f"[Output] {output_dir}")

    # ---- Data ----
    train_set = Dataset.from_pandas(train_df, preserve_index=False)
    val_set   = Dataset.from_pandas(val_df,   preserve_index=False)
    splits = {"train": train_set, "validation": val_set}
    if test_df is not None:
        splits["test"] = Dataset.from_pandas(test_df, preserve_index=False)
    dataset_dict = DatasetDict(splits)
    print(f"[Data] train={len(train_set)}  val={len(val_set)}  "
          f"test={len(dataset_dict['test']) if 'test' in dataset_dict else 0}")

    # ---- Tokenizer + model ----
    tokenizer, model, device = load_tokenizer_and_model()

    # ---- Tokenize ----
    pre_fn = make_preprocess_fn(tokenizer, src_col, tgt_col, prefix)
    cols_to_remove = [c for c in (hi_col, mr_col) if c in train_set.column_names]

    print("[Tokenize] Mapping splits ...")
    tokenized_train = dataset_dict["train"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="train")
    tokenized_val = dataset_dict["validation"].map(
        pre_fn, batched=True, remove_columns=cols_to_remove, desc="val")
    tokenized_test = (
        dataset_dict["test"].map(pre_fn, batched=True, remove_columns=cols_to_remove, desc="test")
        if "test" in dataset_dict else None
    )

    # ---- Trainer ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # mT5/T5 are prone to NaN loss under fp16 -- bf16 avoids that, so fp16 is
    # intentionally left off below.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[Train] bf16={use_bf16}")

    # === TRAINING ARGS (standard T5/mT5 fine-tuning recipe: Adafactor,
    #     constant LR with warmup) -- identical to the tribal mt5 script ===
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        optim="adafactor",
        learning_rate=1e-3,
        lr_scheduler_type="constant_with_warmup",
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
        dataloader_num_workers=2,
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

    print(f"[Train] epochs={epochs}  steps/epoch~{len(tokenized_train)//(32*2)}")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Save] Best model written to {output_dir}")

    # ---- Final eval ----
    print("\n=== Validation set (Trainer.evaluate, beam search) ===")
    val_results = trainer.evaluate(metric_key_prefix="val")
    val_bleu = val_results.get("val_bleu",   float("nan"))
    val_chrf = val_results.get("val_chrf++", float("nan"))
    print(f"VAL  BLEU   = {val_bleu:.4f}")
    print(f"VAL  chrF++ = {val_chrf:.4f}")

    test_bleu = test_chrf = float("nan")
    final_bleu = final_chrfpp = float("nan")
    if tokenized_test is not None:
        print("\n=== Test set (Trainer.evaluate, beam search) ===")
        test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
        test_bleu = test_results.get("test_bleu",   float("nan"))
        test_chrf = test_results.get("test_chrf++", float("nan"))
        print(f"TEST BLEU   = {test_bleu:.4f}")
        print(f"TEST chrF++ = {test_chrf:.4f}")

        # ---- Manual sentence-level inference (beam=5) for inspection ----
        print("\n=== Manual inference on test set (beam=5) ===")
        model.eval()
        test_records = list(dataset_dict["test"])
        src_texts = [r[src_col] for r in test_records]
        ref_texts = [r[tgt_col] for r in test_records]

        all_preds = []
        batch = 16
        with torch.no_grad():
            for i in range(0, len(src_texts), batch):
                chunk = [prefix + str(t) for t in src_texts[i:i + batch]]
                enc = tokenizer(
                    chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
                ).to(device)
                out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=5)
                all_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
                if (i // batch) % 20 == 0:
                    print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

        preds_path = f"test_predictions_{LANGUAGE}_{direction}.csv"
        pd.DataFrame({
            f"source_{src_col.lower()}":     src_texts,
            f"reference_{tgt_col.lower()}":  ref_texts,
            f"prediction_{tgt_col.lower()}": all_preds,
        }).to_csv(preds_path, index=False, encoding="utf-8-sig")
        print(f"[Save] Predictions: {preds_path}")

        final_bleu   = sacrebleu.corpus_bleu(all_preds, [ref_texts]).score
        final_chrfpp = sacrebleu.corpus_chrf(all_preds, [ref_texts], word_order=2).score
        print(f"\n=== FINAL TEST SCORES [{LANGUAGE}/{direction}] (manual beam=5) ===")
        print(f"BLEU   : {final_bleu:.4f}")
        print(f"chrF++ : {final_chrfpp:.4f}")

    with open(f"final_scores_{LANGUAGE}_{direction}.txt", "w", encoding="utf-8") as f:
        f.write(f"Language: {LANGUAGE}  Direction: {direction}  ({src_col} -> {tgt_col})\n")
        f.write(f"VAL  BLEU   : {val_bleu:.4f}\n")
        f.write(f"VAL  chrF++ : {val_chrf:.4f}\n")
        f.write(f"TEST BLEU   : {test_bleu:.4f}\n")
        f.write(f"TEST chrF++ : {test_chrf:.4f}\n")
        f.write(f"TEST BLEU (beam=5 manual)   : {final_bleu:.4f}\n")
        f.write(f"TEST chrF++ (beam=5 manual) : {final_chrfpp:.4f}\n")

    append_master_scores(LANGUAGE, direction, val_bleu, val_chrf, test_bleu, test_chrf,
                         final_bleu, final_chrfpp)
    print(f"\nDone with {LANGUAGE}/{direction}. Master scores: {SCORES_CSV}")


def main():
    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    directions = run_cfg.get("direction", DIRECTIONS)
    if isinstance(directions, str):
        directions = [directions]
    train_csv = run_cfg["train_csv"]
    test_csv  = run_cfg.get("test_csv")  # optional
    epochs    = run_cfg["epochs"]
    val_ratio = run_cfg.get("val_ratio", 0.025)
    hi_col    = run_cfg.get("hi_col", "hindi")
    mr_col    = run_cfg.get("mr_col", "marathi")

    # Split once, reuse for every direction (identical rows across directions).
    train_df, val_df, test_df = load_splits(train_csv, test_csv, val_ratio, hi_col, mr_col)

    for direction in directions:
        run_direction(direction, train_df, val_df, test_df, epochs, hi_col, mr_col)


if __name__ == "__main__":
    main()
