"""
Compare the BASE (not fine-tuned) IndicTrans2 and Sarvam-Translate models on
Hindi -> English translation, for the 100 worst-English rows flagged in
eng_test.csv (see find_worst_english.py / the badness_score columns that
were in that file before it was trimmed to just Hindi/English).

Neither model is fine-tuned here -- this is purely zero-shot inference with
each model's own pretrained checkpoint, to compare which one produces better
Hindi->English translations before committing to re-generating the English
column for the full ~228K-row corpus with the winner.

Runs both models sequentially on a single GPU (loads IndicTrans2, translates,
frees it, then loads Sarvam-Translate, translates) -- intended for a single
interactive PBS job, not a multi-GPU batch job.

Input:  eng_test.csv (same directory) -- columns: Hindi, English
Output: eng_test_predictions.csv (same directory) -- columns:
    Hindi, English (original garbled version, for reference),
    Predicted_English_IndicTrans2, Predicted_English_Sarvam

Run (1 GPU, interactive job):
    python infer_hi2en_base_models.py
"""

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
INPUT_CSV = str(HERE / "eng_test.csv")
OUTPUT_CSV = str(HERE / "eng_test_predictions.csv")

REPO_ROOT = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP"
INDICTRANS2_BASE_MODEL = f"{REPO_ROOT}/Models/IndicTrans2-indic-en-1B"
SARVAM_BASE_MODEL = f"{REPO_ROOT}/Models/Sarvam-Translate"

HIN_CODE = "hin_Deva"
ENG_CODE = "eng_Latn"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_BEAMS = 4


def translate_with_indictrans2(src_texts, device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    print(f"[IndicTrans2] loading base model: {INDICTRANS2_BASE_MODEL}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS2_BASE_MODEL, trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS2_BASE_MODEL, trust_remote_code=True)
    model.eval()
    processor = IndicProcessor(inference=True)

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = src_texts[i:i + BATCH_SIZE]
            batch = processor.preprocess_batch(chunk, src_lang=HIN_CODE, tgt_lang=ENG_CODE)
            inputs = tokenizer(
                batch, truncation=True, padding="longest",
                return_tensors="pt", max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds.extend(processor.postprocess_batch(decoded, lang=ENG_CODE))
            print(f"[IndicTrans2] decoded {i + len(chunk)}/{len(src_texts)}")

    del model
    torch.cuda.empty_cache()
    return preds


def build_sarvam_messages(src_text):
    return [
        {"role": "system", "content": "Translate the text below to English."},
        {"role": "user", "content": str(src_text)},
    ]


def translate_with_sarvam(src_texts, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Sarvam] loading base model: {SARVAM_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        SARVAM_BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = src_texts[i:i + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    build_sarvam_messages(t), tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc, max_new_tokens=MAX_LENGTH, num_beams=NUM_BEAMS,
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            preds.extend(p.strip() for p in decoded)
            print(f"[Sarvam] decoded {i + len(chunk)}/{len(src_texts)}")

    del model
    torch.cuda.empty_cache()
    return preds


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, na_values=[])
    print(f"[read] {INPUT_CSV} ({len(df)} rows)")
    for col in ("Hindi", "English"):
        assert col in df.columns, f"expected column '{col}' in {INPUT_CSV}, got {list(df.columns)}"

    src_texts = df["Hindi"].astype(str).str.strip().tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  (cuda available: {torch.cuda.is_available()})")

    df["Predicted_English_IndicTrans2"] = translate_with_indictrans2(src_texts, device)
    df["Predicted_English_Sarvam"] = translate_with_sarvam(src_texts, device)

    out_cols = ["Hindi", "English", "Predicted_English_IndicTrans2", "Predicted_English_Sarvam"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[done] {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
