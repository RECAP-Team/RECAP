"""
One-off diagnostic (NOT part of the pipeline) for the tgt2hi NaN-loss issue:
tokenizer-only, no GPU/model weights needed, runs in seconds. Checks whether
Marathi->Hindi examples are losing their response tokens to left-truncation
at MAX_LENGTH=256 -- reusing llama_finetune.py's own build_prompt_text/
build_full_text so the check matches the real preprocessing exactly.

Run (CPU is fine, on a login node or inside your GPU allocation):
    python diagnose_tgt2hi.py
"""
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_finetune import (
    MODEL_NAME, MAX_LENGTH, BOS_TOKEN, EOT_TOKEN,
    build_prompt_text, build_full_text, HIN_NAME,
)

import json
cfg = json.load(open(Path(__file__).resolve().parent / "config.json"))
train_csv = cfg["train_csv"]
N = 2000

# Check for a tokenizer/embedding-table size mismatch -- some repackaged
# Llama-3.1 checkpoints ship an embedding matrix smaller than the
# tokenizer's actual vocab size, so an out-of-range token id silently
# indexes garbage/NaN on GPU (undefined behaviour) instead of crashing.
# AutoConfig only reads config.json, no weights loaded -- cheap.
model_config = AutoConfig.from_pretrained(MODEL_NAME)
print(f"model config.vocab_size = {model_config.vocab_size}")

df = pd.read_csv(train_csv)
df = df[["Hindi", "Marathi"]].dropna()
for c in ["Hindi", "Marathi"]:
    df[c] = df[c].astype(str).str.strip()
df = df[(df["Hindi"] != "") & (df["Marathi"] != "")].reset_index(drop=True)
df = df.head(N)

print(f"Loading tokenizer from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"len(tokenizer) = {len(tokenizer)}")
if len(tokenizer) > model_config.vocab_size:
    print(f"!!! MISMATCH: tokenizer has {len(tokenizer) - model_config.vocab_size} "
          f"more tokens than the model's embedding table has rows -- any token id "
          f">= {model_config.vocab_size} indexes out of bounds.")

# tgt2hi: src=Marathi, tgt=Hindi
src_name, tgt_name = "Marathi", HIN_NAME

n_all_masked = 0
n_over_length = 0
n_oob_token = 0
max_token_id_seen = -1
prompt_lens, full_lens, resp_lens = [], [], []

for i, row in df.iterrows():
    src_text, tgt_text = row["Marathi"], row["Hindi"]
    prompt_text = build_prompt_text(src_name, tgt_name, src_text)
    full_text = build_full_text(src_name, tgt_name, src_text, tgt_text)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    row_max_id = max(full_ids) if full_ids else -1
    max_token_id_seen = max(max_token_id_seen, row_max_id)
    if row_max_id >= model_config.vocab_size:
        n_oob_token += 1
        if n_oob_token <= 3:
            bad_ids = [t for t in full_ids if t >= model_config.vocab_size]
            print(f"\n[OUT-OF-BOUNDS token id, row {i}] ids >= vocab_size: {bad_ids}")
            print(f"  decoded: {tokenizer.decode(bad_ids)!r}")
            print(f"  Marathi src: {src_text[:200]!r}")

    prompt_lens.append(len(prompt_ids))
    full_lens.append(len(full_ids))
    resp_lens.append(len(full_ids) - len(prompt_ids))

    if len(full_ids) > MAX_LENGTH:
        overflow = len(full_ids) - MAX_LENGTH
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]
        n_over_length += 1

    if all(l == -100 for l in labels):
        n_all_masked += 1
        if n_all_masked <= 3:
            print(f"\n[ALL-MASKED example, row {i}]")
            print(f"  Marathi src (len={len(src_text)} chars): {src_text[:200]!r}")
            print(f"  Hindi tgt  (len={len(tgt_text)} chars): {tgt_text[:200]!r}")
            print(f"  prompt_ids={len(prompt_ids)}  full_ids(pre-trunc)={len(prompt_ids)+len(full_ids) if False else 'see above'}")

print(f"\n=== Checked {len(df)} tgt2hi (Marathi->Hindi) rows ===")
print(f"prompt_len   min/mean/max: {min(prompt_lens)}/{sum(prompt_lens)/len(prompt_lens):.1f}/{max(prompt_lens)}")
print(f"full_len     min/mean/max: {min(full_lens)}/{sum(full_lens)/len(full_lens):.1f}/{max(full_lens)}")
print(f"response_len min/mean/max: {min(resp_lens)}/{sum(resp_lens)/len(resp_lens):.1f}/{max(resp_lens)}")
print(f"rows truncated (full_ids > {MAX_LENGTH}): {n_over_length}")
print(f"rows with ALL labels masked (-100) after truncation: {n_all_masked}")
print(f"max token id seen: {max_token_id_seen}  (model vocab_size={model_config.vocab_size})")
print(f"rows containing an out-of-bounds token id: {n_oob_token}")
