"""
One-off diagnostic (NOT part of the pipeline) for the tgt2hi NaN-loss issue,
step 2: does the FRESH, UNTRAINED base model already produce NaN loss/logits
on real tgt2hi batches? Single GPU, plain forward pass, no DeepSpeed, no
training -- isolates "bad batch construction" from "training-dynamics
divergence". Reuses llama_finetune.py's own preprocessing/collator so the
batches are byte-for-byte what real training would feed the model.

Run (needs exactly 1 GPU, a few minutes for model load):
    CUDA_VISIBLE_DEVICES=0 python diagnose_forward_pass.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_finetune import (
    MODEL_NAME, MAX_LENGTH, HIN_NAME,
    build_prompt_text, build_full_text, CausalLMPaddingCollator,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

cfg = json.load(open(Path(__file__).resolve().parent / "config.json"))
train_csv = cfg["train_csv"]
N_BATCHES = 5
BATCH_SIZE = 4

df = pd.read_csv(train_csv)
df = df[["Hindi", "Marathi"]].dropna()
for c in ["Hindi", "Marathi"]:
    df[c] = df[c].astype(str).str.strip()
df = df[(df["Hindi"] != "") & (df["Marathi"] != "")].reset_index(drop=True)
df = df.head(N_BATCHES * BATCH_SIZE)

print(f"Loading tokenizer + model from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
device = torch.device("cuda:0")
model.to(device)
model.eval()

src_name, tgt_name = "Marathi", HIN_NAME  # tgt2hi direction
collator = CausalLMPaddingCollator(pad_token_id=tokenizer.pad_token_id)

examples = []
for _, row in df.iterrows():
    src_text, tgt_text = row["Marathi"], row["Hindi"]
    prompt_text = build_prompt_text(src_name, tgt_name, src_text)
    full_text = build_full_text(src_name, tgt_name, src_text, tgt_text)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if len(full_ids) > MAX_LENGTH:
        overflow = len(full_ids) - MAX_LENGTH
        full_ids = full_ids[overflow:]
        labels = labels[overflow:]
    examples.append({
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "_src": src_text, "_tgt": tgt_text,
    })

any_nan = False
with torch.no_grad():
    for b in range(N_BATCHES):
        batch = examples[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        enc = collator(batch)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        loss = out.loss
        logits_nan = torch.isnan(out.logits).any().item()
        logits_inf = torch.isinf(out.logits).any().item()
        loss_bad = torch.isnan(loss).item() or torch.isinf(loss).item()
        print(f"[batch {b}] loss={loss.item():.4f}  logits_nan={logits_nan}  "
              f"logits_inf={logits_inf}")
        if loss_bad or logits_nan or logits_inf:
            any_nan = True
            for ex in batch:
                print(f"  src={ex['_src'][:150]!r}")
                print(f"  tgt={ex['_tgt'][:150]!r}")

print(f"\n=== {'NaN/Inf FOUND on fresh base model' if any_nan else 'No NaN/Inf on fresh base model -- batches are clean'} ===")
