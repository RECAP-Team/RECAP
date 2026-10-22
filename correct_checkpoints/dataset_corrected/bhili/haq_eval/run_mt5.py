"""
Inference-only: score the corrected mt5-bhili-hi2tgt-finetuned checkpoint
against the HAQ Bhili test set (Hindi -> Bhili only -- that's the only
direction in this test file), num_beams=4. Same T5-style instruction
prefix ("translate Hindi to Bhili: ") as mt5_finetune.py's own manual
inference block. Results saved under config.json's "scores_root"/mt5/
(see common_haq.py).

Run (1 GPU, mt5 conda env):
    python run_mt5.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common_haq import load_config, load_pairs, score_and_save

MAX_LENGTH = 128
BATCH_SIZE = 16
PREFIX = "translate Hindi to Bhili: "


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ckpt_dir = Path(cfg["mt5_ckpt"])
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()

    ids, src_texts, ref_texts = load_pairs(cfg)
    print(f"[rows] {len(ids)}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = [PREFIX + str(t) for t in src_texts[i:i + BATCH_SIZE]]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=cfg["num_beams"])
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    score_and_save("mt5", ids, src_texts, ref_texts, preds, cfg)
    print("\nmT5/HAQ-Bhili eval done.")


if __name__ == "__main__":
    main()
