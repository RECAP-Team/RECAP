"""
Inference-only: score the existing correct_checkpoints/mt5/<lang>/
mt5-<lang>-<direction>-finetuned checkpoints (already trained, not
retrained here) against the 210-row reference set, hi2tgt + tgt2hi,
num_beams=4. Same T5-style instruction prefix ("translate X to Y: ") and
generation settings (MAX_LENGTH=128) as mt5_finetune.py's own manual
beam=5 inference block, just with num_beams forced to config.json's value.

Run (1 GPU):
    python run_mt5.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import load_config, load_test_pairs, score_and_save

MODEL_NAME = "mt5"
MAX_LENGTH = 128
BATCH_SIZE = 16
HIN_NAME = "Hindi"


def run_direction(lang, direction, cfg, device):
    ckpt_dir = Path(cfg["checkpoints_root"]) / "mt5" / lang / f"mt5-{lang}-{direction}-finetuned"
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== mt5 / {lang} / {direction} =====\n  checkpoint: {ckpt_dir}")

    tgt_name = cfg["languages"][lang]["column"]
    src_name, prefix_tgt_name = (HIN_NAME, tgt_name) if direction == "hi2tgt" else (tgt_name, HIN_NAME)
    prefix = f"translate {src_name} to {prefix_tgt_name}: "

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()

    ids, src_texts, ref_texts = load_test_pairs(cfg, lang, direction)

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = [prefix + str(t) for t in src_texts[i:i + BATCH_SIZE]]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(**enc, max_length=MAX_LENGTH, num_beams=cfg["num_beams"])
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    score_and_save(MODEL_NAME, lang, direction, ids, src_texts, ref_texts, preds, cfg)
    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    for lang in cfg["languages"]:
        for direction in cfg["directions"]:
            run_direction(lang, direction, cfg, device)
    print("\nAll mT5 reference-eval jobs done.")


if __name__ == "__main__":
    main()
