"""
Inference-only: score the ORIGINAL mt5_finetune/Marathi/
mt5-marathi-<direction>-finetuned checkpoints (hi2mr, mr2hi -- already
trained, not retrained here) against the 210-row reference set,
num_beams=4. Same T5-style instruction prefix as mt5_finetune.py's own
manual inference block. Results saved inside mt5_finetune/Marathi/
reference_eval_210/ (see common_marathi.py).

Run (1 GPU):
    python run_mt5.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common_marathi import load_config, load_marathi_pairs, score_and_save

MAX_LENGTH = 128
BATCH_SIZE = 16
HIN_NAME = "Hindi"
MAR_NAME = "Marathi"


def run_direction(direction, cfg, mcfg, device):
    ckpt_dir = Path(mcfg["output_root"]) / mcfg["ckpt_name_fmt"].format(direction=direction)
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== mt5 / marathi / {direction} =====\n  checkpoint: {ckpt_dir}")

    hi_to_mar = direction == "hi2mr"
    src_name, tgt_name = (HIN_NAME, MAR_NAME) if hi_to_mar else (MAR_NAME, HIN_NAME)
    prefix = f"translate {src_name} to {tgt_name}: "

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()

    ids, src_texts, ref_texts = load_marathi_pairs(cfg["test_csv"], hi_to_mar)

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

    score_and_save(mcfg["output_root"], direction, ids, src_texts, ref_texts, preds)
    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    mcfg = cfg["models"]["mt5"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    for direction in mcfg["directions"]:
        run_direction(direction, cfg, mcfg, device)
    print("\nAll mT5/Marathi reference-eval jobs done.")


if __name__ == "__main__":
    main()
