"""
Inference-only: score the ORIGINAL NLLB-finetune/Marathi/
nllb-marathi-<direction>-finetuned checkpoints (hi2mr, mr2hi -- already
trained, not retrained here) against the 210-row reference set,
num_beams=4. Hindi (hin_Deva) and Marathi (mar_Deva) are both native
NLLB-200 language codes -- no custom tag/embedding resize needed (unlike
the tribal-language NLLB checkpoints). Results saved centrally alongside
bhili/mundari/gondi under Interpretability-low-resource/scores/marathi/nllb/
(see common_marathi.py).

Run (1 GPU):
    python run_nllb.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common_marathi import load_config, load_marathi_pairs, score_and_save

MAX_LENGTH = 128
BATCH_SIZE = 16
HIN_CODE = "hin_Deva"
MAR_CODE = "mar_Deva"


def run_direction(direction, cfg, mcfg, device):
    ckpt_dir = Path(mcfg["output_root"]) / mcfg["ckpt_name_fmt"].format(direction=direction)
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== nllb / marathi / {direction} =====\n  checkpoint: {ckpt_dir}")

    hi_to_mar = direction == "hi2mr"
    src_code, tgt_code = (HIN_CODE, MAR_CODE) if hi_to_mar else (MAR_CODE, HIN_CODE)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)

    ids, src_texts, ref_texts = load_marathi_pairs(cfg["test_csv"], hi_to_mar)

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = [str(t) for t in src_texts[i:i + BATCH_SIZE]]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **enc, max_length=MAX_LENGTH, num_beams=cfg["num_beams"],
                forced_bos_token_id=forced_bos_id,
            )
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    score_and_save("nllb", direction, ids, src_texts, ref_texts, preds, cfg)
    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    mcfg = cfg["models"]["nllb"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    for direction in mcfg["directions"]:
        run_direction(direction, cfg, mcfg, device)
    print("\nAll NLLB/Marathi reference-eval jobs done.")


if __name__ == "__main__":
    main()
