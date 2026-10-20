"""
Inference-only: score the existing correct_checkpoints/nllb/<lang>/
nllb-<lang>-<direction>-finetuned checkpoints (already trained, not
retrained here) against the 210-row reference set, hi2tgt + tgt2hi,
num_beams=4.

The tribal-language special token (bhb_Deva/unr_Deva/gon_Deva) was added
via tokenizer.add_special_tokens() + model.resize_token_embeddings() at
training time (hb_pja_nllb_finetune.py's load_tokenizer_and_model()) and
both got persisted by save_pretrained() -- plain AutoTokenizer/AutoModel
.from_pretrained(checkpoint_dir) already has it, no re-adding needed here.
forced_bos_token_id + tokenizer.src_lang control the generation direction,
same as the finetune script's own test-set inference block.

Run (1 GPU):
    python run_nllb.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import load_config, load_test_pairs, score_and_save

MODEL_NAME = "nllb"
MAX_LENGTH = 128
BATCH_SIZE = 16
HIN_CODE = "hin_Deva"


def run_direction(lang, direction, cfg, device):
    ckpt_dir = Path(cfg["checkpoints_root"]) / "nllb" / lang / f"nllb-{lang}-{direction}-finetuned"
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== nllb / {lang} / {direction} =====\n  checkpoint: {ckpt_dir}")

    tribal_code = cfg["languages"][lang]["lang_code"]
    src_code, tgt_code = (HIN_CODE, tribal_code) if direction == "hi2tgt" else (tribal_code, HIN_CODE)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_dir)).to(device)
    model.eval()
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)

    ids, src_texts, ref_texts = load_test_pairs(cfg, lang, direction)

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
    print("\nAll NLLB reference-eval jobs done.")


if __name__ == "__main__":
    main()
