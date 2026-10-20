"""
Inference-only: score the existing correct_checkpoints/sarvam_translate/
<lang>/sarvam-<lang>-<direction>-finetuned checkpoints (already trained,
not retrained here) against the 210-row reference set, hi2tgt + tgt2hi,
num_beams=4. Same prompt format (system: "Translate the text below to
{tgt_name}.", user: source verbatim) and generation settings
(max_new_tokens=256) as sarvam_finetune.py's generate_and_score().

Run (1 GPU, sarvam conda env):
    python run_sarvam.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, load_test_pairs, score_and_save

MODEL_NAME = "sarvam"
MAX_NEW_TOKENS = 256
MAX_INPUT_LENGTH = 2048
BATCH_SIZE = 16


def build_messages(tgt_name, src_text):
    return [
        {"role": "system", "content": f"Translate the text below to {tgt_name}."},
        {"role": "user", "content": str(src_text)},
    ]


def run_direction(lang, direction, cfg, device):
    ckpt_dir = Path(cfg["checkpoints_root"]) / "sarvam_translate" / lang / f"sarvam-{lang}-{direction}-finetuned"
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== sarvam / {lang} / {direction} =====\n  checkpoint: {ckpt_dir}")

    tgt_col = cfg["languages"][lang]["column"]
    tgt_name = tgt_col if direction == "hi2tgt" else "Hindi"

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer.padding_side = "left"

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

    ids, src_texts, ref_texts = load_test_pairs(cfg, lang, direction)

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = src_texts[i:i + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(tgt_name, t), tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_INPUT_LENGTH, add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=cfg["num_beams"],
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            preds.extend(p.strip() for p in decoded)
            print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    tokenizer.padding_side = "right"
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
    print("\nAll Sarvam reference-eval jobs done.")


if __name__ == "__main__":
    main()
