"""
Inference-only: score the existing correct_checkpoints/qwen/<lang>/
qwen-<lang>-<direction>-finetuned checkpoints (already trained, not
retrained here) against the 210-row reference set, hi2tgt + tgt2hi,
num_beams=4. Same ChatML prompt format and generation settings
(MAX_LENGTH=256, MAX_NEW_TOKENS=128) as qwen_finetune.py's
generate_and_score().

Run (1 GPU, qwen_llama conda env):
    python run_qwen.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, load_test_pairs, score_and_save

MODEL_NAME = "qwen"
MAX_LENGTH = 256
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16
SYSTEM_PROMPT = "You are a helpful assistant."
HIN_NAME = "Hindi"


def build_messages(src_name, tgt_name, src_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Translate the following text from {src_name} to {tgt_name}:\n{src_text}"},
    ]


def run_direction(lang, direction, cfg, device):
    ckpt_dir = Path(cfg["checkpoints_root"]) / "qwen" / lang / f"qwen-{lang}-{direction}-finetuned"
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== qwen / {lang} / {direction} =====\n  checkpoint: {ckpt_dir}")

    tgt_name = cfg["languages"][lang]["column"]
    src_name, dst_name = (HIN_NAME, tgt_name) if direction == "hi2tgt" else (tgt_name, HIN_NAME)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer.padding_side = "left"

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id]

    ids, src_texts, ref_texts = load_test_pairs(cfg, lang, direction)

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = src_texts[i:i + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(src_name, dst_name, t), tokenize=False, add_generation_prompt=True)
                for t in chunk
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_LENGTH, add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=cfg["num_beams"],
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
            gen_only = out[:, enc["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            preds.extend(p.strip() for p in decoded)
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
    print("\nAll Qwen reference-eval jobs done.")


if __name__ == "__main__":
    main()
