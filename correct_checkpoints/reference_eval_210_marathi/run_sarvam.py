"""
Inference-only: score the ORIGINAL sarvam_finetune/Marathi/
sarvam-marathi-<direction>-finetuned checkpoints (hi2tgt, tgt2hi --
already trained, not retrained here) against the 210-row reference set,
num_beams=4. Same prompt format as sarvam_finetune.py's generate_and_score
(system: "Translate the text below to {tgt_name}.", user: source
verbatim). Results saved inside sarvam_finetune/Marathi/
reference_eval_210/ (see common_marathi.py).

Run (1 GPU, sarvam conda env):
    python run_sarvam.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_marathi import load_config, load_marathi_pairs, score_and_save

MAX_NEW_TOKENS = 256
MAX_INPUT_LENGTH = 2048
BATCH_SIZE = 16


def build_messages(tgt_name, src_text):
    return [
        {"role": "system", "content": f"Translate the text below to {tgt_name}."},
        {"role": "user", "content": str(src_text)},
    ]


def run_direction(direction, cfg, mcfg, device):
    ckpt_dir = Path(mcfg["output_root"]) / mcfg["ckpt_name_fmt"].format(direction=direction)
    assert ckpt_dir.is_dir(), f"missing checkpoint: {ckpt_dir}"
    print(f"\n===== sarvam / marathi / {direction} =====\n  checkpoint: {ckpt_dir}")

    hi_to_mar = direction == "hi2tgt"
    tgt_name = "Marathi" if hi_to_mar else "Hindi"

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer.padding_side = "left"

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id, end_of_turn_id]

    ids, src_texts, ref_texts = load_marathi_pairs(cfg["test_csv"], hi_to_mar)

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
    score_and_save(mcfg["output_root"], direction, ids, src_texts, ref_texts, preds)
    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    mcfg = cfg["models"]["sarvam"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    for direction in mcfg["directions"]:
        run_direction(direction, cfg, mcfg, device)
    print("\nAll Sarvam/Marathi reference-eval jobs done.")


if __name__ == "__main__":
    main()
