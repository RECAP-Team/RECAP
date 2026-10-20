"""
Inference-only: score the ORIGINAL indictrans2_finetune/Marathi/
indictrans2-marathi-<direction>-lora LoRA adapters (hi2tgt, tgt2hi --
already trained, not retrained here) against the 210-row reference set,
num_beams=4. Marathi (mar_Deva) is already a native IndicTrans2 language
tag -- no add_bhili_tag-style vocab extension needed (needs_new_tag=false
for Marathi in indictrans2_finetune/config.json). Same base model
(IndicTrans2-indic-indic-1B) and IndicProcessor pre/post-processing as
finetune_indictrans2.py / infer_indictrans2.py. Results saved inside
indictrans2_finetune/Marathi/reference_eval_210/ (see common_marathi.py).

Run (1 GPU, mt5 conda env -- same env the original indictrans2 scripts use):
    python run_indictrans2.py
"""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from IndicTransToolkit import IndicProcessor
from peft import PeftModel

from common_marathi import load_config, load_marathi_pairs, score_and_save

MAX_LENGTH = 256
BATCH_SIZE = 16
HIN_CODE = "hin_Deva"
MAR_CODE = "mar_Deva"


def run_direction(direction, cfg, mcfg, device):
    adapter_dir = Path(mcfg["output_root"]) / mcfg["ckpt_name_fmt"].format(direction=direction)
    assert adapter_dir.is_dir(), f"missing LoRA adapter dir: {adapter_dir}"

    # Prefer a finished top-level adapter; otherwise fall back to the
    # latest checkpoint-N/ (job killed mid-training by walltime/preemption
    # before trainer.save_model() ever ran at the top level -- same
    # situation as the Mundari IndicTrans2 en2tgt job).
    if (adapter_dir / "adapter_config.json").exists():
        checkpoint_path = adapter_dir
    else:
        checkpoint_path = get_last_checkpoint(str(adapter_dir))
        assert checkpoint_path is not None, \
            f"no finished adapter or checkpoint-N/ found under {adapter_dir}"
    print(f"\n===== indictrans2 / marathi / {direction} =====\n  adapter: {checkpoint_path}")

    hi_to_mar = direction == "hi2tgt"
    src_code, dst_code = (HIN_CODE, MAR_CODE) if hi_to_mar else (MAR_CODE, HIN_CODE)

    base_model_path = cfg["indictrans2_base_model"]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path, trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    processor = IndicProcessor(inference=True)

    ids, src_texts, ref_texts = load_marathi_pairs(cfg["test_csv"], hi_to_mar)
    src_texts = [str(t) for t in src_texts]
    ref_texts = [str(t) for t in ref_texts]

    preds = []
    with torch.no_grad():
        for i in range(0, len(src_texts), BATCH_SIZE):
            chunk = src_texts[i:i + BATCH_SIZE]
            batch = processor.preprocess_batch(chunk, src_lang=src_code, tgt_lang=dst_code)
            inputs = tokenizer(
                batch, truncation=True, padding="longest",
                return_tensors="pt", max_length=MAX_LENGTH,
            ).to(device)
            out = model.generate(
                **inputs, max_length=MAX_LENGTH,
                num_beams=cfg["num_beams"], num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds.extend(processor.postprocess_batch(decoded, lang=dst_code))
            print(f"  decoded {i + len(chunk)}/{len(src_texts)}")

    score_and_save(mcfg["output_root"], direction, ids, src_texts, ref_texts, preds)
    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    mcfg = cfg["models"]["indictrans2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    for direction in mcfg["directions"]:
        run_direction(direction, cfg, mcfg, device)
    print("\nAll IndicTrans2/Marathi reference-eval jobs done.")


if __name__ == "__main__":
    main()
