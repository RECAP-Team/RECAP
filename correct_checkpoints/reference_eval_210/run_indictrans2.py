"""
Inference-only: score the existing correct_checkpoints/indictrans2/<lang>/
indictrans2-<lang>-<direction>-lora LoRA adapters (already trained, not
retrained here) against the 210-row reference set, hi2tgt + tgt2hi,
num_beams=4. Same base model (IndicTrans2-indic-indic-1B -- both hi2tgt
and tgt2hi are indic<->indic directions), IndicProcessor pre/post-
processing, and bhb_Deva tag registration (add_bhili_tag, needed only for
Bhili -- Mundari/Gondi were already native IndicTrans2 language tags) as
finetune_indictrans2.py / infer_indictrans2.py.

Run (1 GPU, mt5 conda env -- same env the original indictrans2 scripts use):
    python run_indictrans2.py
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from peft import PeftModel

from common import load_config, load_test_pairs, score_and_save

MODEL_NAME = "indictrans2"
MAX_LENGTH = 256
BATCH_SIZE = 16
HIN_CODE = "hin_Deva"


def add_bhili_tag(model, tokenizer, new_tag, donor_tag=HIN_CODE):
    """Verbatim copy of finetune_indictrans2.py's add_bhili_tag() -- must run
    BEFORE loading the LoRA adapter, same order as training, since the
    adapter's modules_to_save state for encoder.embed_tokens was saved at
    this resized shape."""
    if new_tag in tokenizer.src_encoder:
        return

    donor_id = tokenizer.src_encoder[donor_tag]
    new_id = len(tokenizer.src_encoder)
    tokenizer.src_encoder[new_tag] = new_id
    tokenizer.src_decoder[new_id] = new_tag
    tokenizer.add_new_language_tags([new_tag])

    embed = model.model.encoder.embed_tokens
    old_n, dim = embed.weight.shape
    assert new_id == old_n, f"vocab/embedding size mismatch: new_id={new_id} old_n={old_n}"

    new_embed = nn.Embedding(old_n + 1, dim, padding_idx=embed.padding_idx)
    new_embed.weight.data[:old_n] = embed.weight.data
    new_embed.weight.data[old_n] = embed.weight.data[donor_id].clone()
    new_embed = new_embed.to(dtype=embed.weight.dtype, device=embed.weight.device)

    model.model.encoder.embed_tokens = new_embed
    model.config.encoder_vocab_size = old_n + 1
    print(f"  [vocab] added {new_tag!r} as src-vocab id {new_id} (warm-started from {donor_tag!r})")


def run_direction(lang, direction, cfg, device):
    adapter_dir = Path(cfg["checkpoints_root"]) / "indictrans2" / lang / f"indictrans2-{lang}-{direction}-lora"
    assert adapter_dir.is_dir(), f"missing LoRA adapter: {adapter_dir}"
    print(f"\n===== indictrans2 / {lang} / {direction} =====\n  adapter: {adapter_dir}")

    lang_cfg = cfg["languages"][lang]
    tribal_code = lang_cfg["lang_code"]
    src_code, dst_code = (HIN_CODE, tribal_code) if direction == "hi2tgt" else (tribal_code, HIN_CODE)

    base_model_path = cfg["indictrans2_base_model"]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path, trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    if lang_cfg["needs_new_tag"]:
        add_bhili_tag(model, tokenizer, new_tag=tribal_code, donor_tag=HIN_CODE)

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    processor = IndicProcessor(inference=True)

    ids, src_texts, ref_texts = load_test_pairs(cfg, lang, direction)
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
    print("\nAll IndicTrans2 reference-eval jobs done.")


if __name__ == "__main__":
    main()
