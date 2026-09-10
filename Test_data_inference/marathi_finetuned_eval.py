"""
Run the 4 finetuned Marathi<->Hindi checkpoints (mt5 + nllb, both directions)
over a test CSV and write one scored CSV per checkpoint, with row-wise BLEU
and chrF++.

Input CSV columns: id, hindi, marathi   (see test_sample_100.csv)

Jobs:
    mt5_hi2mr   mt5-marathi-hi2mr-finetuned    source=hindi   ref=marathi
    mt5_mr2hi   mt5-marathi-mr2hi-finetuned    source=marathi ref=hindi
    nllb_hi2mr  nllb-marathi-hi2mr-finetuned   source=hindi   ref=marathi
    nllb_mr2hi  nllb-marathi-mr2hi-finetuned   source=marathi ref=hindi

Generation recipe matches the finetune / infer scripts:
    mt5  : T5 prefix "translate <Src> to <Tgt>: "
    nllb : tokenizer.src_lang + forced_bos_token_id (no prefix)
    MAX_LENGTH=128, NUM_BEAMS=4 (both overridable), batch_decode(skip_special_tokens=True)

Scoring (sacrebleu):
    row   : sentence_bleu, sentence_chrf(word_order=2)   -> chrF++
    corpus: corpus_bleu, corpus_chrf(word_order=2)       -> printed summary

Outputs (into --out_dir, default = this script's dir): 4 files
    mt5_hi2mr_scored.csv   mt5_mr2hi_scored.csv
    nllb_hi2mr_scored.csv  nllb_mr2hi_scored.csv
each: id, model, direction, source, reference, prediction, bleu, chrf++

Run:
    python marathi_finetuned_eval.py
"""

import os
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

HERE = Path(__file__).resolve().parent

DEFAULT_INPUT = "/home/scai/msr/aiy257590/flash/Interpretability-low-resource/cross_Attn_faithfulness/data_curation/Marathi/test_sample_100.csv"
_BASE = "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP"
DEFAULT_CKPTS = {
    "mt5_hi2mr":  f"{_BASE}/mt5_finetune/Marathi/mt5-marathi-hi2mr-finetuned",
    "mt5_mr2hi":  f"{_BASE}/mt5_finetune/Marathi/mt5-marathi-mr2hi-finetuned",
    "nllb_hi2mr": f"{_BASE}/NLLB-finetune/Marathi/nllb-marathi-hi2mr-finetuned",
    "nllb_mr2hi": f"{_BASE}/NLLB-finetune/Marathi/nllb-marathi-mr2hi-finetuned",
}

# (job_name, family, direction)
JOBS = [
    ("mt5_hi2mr",  "mt5",  "hi2mr"),
    ("mt5_mr2hi",  "mt5",  "mr2hi"),
    ("nllb_hi2mr", "nllb", "hi2mr"),
    ("nllb_mr2hi", "nllb", "mr2hi"),
]

HIN_NAME, MR_NAME = "Hindi", "Marathi"
HIN_CODE, MR_CODE = "hin_Deva", "mar_Deva"


def run_job(job_name, family, direction, ckpt, df, args, device):
    hi_col, mr_col = args.hi_col, args.mr_col
    if direction == "hi2mr":
        src_col, ref_col, src_name, tgt_name, src_code, tgt_code = \
            hi_col, mr_col, HIN_NAME, MR_NAME, HIN_CODE, MR_CODE
    else:  # mr2hi
        src_col, ref_col, src_name, tgt_name, src_code, tgt_code = \
            mr_col, hi_col, MR_NAME, HIN_NAME, MR_CODE, HIN_CODE

    ids  = df[args.id_col].astype(str).tolist()
    srcs = df[src_col].fillna("").astype(str).str.strip().tolist()
    refs = df[ref_col].fillna("").astype(str).str.strip().tolist()
    n = len(srcs)
    print(f"\n===== {job_name}  ({src_name} -> {tgt_name})  {n} rows =====")
    print(f"  checkpoint: {ckpt}")

    if family == "nllb":
        tok = AutoTokenizer.from_pretrained(ckpt, src_lang=src_code)
        forced_bos_id = tok.convert_tokens_to_ids(tgt_code)
        if forced_bos_id == tok.unk_token_id:
            raise ValueError(f"{job_name}: tgt lang tag {tgt_code!r} is <unk> for this checkpoint's tokenizer")
        prefix = ""
        gen_kwargs = {"forced_bos_token_id": forced_bos_id}
    else:  # mt5
        tok = AutoTokenizer.from_pretrained(ckpt)
        prefix = f"translate {src_name} to {tgt_name}: "
        gen_kwargs = {}

    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()

    preds = [""] * n
    todo = [i for i in range(n) if srcs[i]]
    bs = args.batch_size
    with torch.no_grad():
        for b in range(0, len(todo), bs):
            idxs = todo[b:b + bs]
            chunk = [prefix + srcs[i] for i in idxs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(device)
            out = model.generate(**enc, max_length=args.max_length,
                                 num_beams=args.num_beams, **gen_kwargs)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            for i, d in zip(idxs, dec):
                preds[i] = d.strip()
            print(f"  decoded {min(b + bs, len(todo))}/{len(todo)}", flush=True)

    rows, hyps_c, refs_c = [], [], []
    for i in range(n):
        h, r = preds[i], refs[i]
        if h and r:
            sb = sacrebleu.sentence_bleu(h, [r]).score
            sc = sacrebleu.sentence_chrf(h, [r], word_order=2).score
            hyps_c.append(h)
            refs_c.append(r)
        else:
            sb = sc = None
        rows.append({
            "id": ids[i], "model": family, "direction": direction,
            "source": srcs[i], "reference": r, "prediction": h,
            "bleu": round(sb, 4) if sb is not None else "",
            "chrf++": round(sc, 4) if sc is not None else "",
        })

    out_path = Path(args.out_dir) / f"{job_name}_scored.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    corpus_bleu = sacrebleu.corpus_bleu(hyps_c, [refs_c]).score if hyps_c else float("nan")
    corpus_chrf = sacrebleu.corpus_chrf(hyps_c, [refs_c], word_order=2).score if hyps_c else float("nan")
    row_bleu_mean = pd.to_numeric([r["bleu"] for r in rows if r["bleu"] != ""]).mean()
    row_chrf_mean = pd.to_numeric([r["chrf++"] for r in rows if r["chrf++"] != ""]).mean()
    print(f"  -> {out_path}")
    print(f"  corpus  BLEU={corpus_bleu:.4f}  chrF++={corpus_chrf:.4f}")
    print(f"  row-mean BLEU={row_bleu_mean:.4f}  chrF++={row_chrf_mean:.4f}  (scored {len(hyps_c)}/{n})")
    return {
        "job": job_name, "model": family, "direction": direction,
        "scored_rows": len(hyps_c),
        "corpus_bleu": round(corpus_bleu, 4), "corpus_chrf++": round(corpus_chrf, 4),
        "row_mean_bleu": round(float(row_bleu_mean), 4), "row_mean_chrf++": round(float(row_chrf_mean), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=DEFAULT_INPUT)
    ap.add_argument("--out_dir", default=str(HERE))
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--hi_col", default="hindi")
    ap.add_argument("--mr_col", default="marathi")
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    for k, v in DEFAULT_CKPTS.items():
        ap.add_argument(f"--{k}", default=v, help=f"checkpoint for {k}")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  | beams={args.num_beams} max_length={args.max_length}")

    df = pd.read_csv(args.input_csv)
    for c in (args.id_col, args.hi_col, args.mr_col):
        assert c in df.columns, f"column {c!r} not in {list(df.columns)}"
    print(f"[read] {args.input_csv}  ({len(df)} rows)")

    summary = []
    for job_name, family, direction in JOBS:
        ckpt = getattr(args, job_name)
        if not Path(ckpt).exists():
            print(f"[skip] {job_name}: checkpoint not found -> {ckpt}")
            continue
        summary.append(run_job(job_name, family, direction, ckpt, df, args, device))

    if summary:
        print("\n================ SUMMARY ================")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
