"""
Stage 3 of the human-evaluation pipeline -- computes SENTENCE-LEVEL spBLEU
(sacrebleu tokenize="flores200", effective_order=True -- the correct
setting for sentence-level, not corpus-level, scoring) and chrF++
(word_order=2) for every one of the 400 human-eval rows, against the
sampled reference from the real test set.

Not for annotators -- this is for later correlating automatic metrics
against the human DA/MQM scores once annotation is done (mirrors the
reference paper's Table 5: segment-level Kendall's tau / Pearson's rho
between spBLEU, chrF++, and human MQM scores).

Reads assemble_human_eval_csv.py's (stage 2) output KEY file to recover
which model is in translation_1..4 for each row, so this script's
spBLEU_N/chrF++_N columns line up with the SAME slot numbering as
<Lang>_human_eval.csv and its Direct_Assessment_N columns -- no
de-anonymizing needed to compare them later.

Run (after stage 1 + stage 2 complete; CPU is fine, needs network once for
the FLORES-200 tokenizer download, same as spBLEU_scores/compute_spbleu_scores.py):
    python compute_human_eval_metrics.py
"""

import argparse
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
N_SLOTS = 4

_BLEU_METRIC = BLEU(tokenize="flores200", effective_order=True)
_CHRF_METRIC = CHRF(word_order=2)


def compute_language(lang, directions, work_dir, out_dir):
    key_path = out_dir / f"{lang}_human_eval_KEY.csv"
    if not key_path.exists():
        print(f"[skip] {lang}: no KEY file ({key_path}) -- run assemble_human_eval_csv.py (stage 2) first")
        return None
    key_df = pd.read_csv(key_path)

    rows = []
    for direction in directions:
        samples_path = work_dir / f"{lang}_{direction}_samples.csv"
        if not samples_path.exists():
            print(f"[skip] {lang}/{direction}: no samples file ({samples_path})")
            continue
        samples = pd.read_csv(samples_path)
        refs = samples["reference"].astype(str).tolist()

        model_preds = {}
        for model_name in ["mT5", "NLLB", "Qwen", "Sarvam"]:
            p = work_dir / f"{lang}_{direction}_{model_name}_translations.csv"
            if not p.exists():
                print(f"[skip] {lang}/{direction}: missing {p}")
                model_preds = None
                break
            model_preds[model_name] = pd.read_csv(p)["prediction"].astype(str).tolist()
        if model_preds is None:
            continue

        dir_key = key_df[key_df["direction"] == direction].reset_index(drop=True)
        if len(dir_key) != len(samples):
            print(f"[skip] {lang}/{direction}: KEY row count {len(dir_key)} != samples {len(samples)}")
            continue

        for idx in range(len(samples)):
            row_id = dir_key.loc[idx, "row_id"]
            row = {"row_id": row_id, "direction": direction}
            ref = refs[idx]
            for slot in range(1, N_SLOTS + 1):
                model_name = dir_key.loc[idx, f"translation_{slot}_model"]
                hyp = model_preds[model_name][idx]
                bleu = _BLEU_METRIC.sentence_score(hyp, [ref]).score
                chrf = _CHRF_METRIC.sentence_score(hyp, [ref]).score
                row[f"spBLEU_{slot}"] = round(bleu, 4)
                row[f"chrF++_{slot}"] = round(chrf, 4)
            rows.append(row)

        print(f"[ok] {lang}/{direction}: {len(samples)} rows scored")

    if not rows:
        return None
    cols = ["row_id", "direction"]
    for slot in range(1, N_SLOTS + 1):
        cols += [f"spBLEU_{slot}", f"chrF++_{slot}"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGUAGES))
    ap.add_argument("--directions", default=",".join(DIRECTIONS))
    ap.add_argument("--work_dir", default=str(HERE / "_work"))
    ap.add_argument("--out_dir", default=str(HERE))
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    for lang in langs:
        print(f"\n===== {lang} =====")
        df = compute_language(lang, directions, work_dir, out_dir)
        if df is None:
            print(f"[skip] {lang}: nothing computed")
            continue
        out_path = out_dir / f"{lang}_human_eval_metrics.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"-> {out_path} ({len(df)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
