"""
Builds the shared-50-row human-eval CSV entirely from ALREADY-DOWNLOADED
local predictions (inference_sentences/) -- no Pragya/GPU needed, since all
4 models' full test-set predictions for all 4 directions are already
sitting locally.

Approach: sample 50 (Hindi, English, Target) sentence-triples from one
model's hi2tgt file (the base sample), then for every (model, direction)
combination, look up that EXACT model's prediction for those same
underlying sentences by matching on source text -- this guarantees all 4
directions, for every model, are built from the identical 50 sentences per
language (Mundari's 50 independent of Gondi's 50, per the user's earlier
confirmation that the two languages don't share source sentences).

Run (local Mac, from this directory, after cd-ing into inference_sentences
is NOT needed -- paths are relative to --inference_root):
    python build_shared50_from_existing.py
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
MODELS = [("mT5", "mt5_finetune"), ("NLLB", "NLLB-finetune"),
          ("Qwen", "qwen_finetune"), ("Sarvam", "sarvam_finetune")]
N_SAMPLES = 50
SEED = 42


def load_direction_csv(inference_root, model_root, lang, direction):
    path = Path(inference_root) / model_root / lang / f"infer_predictions_{lang.lower()}_{direction}_test.csv"
    df = pd.read_csv(path)
    src_col = next(c for c in df.columns if c.startswith("source_"))
    ref_col = next(c for c in df.columns if c.startswith("reference_"))
    return df, src_col, ref_col


def build_base_sample(inference_root, lang, n_samples, seed):
    """50 (Hindi, English, Target) triples, sampled from the hi2tgt file
    (gives Hindi+Target directly) and joined to the en2tgt file (to pick up
    the matching English text for the same underlying row, via exact
    Target-text match)."""
    hi2tgt_df, hi_src_col, hi_ref_col = load_direction_csv(inference_root, "mt5_finetune", lang, "hi2tgt")
    en2tgt_df, en_src_col, en_ref_col = load_direction_csv(inference_root, "mt5_finetune", lang, "en2tgt")

    hi2tgt_df = hi2tgt_df.rename(columns={hi_src_col: "Hindi", hi_ref_col: "Target"})[["Hindi", "Target"]]
    en2tgt_df = en2tgt_df.rename(columns={en_src_col: "English", en_ref_col: "Target"})[["English", "Target"]]

    # Keep only Target sentences present in BOTH (so English lookup succeeds for every sampled row)
    merged = hi2tgt_df.merge(en2tgt_df, on="Target", how="inner").drop_duplicates(subset="Target")
    assert len(merged) >= n_samples, f"{lang}: only {len(merged)} rows have both Hindi+English+Target -- need {n_samples}"

    sampled = merged.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    return sampled  # columns: Hindi, Target, English


def lookup_predictions(inference_root, model_root, lang, direction, query_texts):
    df, src_col, _ = load_direction_csv(inference_root, model_root, lang, direction)
    lookup = dict(zip(df[src_col].astype(str), df["prediction"].astype(str)))
    missing = [t for t in query_texts if t not in lookup]
    if missing:
        print(f"  [warn] {model_root}/{lang}/{direction}: {len(missing)}/{len(query_texts)} sentences not found "
              f"(e.g. {missing[0]!r}) -- filled with empty string")
    return [lookup.get(t, "") for t in query_texts]


SRC_TEXT_BY_DIRECTION = {
    "hi2tgt": lambda base: base["Hindi"],
    "tgt2hi": lambda base: base["Target"],
    "en2tgt": lambda base: base["English"],
    "tgt2en": lambda base: base["Target"],
}
REF_TEXT_BY_DIRECTION = {
    "hi2tgt": lambda base: base["Target"],
    "tgt2hi": lambda base: base["Hindi"],
    "en2tgt": lambda base: base["Target"],
    "tgt2en": lambda base: base["English"],
}


def build_language(inference_root, lang, n_samples, seed, work_dir):
    print(f"\n===== {lang} =====")
    base = build_base_sample(inference_root, lang, n_samples, seed)
    base.to_csv(work_dir / f"{lang}_shared50_base.csv", index=False, encoding="utf-8-sig")
    print(f"  base sample: {len(base)} rows -> {work_dir / f'{lang}_shared50_base.csv'}")

    for direction in DIRECTIONS:
        src_texts = SRC_TEXT_BY_DIRECTION[direction](base).astype(str).tolist()
        ref_texts = REF_TEXT_BY_DIRECTION[direction](base).astype(str).tolist()
        samples_df = pd.DataFrame({"source": src_texts, "reference": ref_texts})
        samples_df.to_csv(work_dir / f"{lang}_{direction}_samples.csv", index=False, encoding="utf-8-sig")

        for model_name, model_root in MODELS:
            preds = lookup_predictions(inference_root, model_root, lang, direction, src_texts)
            out_df = pd.DataFrame({"source": src_texts, "prediction": preds})
            out_path = work_dir / f"{lang}_{direction}_{model_name}_translations.csv"
            out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  {direction}: done (4 models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_root", default=str(HERE.parent / "inference_sentences"))
    ap.add_argument("--langs", default=",".join(LANGUAGES))
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--work_dir", default=str(HERE / "_work_50"))
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        build_language(args.inference_root, lang, args.n_samples, args.seed, work_dir)

    print(f"\nAll done. Now run:\n"
          f"  python assemble_human_eval_csv.py --work_dir {work_dir} --out_dir {HERE / '50samples'}\n"
          f"  python compute_human_eval_metrics.py --work_dir {work_dir} --out_dir {HERE / '50samples'}")


if __name__ == "__main__":
    main()
