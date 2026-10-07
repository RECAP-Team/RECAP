"""
Stage 2 of the human-evaluation pipeline -- takes the per-model translation
files produced by generate_human_eval_translations.py (stage 1) and builds:

  1. <Lang>_human_eval.csv  -- the ANNOTATOR-FACING sheet. For each of the 4
     translation slots per row, which MODEL it actually came from is
     shuffled independently per row (not a fixed column->model mapping) and
     never appears in this file -- this mirrors the reference paper's own
     protocol (Section 5.1): "these source translation pairs were presented
     to language experts in randomized order without revealing the language
     model identities". DA/Error_type/Severity columns are left blank for
     annotators to fill in.

  2. <Lang>_human_eval_KEY.csv  -- SECRET. Row-by-row mapping of which model
     produced translation_1..translation_4 for that row. Keep this file
     OUT of whatever annotators see -- it is what lets you recover
     per-model results after annotation is done.

  3. README.md -- protocol documentation (scales, column meanings,
     instructions), written once.

Run (after stage 1 completes, no GPU needed):
    python assemble_human_eval_csv.py
"""

import argparse
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LANGUAGES = ["Mundari", "Gondi"]
DIRECTIONS = ["hi2tgt", "tgt2hi", "en2tgt", "tgt2en"]
MODELS = ["mT5", "NLLB", "Qwen", "Sarvam"]
N_SLOTS = len(MODELS)

ANNOTATOR_COLUMNS = ["source_sentence", "direction"]
for i in range(1, N_SLOTS + 1):
    ANNOTATOR_COLUMNS += [f"translation_{i}", f"Direct_Assessment_{i}", f"Error_type_{i}", f"Severity_{i}"]

KEY_COLUMNS = ["row_id", "direction"] + [f"translation_{i}_model" for i in range(1, N_SLOTS + 1)]


def assemble_language(lang, directions, work_dir, seed):
    rng = random.Random(seed)  # one RNG, advanced row-by-row -- deterministic given (lang, seed)
    annotator_rows = []
    key_rows = []

    for direction in directions:
        samples_path = work_dir / f"{lang}_{direction}_samples.csv"
        if not samples_path.exists():
            print(f"[skip] {lang}/{direction}: no samples file ({samples_path}) -- run stage 1 first")
            continue
        samples = pd.read_csv(samples_path)

        model_preds = {}
        missing = []
        for model_name in MODELS:
            p = work_dir / f"{lang}_{direction}_{model_name}_translations.csv"
            if not p.exists():
                missing.append(model_name)
                continue
            df = pd.read_csv(p)
            if len(df) != len(samples):
                missing.append(f"{model_name} (row count {len(df)} != {len(samples)})")
                continue
            model_preds[model_name] = df["prediction"].astype(str).tolist()
        if missing:
            print(f"[skip] {lang}/{direction}: missing/mismatched models {missing} -- run stage 1 for these first")
            continue

        n = len(samples)
        for idx in range(n):
            row_id = f"{lang}_{direction}_{idx:03d}"
            slot_models = MODELS.copy()
            rng.shuffle(slot_models)  # independent per-row shuffle -- see module docstring

            row = {"source_sentence": samples.loc[idx, "source"], "direction": direction}
            key_row = {"row_id": row_id, "direction": direction}
            for slot, model_name in enumerate(slot_models, start=1):
                row[f"translation_{slot}"] = model_preds[model_name][idx]
                row[f"Direct_Assessment_{slot}"] = ""
                row[f"Error_type_{slot}"] = ""
                row[f"Severity_{slot}"] = ""
                key_row[f"translation_{slot}_model"] = model_name
            annotator_rows.append(row)
            key_rows.append(key_row)

        print(f"[ok] {lang}/{direction}: {n} rows assembled")

    return pd.DataFrame(annotator_rows, columns=ANNOTATOR_COLUMNS), pd.DataFrame(key_rows, columns=KEY_COLUMNS)


README_TEMPLATE = """# Human Evaluation -- {lang}

Protocol mirrors 2025.findings-emnlp.508.pdf, Section 5.1 (Quantitative
Human Evaluation: Alignment with Automatic Metrics), scaled down to this
project's 4 models / 2 languages / 4 directions.

## Files
- `{lang}_human_eval.csv` -- give this to annotators. {n_rows} rows total
  ({n_dirs} directions x {n_per_dir} sampled test-set segments each).
- `{lang}_human_eval_KEY.csv` -- **DO NOT share with annotators.** Maps each
  row's translation_1..translation_{n_slots} back to the real model name.
  Needed to recover per-model scores after annotation.

## Columns (annotator-facing CSV)
- `source_sentence` -- the source-language segment to translate (sampled
  from the real test set, same {n_per_dir} segments across all {n_slots}
  models for a given direction).
- `direction` -- one of hi2tgt (Hindi->{lang}), tgt2hi ({lang}->Hindi),
  en2tgt (English->{lang}), tgt2en ({lang}->English).
- `translation_N` -- one model's candidate translation. WHICH model
  produced it is shuffled independently per row (see KEY file) -- this
  blinds the annotator to model identity, matching the reference paper's
  protocol exactly.
- `Direct_Assessment_N` -- annotator fills in, 1-5 scale (overall
  adequacy/fluency of translation_N).
- `Error_type_N` -- annotator fills in, from the agreed error-category list
  (see below).
- `Severity_N` -- annotator fills in, one of: good, low, medium, high,
  veryhigh (mapped 1-5 for analysis: good=1, low=2, medium=3, high=4,
  veryhigh=5).

## Error type categories
TODO -- finalize with annotators before distributing. Candidate starting
list (from the reference paper's own qualitative findings, Section 5.2):
  - Language Mixing
  - Hallucination / Omission
  - Polysemy / Lexical Ambiguity
  - Domain-Specific Translation Failure
  - (add/replace as needed for {lang})

## Generation details
- Beam size 4 (num_beams=4), same as every model's real benchmark
  inference -- see generate_human_eval_translations.py.
- Segments sampled with a fixed random seed from each direction's real
  test.csv, independent of any model's own test-time predictions already
  used for spBLEU/chrF++ scoring.
- Shuffle seed: {seed} (deterministic -- rerunning assemble_human_eval_csv.py
  with the same seed reproduces the identical shuffle/KEY mapping).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGUAGES))
    ap.add_argument("--directions", default=",".join(DIRECTIONS))
    ap.add_argument("--work_dir", default=str(HERE / "_work"))
    ap.add_argument("--out_dir", default=str(HERE))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        print(f"\n===== {lang} =====")
        annotator_df, key_df = assemble_language(lang, directions, work_dir, args.seed)
        if annotator_df.empty:
            print(f"[skip] {lang}: nothing assembled (run stage 1 first)")
            continue

        annotator_path = out_dir / f"{lang}_human_eval.csv"
        key_path = out_dir / f"{lang}_human_eval_KEY.csv"
        annotator_df.to_csv(annotator_path, index=False, encoding="utf-8-sig")
        key_df.to_csv(key_path, index=False, encoding="utf-8-sig")
        print(f"-> {annotator_path} ({len(annotator_df)} rows)")
        print(f"-> {key_path} (KEEP SECRET -- not for annotators)")

        readme_path = out_dir / f"{lang}_README.md"
        readme_path.write_text(README_TEMPLATE.format(
            lang=lang, n_rows=len(annotator_df), n_dirs=len(directions),
            n_per_dir=len(annotator_df) // max(len(directions), 1),
            n_slots=N_SLOTS, seed=args.seed))
        print(f"-> {readme_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
