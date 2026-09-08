"""
Corpus analysis for the Hindi<->Bhili and Hindi<->Marathi parallel data,
per split (train / val / test), matching exactly how each finetuning script
consumes its data:

  - bhili  : three explicit CSVs (train/val/test) with Hindi + Bhili columns
             -- the paths from mt5_finetune/Bhili/config.json.
  - marathi: one train.csv (hindi, marathi) shuffled with SEED=42 and split
             97.5 / 2.5 into train/val (same as NLLB-finetune/Marathi), plus
             a separate test.csv.

For every (dataset, split) it reports:

  1. Row counts (raw / cleaned).
  2. Length stats -- source (Hindi) and target: characters, whitespace
     words, target/source length ratio, and subword-token length under the
     mt5-base and nllb-200-distilled-600M tokenizers.
  3. Identical / near-identical pairs -- exact, whitespace/case-normalized,
     and character-similarity buckets (>=0.8/0.9/0.95), plus mean
     sentence-level chrF between the two sides (how close the languages are
     / how much the model can "translate" by copying).
  4. Duplicate rate -- duplicated source, target, and full pair; unique
     counts; largest single-source repeat.
  5. Language quality (heuristics) -- script composition (Devanagari / Latin
     / digit / punct), target rows with high Latin ratio or no Devanagari,
     very-short rows, in-sentence 4-gram repetition, source/target numeric
     mismatch, sentence-final punctuation mismatch, type-token ratio.
  6. Domain (approximate) -- Hindi vocabulary size, hapax rate, top content
     words, and a labelled heuristic keyword-bucket tally
     (health / geography-travel / government / religion / daily-life).
  7. Hindi sentence distribution -- length, duplicate rate, TTR, vocab, and
     (cross-dataset section) how much Hindi is shared between the Bhili and
     Marathi corpora.
  8. Tokenization stats -- subword length, fertility (subwords / word),
     fraction of sentences over the 128-token training cap, and <unk> rate,
     for Hindi and the target side, under both tokenizers.
  9. Train-test similarity / contamination -- exact pair / source / target
     overlap (test-vs-train, val-vs-train, val-vs-test), word-4-gram
     containment of test in train, source-vocab OOV, and (if scikit-learn
     is available) char-ngram TF-IDF nearest-neighbour near-duplicate rate
     on a sample.

Outputs (into --out_dir):
    corpus_analysis.json   full nested results
    summary.csv            one row per (dataset, split), headline numbers
    leakage.csv            one row per overlap comparison

Run:
    python corpus_analysis.py
    python corpus_analysis.py --only marathi --no_tokenization
"""

import os
import re
import json
import math
import time
import random
import difflib
import argparse
from collections import Counter
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import pandas as pd

try:
    import sacrebleu
except Exception:
    sacrebleu = None

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Dataset definitions -- paths mirror the finetuning configs.
# --------------------------------------------------------------------------
DATASETS = {
    "bhili": {
        "kind": "explicit_splits",
        "train": "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/datasets/bhilli/train.csv",
        "val":   "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/datasets/bhilli/val.csv",
        "test":  "/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/datasets/bhilli/test.csv",
        "hi_col": "Hindi", "tgt_col": "Bhili", "tgt_name": "Bhili",
        "nllb_tgt_lang": None,          # Bhili not in base NLLB-200
    },
    "marathi": {
        "kind": "split_in_script",
        "train": "/home/scai/msr/aiy257590/flash/Interpretability-low-resource/cross_Attn_faithfulness/data_curation/Marathi/train.csv",
        "test":  "/home/scai/msr/aiy257590/flash/Interpretability-low-resource/cross_Attn_faithfulness/data_curation/Marathi/test.csv",
        "val_ratio": 0.025, "seed": 42,
        "hi_col": "hindi", "tgt_col": "marathi", "tgt_name": "Marathi",
        "nllb_tgt_lang": "mar_Deva",
    },
}

DEVA  = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
DIGITS_RE = re.compile(r"[0-9०-९]+")
PUNCT = re.compile(r"[^\w\sऀ-ॿ]")
WS = re.compile(r"\s+")

HI_STOP = set("""के का की को में है और से पर कि हैं यह एक ने था थी थे हो भी तो जो वह इस कर गया
होता होती या तक साथ द्वारा लिए रहा रही रहे नहीं कुछ अपने उस जा दिया गई हुए हुई गए इसके""".split())

KEYWORDS = {
    "health":     "रोग बीमारी उपचार दवा डॉक्टर अस्पताल संक्रमण लक्षण स्वास्थ्य मरीज रक्त दर्द सूजन बुखार खांसी".split(),
    "geo_travel": "पर्वत नदी शहर गाँव गांव किला मंदिर पर्यटक यात्रा जिला घाटी द्वीप समुद्र क्षेत्र राज्य चोटी".split(),
    "government":  "सरकार मंत्री योजना विभाग अधिकारी नीति प्रशासन संसद चुनाव कानून आयोग".split(),
    "religion":   "भगवान पूजा धर्म त्योहार त्यौहार देवता व्रत मूलमंत्र मंत्र".split(),
    "daily_life": "नाश्ता भोजन खाना घर पानी दूध सुबह शाम काम बच्चे".split(),
}

FINAL_PUNCT = set(list("।.!?॥"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pct(a, b):
    return round(100.0 * a / b, 4) if b else 0.0


def num_stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {}
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "p5": round(float(np.percentile(a, 5)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(a.max()), 4),
    }


def wtok(s):
    return s.split()


def norm(s):
    return WS.sub(" ", s.strip()).lower()


def char_profile(series):
    dev = lat = dig = pun = 0
    tot = 0
    for s in series:
        for ch in s:
            tot += 1
            if DEVA.match(ch):
                dev += 1
            elif LATIN.match(ch):
                lat += 1
            elif ch.isdigit():
                dig += 1
            elif not ch.isspace():
                pun += 1
    return {
        "deva_frac": pct(dev, tot) / 100,
        "latin_frac": pct(lat, tot) / 100,
        "digit_frac": pct(dig, tot) / 100,
        "punct_frac": pct(pun, tot) / 100,
    }


def has_repeated_4gram(s):
    t = wtok(s)
    if len(t) < 8:
        return False
    seen = set()
    for i in range(len(t) - 3):
        g = tuple(t[i:i + 4])
        if g in seen:
            return True
        seen.add(g)
    return False


def digit_multiset(s):
    return Counter(DIGITS_RE.findall(s))


# --------------------------------------------------------------------------
# data loading -- mirrors the finetune scripts exactly
# --------------------------------------------------------------------------
def _load_clean(path, hi, tg):
    raw = pd.read_csv(path)
    assert hi in raw.columns and tg in raw.columns, \
        f"{path}: need columns {hi!r},{tg!r}, got {list(raw.columns)}"
    n_raw = len(raw)
    df = raw[[hi, tg]].dropna()
    for c in (hi, tg):
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[hi] != "") & (df[tg] != "")].reset_index(drop=True)
    df.columns = ["hi", "tgt"]
    return df, n_raw


def load_splits(ds):
    hi, tg = ds["hi_col"], ds["tgt_col"]
    raw_counts = {}
    if ds["kind"] == "explicit_splits":
        out = {}
        for split in ("train", "val", "test"):
            df, n_raw = _load_clean(ds[split], hi, tg)
            out[split] = df
            raw_counts[split] = n_raw
        return out, raw_counts

    full, n_raw = _load_clean(ds["train"], hi, tg)
    full = full.sample(frac=1.0, random_state=ds.get("seed", 42)).reset_index(drop=True)
    n_val = int(round(len(full) * ds.get("val_ratio", 0.025)))
    test_df, n_raw_test = _load_clean(ds["test"], hi, tg)
    raw_counts = {"train": n_raw - n_val, "val": n_val, "test": n_raw_test,
                  "train_source_file_rows": n_raw}
    return (
        {"train": full.iloc[n_val:].reset_index(drop=True),
         "val": full.iloc[:n_val].reset_index(drop=True),
         "test": test_df},
        raw_counts,
    )


# --------------------------------------------------------------------------
# per-split analysis
# --------------------------------------------------------------------------
def analyze_split(df, tgt_name, sample_n, rng):
    hi = df["hi"].tolist()
    tg = df["tgt"].tolist()
    n = len(df)
    hi_chars = np.array([len(s) for s in hi])
    tg_chars = np.array([len(s) for s in tg])
    hi_words = np.array([len(wtok(s)) for s in hi])
    tg_words = np.array([len(wtok(s)) for s in tg])
    ratio = tg_chars / np.maximum(hi_chars, 1)

    res = {"n_rows": n}

    res["length"] = {
        "hindi_chars": num_stats(hi_chars),
        "hindi_words": num_stats(hi_words),
        f"{tgt_name.lower()}_chars": num_stats(tg_chars),
        f"{tgt_name.lower()}_words": num_stats(tg_words),
        "tgt_over_src_char_ratio": num_stats(ratio),
        "extreme_ratio_pct": pct(int(((ratio < 0.3) | (ratio > 3.0)).sum()), n),
    }

    # ---- identical / near-identical (source vs target) ----
    exact = sum(1 for a, b in zip(hi, tg) if a == b)
    nexact = sum(1 for a, b in zip(hi, tg) if norm(a) == norm(b))
    idx = list(range(n))
    if n > sample_n:
        idx = rng.sample(idx, sample_n)
    ratios = [difflib.SequenceMatcher(None, hi[i], tg[i]).ratio() for i in idx]
    ratios = np.array(ratios)
    chrf_vals = None
    if sacrebleu is not None:
        chrf_vals = np.array([sacrebleu.sentence_chrf(tg[i], [hi[i]]).score for i in idx])
    res["identical_pairs"] = {
        "exact_pct": pct(exact, n),
        "normalized_exact_pct": pct(nexact, n),
        "sample_size": len(idx),
        "char_sim_ge_0.80_pct": pct(int((ratios >= 0.80).sum()), len(idx)),
        "char_sim_ge_0.90_pct": pct(int((ratios >= 0.90).sum()), len(idx)),
        "char_sim_ge_0.95_pct": pct(int((ratios >= 0.95).sum()), len(idx)),
        "char_sim_mean": round(float(ratios.mean()), 4),
        "src_tgt_sentence_chrf_mean": round(float(chrf_vals.mean()), 4) if chrf_vals is not None else None,
    }

    # ---- duplicates ----
    sc = df["hi"].value_counts()
    tc = df["tgt"].value_counts()
    pc = (df["hi"] + " ||| " + df["tgt"]).value_counts()
    res["duplicates"] = {
        "dup_source_rows_pct": pct(int((df["hi"].map(sc) > 1).sum()), n),
        "dup_target_rows_pct": pct(int((df["tgt"].map(tc) > 1).sum()), n),
        "dup_pair_rows_pct": pct(int(n - len(pc)), n),
        "unique_sources": int(len(sc)),
        "unique_targets": int(len(tc)),
        "unique_pairs": int(len(pc)),
        "max_single_source_repeat": int(sc.iloc[0]) if len(sc) else 0,
    }

    # ---- language quality ----
    hi_prof = char_profile(hi)
    tg_prof = char_profile(tg)
    tg_latin_hi = sum(1 for s in tg if s and sum(c.isascii() and c.isalpha() for c in s) / len(s) > 0.30)
    tg_no_deva = sum(1 for s in tg if not DEVA.search(s))
    hi_no_deva = sum(1 for s in hi if not DEVA.search(s))
    very_short = sum(1 for s in tg if len(s) < 3)
    rep4_hi = sum(has_repeated_4gram(s) for s in hi)
    rep4_tg = sum(has_repeated_4gram(s) for s in tg)
    num_mismatch = sum(1 for a, b in zip(hi, tg) if digit_multiset(a) != digit_multiset(b))
    fp_mismatch = sum(
        1 for a, b in zip(hi, tg)
        if a and b and ((a[-1] in FINAL_PUNCT) != (b[-1] in FINAL_PUNCT))
    )
    ttr_hi = np.array([len(set(wtok(s))) / max(len(wtok(s)), 1) for s in hi])
    ttr_tg = np.array([len(set(wtok(s))) / max(len(wtok(s)), 1) for s in tg])
    res["language_quality"] = {
        "hindi_char_profile": hi_prof,
        f"{tgt_name.lower()}_char_profile": tg_prof,
        "target_latin_gt_30pct_rows_pct": pct(tg_latin_hi, n),
        "target_no_devanagari_rows_pct": pct(tg_no_deva, n),
        "hindi_no_devanagari_rows_pct": pct(hi_no_deva, n),
        "target_shorter_than_3_chars_pct": pct(very_short, n),
        "hindi_internal_4gram_repeat_pct": pct(rep4_hi, n),
        "target_internal_4gram_repeat_pct": pct(rep4_tg, n),
        "src_tgt_numeric_mismatch_pct": pct(num_mismatch, n),
        "final_punct_mismatch_pct": pct(fp_mismatch, n),
        "hindi_type_token_ratio_mean": round(float(ttr_hi.mean()), 4),
        "target_type_token_ratio_mean": round(float(ttr_tg.mean()), 4),
    }

    # ---- domain (Hindi side) ----
    tokens = [w for s in hi for w in wtok(s)]
    vocab = Counter(tokens)
    content = Counter({w: c for w, c in vocab.items() if w not in HI_STOP and len(w) > 1})
    hapax = sum(1 for c in vocab.values() if c == 1)
    buckets = {}
    for name, kws in KEYWORDS.items():
        kset = set(kws)
        hits = sum(1 for s in hi if any(w in kset for w in wtok(s)))
        buckets[name] = pct(hits, n)
    res["domain"] = {
        "hindi_vocab_size": len(vocab),
        "hindi_token_count": len(tokens),
        "hindi_hapax_pct_of_vocab": pct(hapax, len(vocab)),
        "hindi_top_content_words": content.most_common(30),
        "heuristic_keyword_bucket_row_pct": buckets,
    }
    return res


# --------------------------------------------------------------------------
# tokenization stats
# --------------------------------------------------------------------------
def tok_lengths(tokenizer, texts, chunk=10000):
    n_special = len(tokenizer("", add_special_tokens=True).input_ids)
    unk = tokenizer.unk_token_id
    lens, n_unk, n_tok = [], 0, 0
    for i in range(0, len(texts), chunk):
        ids_batch = tokenizer(texts[i:i + chunk], add_special_tokens=True).input_ids
        for ids in ids_batch:
            lens.append(len(ids))
            n_tok += len(ids)
            if unk is not None:
                n_unk += ids.count(unk)
    return np.array(lens), n_special, n_unk, n_tok


def analyze_tokenization(splits, tgt_name, mt5_tok, nllb_tok, cap=128):
    out = {}
    for tname, tk in (("mt5", mt5_tok), ("nllb", nllb_tok)):
        if tk is None:
            out[tname] = {"skipped": "tokenizer not loaded"}
            continue
        per_tok = {}
        for split, df in splits.items():
            entry = {}
            for side, name in (("hi", "hindi"), ("tgt", tgt_name.lower())):
                texts = df[side].tolist()
                words = np.array([max(len(wtok(s)), 1) for s in texts])
                lens, n_special, n_unk, n_tok = tok_lengths(tk, texts)
                content = np.maximum(lens - n_special, 1)
                entry[name] = {
                    "subword_len": num_stats(lens),
                    "fertility_subw_per_word_mean": round(float((content / words).mean()), 4),
                    "over_%d_tok_pct" % cap: pct(int((lens > cap).sum()), len(lens)),
                    "unk_rate_pct": pct(n_unk, n_tok),
                }
            per_tok[split] = entry
        out[tname] = per_tok
    return out


# --------------------------------------------------------------------------
# train-test similarity / contamination
# --------------------------------------------------------------------------
def hashed_4grams(texts):
    out = set()
    for s in texts:
        t = wtok(s)
        for i in range(len(t) - 3):
            out.add(hash((t[i], t[i + 1], t[i + 2], t[i + 3])))
    return out


def containment_4gram(test_texts, train_grams):
    covered, total, rowfracs = 0, 0, []
    for s in test_texts:
        t = wtok(s)
        g = [hash((t[i], t[i + 1], t[i + 2], t[i + 3])) for i in range(len(t) - 3)]
        if not g:
            continue
        c = sum(1 for x in g if x in train_grams)
        covered += c
        total += len(g)
        rowfracs.append(c / len(g))
    return {
        "corpus_4gram_containment_pct": pct(covered, total),
        "mean_per_test_row_4gram_containment_pct": round(100 * float(np.mean(rowfracs)), 4) if rowfracs else 0.0,
        "test_rows_fully_covered_pct": pct(sum(1 for f in rowfracs if f >= 0.999), len(rowfracs)) if rowfracs else 0.0,
    }


def overlap_block(a, b, label):
    """a, b are DataFrames with hi/tgt. 'How much of `a` appears in `b`.'"""
    b_pairs = set((b["hi"] + " ||| " + b["tgt"]).tolist())
    b_src = set(b["hi"].tolist())
    b_tgt = set(b["tgt"].tolist())
    b_src_norm = set(norm(s) for s in b["hi"].tolist())
    a_pairs = (a["hi"] + " ||| " + a["tgt"]).tolist()
    n = len(a)
    pair_hits = sum(1 for p in a_pairs if p in b_pairs)
    src_hits = sum(1 for s in a["hi"].tolist() if s in b_src)
    tgt_hits = sum(1 for s in a["tgt"].tolist() if s in b_tgt)
    src_norm_hits = sum(1 for s in a["hi"].tolist() if norm(s) in b_src_norm)
    return {
        "comparison": label,
        "rows_in_a": n,
        "exact_pair_overlap_n": pair_hits,
        "exact_pair_overlap_pct": pct(pair_hits, n),
        "exact_source_overlap_pct": pct(src_hits, n),
        "normalized_source_overlap_pct": pct(src_norm_hits, n),
        "exact_target_overlap_pct": pct(tgt_hits, n),
    }


def vocab_oov(test_texts, train_texts):
    train_vocab = set(w for s in train_texts for w in wtok(s))
    test_types = set(w for s in test_texts for w in wtok(s))
    test_tokens = [w for s in test_texts for w in wtok(s)]
    oov_types = sum(1 for w in test_types if w not in train_vocab)
    oov_tok = sum(1 for w in test_tokens if w not in train_vocab)
    return {
        "test_source_oov_type_pct": pct(oov_types, len(test_types)),
        "test_source_oov_token_pct": pct(oov_tok, len(test_tokens)),
    }


def near_dup_tfidf(train_texts, test_texts, sample_n, rng, thresholds=(0.90, 0.95)):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        return {"skipped": f"scikit-learn unavailable ({e})"}

    tr = train_texts
    if len(tr) > 120000:
        tr = rng.sample(tr, 120000)
    te = test_texts
    if len(te) > sample_n:
        te = rng.sample(te, sample_n)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, max_features=200000)
    X = vec.fit_transform(tr)
    Q = vec.transform(te)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, _ = nn.kneighbors(Q)
    sims = 1.0 - dist.ravel()
    out = {
        "train_indexed": len(tr),
        "test_sampled": len(te),
        "top1_sim_mean": round(float(sims.mean()), 4),
        "top1_sim_median": round(float(np.median(sims)), 4),
    }
    for th in thresholds:
        out["test_rows_ge_%.2f_pct" % th] = pct(int((sims >= th).sum()), len(sims))
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(HERE / "corpus_analysis_out"))
    ap.add_argument("--only", default=None, help="comma list: bhili,marathi")
    ap.add_argument("--sample", type=int, default=5000,
                    help="rows for O(n) per-pair metrics (char-sim, chrf)")
    ap.add_argument("--near_dup_sample", type=int, default=4000, help="0 disables")
    ap.add_argument("--no_tokenization", action="store_true")
    ap.add_argument("--mt5_model",
                    default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/Models/mt5-base")
    ap.add_argument("--nllb_model",
                    default="/home/scai/msr/aiy257590/flash/final-climb-adivaani/RECAP/Models/nllb-200-distilled-600M")
    args = ap.parse_args()

    rng = random.Random(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.only.split(",") if args.only else list(DATASETS)

    mt5_tok = nllb_tok = None
    if not args.no_tokenization:
        from transformers import AutoTokenizer
        try:
            mt5_tok = AutoTokenizer.from_pretrained(args.mt5_model)
            print(f"[tok] mt5 loaded from {args.mt5_model}")
        except Exception as e:
            print(f"[tok] mt5 load failed: {e}")
        try:
            nllb_tok = AutoTokenizer.from_pretrained(args.nllb_model)
            print(f"[tok] nllb loaded from {args.nllb_model}")
        except Exception as e:
            print(f"[tok] nllb load failed: {e}")

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"sample": args.sample, "near_dup_sample": args.near_dup_sample,
                   "tokenization": not args.no_tokenization},
        "datasets": {},
    }
    summary_rows = []
    leak_rows = []
    hindi_sets = {}  # (dataset) -> set of all Hindi sentences across splits

    for name in wanted:
        ds = DATASETS[name]
        tgt_name = ds["tgt_name"]
        print(f"\n===== {name} =====")
        splits, raw_counts = load_splits(ds)
        dres = {"raw_counts": raw_counts, "splits": {}}

        for split, df in splits.items():
            print(f"  [{split}] {len(df)} rows -- analyzing ...")
            dres["splits"][split] = analyze_split(df, tgt_name, args.sample, rng)

        if not args.no_tokenization and (mt5_tok is not None or nllb_tok is not None):
            print("  tokenization stats ...")
            dres["tokenization"] = analyze_tokenization(splits, tgt_name, mt5_tok, nllb_tok)

        # ---- contamination ----
        print("  train/val/test similarity ...")
        tr, va, te = splits["train"], splits["val"], splits["test"]
        train_grams = hashed_4grams(tr["hi"].tolist())
        contamination = {
            "overlaps": [
                overlap_block(te, tr, "test_vs_train"),
                overlap_block(va, tr, "val_vs_train"),
                overlap_block(va, te, "val_vs_test"),
            ],
            "test_in_train_4gram": containment_4gram(te["hi"].tolist(), train_grams),
            "test_vs_train_vocab": vocab_oov(te["hi"].tolist(), tr["hi"].tolist()),
        }
        if args.near_dup_sample > 0:
            print("  near-duplicate (tfidf) ...")
            contamination["test_vs_train_near_dup"] = near_dup_tfidf(
                tr["hi"].tolist(), te["hi"].tolist(), args.near_dup_sample, rng)
        dres["contamination"] = contamination

        for ob in contamination["overlaps"]:
            leak_rows.append({"dataset": name, **ob})

        report["datasets"][name] = dres
        hindi_sets[name] = set(pd.concat([splits[s]["hi"] for s in splits]).tolist())

        # ---- summary rows ----
        for split, sr in dres["splits"].items():
            tok = dres.get("tokenization", {})
            def tg(path_tk, path_split, side, key, default=None):
                try:
                    return tok[path_tk][path_split][side][key]
                except Exception:
                    return default
            tl = tgt_name.lower()
            summary_rows.append({
                "dataset": name, "split": split,
                "rows": sr["n_rows"],
                "hi_chars_mean": sr["length"]["hindi_chars"].get("mean"),
                "hi_words_mean": sr["length"]["hindi_words"].get("mean"),
                "tgt_chars_mean": sr["length"][f"{tl}_chars"].get("mean"),
                "tgt_words_mean": sr["length"][f"{tl}_words"].get("mean"),
                "tgt_src_ratio_median": sr["length"]["tgt_over_src_char_ratio"].get("median"),
                "identical_pct": sr["identical_pairs"]["exact_pct"],
                "near_identical_ge90_pct": sr["identical_pairs"]["char_sim_ge_0.90_pct"],
                "src_tgt_chrf_mean": sr["identical_pairs"]["src_tgt_sentence_chrf_mean"],
                "dup_source_pct": sr["duplicates"]["dup_source_rows_pct"],
                "dup_pair_pct": sr["duplicates"]["dup_pair_rows_pct"],
                "tgt_latin_gt30_pct": sr["language_quality"]["target_latin_gt_30pct_rows_pct"],
                "tgt_no_deva_pct": sr["language_quality"]["target_no_devanagari_rows_pct"],
                "numeric_mismatch_pct": sr["language_quality"]["src_tgt_numeric_mismatch_pct"],
                "mt5_hi_subw_mean": tg("mt5", split, "hindi", "subword_len", {}).get("mean") if tg("mt5", split, "hindi", "subword_len") else None,
                "mt5_tgt_subw_mean": tg("mt5", split, tl, "subword_len", {}).get("mean") if tg("mt5", split, tl, "subword_len") else None,
                "mt5_tgt_fertility": tg("mt5", split, tl, "fertility_subw_per_word_mean"),
                "mt5_tgt_over128_pct": tg("mt5", split, tl, "over_128_tok_pct"),
                "nllb_tgt_fertility": tg("nllb", split, tl, "fertility_subw_per_word_mean"),
                "nllb_tgt_over128_pct": tg("nllb", split, tl, "over_128_tok_pct"),
                "nllb_tgt_unk_pct": tg("nllb", split, tl, "unk_rate_pct"),
            })

    # ---- cross-dataset Hindi overlap ----
    if len(hindi_sets) > 1:
        names = list(hindi_sets)
        cross = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = hindi_sets[names[i]], hindi_sets[names[j]]
                inter = len(a & b)
                cross[f"{names[i]}__vs__{names[j]}"] = {
                    "hindi_sentences_a": len(a),
                    "hindi_sentences_b": len(b),
                    "shared_exact": inter,
                    "jaccard": round(inter / len(a | b), 6) if (a or b) else 0.0,
                    "pct_of_a_shared": pct(inter, len(a)),
                    "pct_of_b_shared": pct(inter, len(b)),
                }
        report["cross_dataset_hindi_overlap"] = cross

    (out_dir / "corpus_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leak_rows).to_csv(out_dir / "leakage.csv", index=False, encoding="utf-8-sig")
    print(f"\n[done] wrote:\n  {out_dir/'corpus_analysis.json'}\n  {out_dir/'summary.csv'}\n  {out_dir/'leakage.csv'}")


if __name__ == "__main__":
    main()
