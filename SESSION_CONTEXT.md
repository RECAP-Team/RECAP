# RECAP Project — Full Session Context

> Snapshot written 2026-08-23. This file is a complete dump of everything
> discussed/built/debugged in this Claude Code session, so a future session
> (or a human) can pick up with zero prior context. It covers: what RECAP is,
> the full repo layout, every design decision and why, the complete code-review
> and bug-fix history, the tqdm-instrumentation pass, and the entire HPC
> cluster environment-setup debugging saga (with exact commands).

---

## 1. What RECAP is

**RECAP = REward-Calibrated Alignment for Preference optimization.** A research
paper implementation (see `RECAP.pdf` in repo root, `IMPLEMENTATION_PLAN.md`
for the full design doc) that fine-tunes NMT models via preference optimization
(DPO / GRPO / PPO) for three low-resource Indian languages translating to/from
Hindi, run on the **Pragya HPC cluster** (IIT Delhi).

- **Languages:** Bhili, Gondi, Mundari
- **Directions:** `hi2tgt` (Hindi → target language), `tgt2hi` (target → Hindi)
- **Candidate/reference models:** NLLB, mT5, Qwen, Llama (4 models whose outputs
  form the preference-pair candidate pool)
- **Backbone actually fine-tuned by RECAP:** mT5 only — chosen ONCE via
  macro-averaged validation BLEU/ChrF++ across all 6 (lang × direction)
  combinations (paper Table 6 / `build_table_6()`), even though individual
  directions may have a different single best model (e.g. paper shows Llama
  wins Hi→Bhili specifically). This is a deliberate controlled-experiment
  design: it isolates "does the RECAP *method* help" from "which architecture
  is best," and all 4 models' outputs still feed into the DPO candidate pool
  regardless of which one is the trained backbone.
- **Core method:** direction-isolated (θ_d ∩ θ_d' = ∅ — 6 independent
  checkpoints, one per lang×direction, never shared weights) preference
  optimization using a **calibrated multi-aspect reward** (quality via
  BLEU+ChrF+++COMET, minus repetition and length-mismatch penalties),
  confidence-margin-filtered + balanced DPO pairs mined from the 4-model
  candidate pool, with an optional GRPO refinement stage on top of DPO, plus
  an independent PPO baseline.

---

## 2. Repository layout

```
RECAP/                                  (git repo, remote origin/main,
                                          github.com/sagarvaibhav-2805/RECAP)
├── AGENTS.md                           agent-agnostic instructions (Karpathy-style:
│                                        think before coding, simplicity first,
│                                        surgical changes, declarative execution)
├── CLAUDE.md                           one-liner: `@AGENTS.md` (Claude Code inherits AGENTS.md)
├── IMPLEMENTATION_PLAN.md              the ORIGINAL design doc — full stage-by-stage
│                                        spec, central-config spec, "Deadlock-free DDP"
│                                        section, reproducibility section. Cross-referenced
│                                        extensively against RECAP.pdf during code review.
├── RECAP.pdf                           the actual research paper being implemented
├── build_maha_data.py                  PRE-RECAP data-prep script: adds a COMET_<model>
│                                        column to each model's prediction in the
│                                        Maha_data/ comparison CSVs (BLEU/chrF++ already
│                                        present) -> writes maha_data_2/ (18 cols).
│                                        Not part of the 19-file RECAP pipeline itself,
│                                        but its OUTPUT (maha_data_2/) is Stage 1's INPUT.
├── reorder_maha_data_2.py              utility, column-reordering for maha_data_2/
├── mt5_finetune/<Lang>/mt5-<lang>-<direction>-finetuned/
│                                        the pre-existing, already-trained mT5 SFT
│                                        checkpoints RECAP starts every DPO/PPO/GRPO run
│                                        from. RECAP NEVER retrains these. No direction
│                                        subfolder (unlike llama_finetune below) —
│                                        direction is baked into the folder name itself.
│                                        See config.sft_checkpoint_path().
├── llama_finetune/<Lang>/...           Llama candidate-model finetune outputs (DOES use
│                                        a direction subfolder, unlike mt5_finetune)
├── NLLB-finetune/<Lang>/...            NLLB candidate-model finetune outputs
├── qwen_finetune/<Lang>/...            Qwen candidate-model finetune outputs
├── BERT-frozen-MLM_only/               (unrelated/earlier experiment, not part of RECAP)
│
├── maha_data_2/<lang>/<direction>.csv  RECAP Stage 1 INPUT — one row per source sentence,
│                                        columns: source, gold_truth, plus per-model
│                                        prediction/BLEU/chrF++/COMET columns (4 models x
│                                        4 metrics = 16 cols + source + gold_truth = 18)
│
└── code/                               *** THE RECAP PIPELINE — 19 Python files ***
    ├── config.py                       single source of truth (see §3 below)
    ├── recap_split.py                  Stage 1 — train/val/test split
    ├── recap_calibrate.py              Stage 2 — fit RewardEngine calibration stats
    ├── recap_score.py                  Stage 3 — compute raw per-candidate metrics
    ├── recap_reward.py                 RewardEngine class (used by many stages)
    ├── recap_preference.py             PreferenceBuilder class (used by Stage 4)
    ├── recap_mine_pairs.py             Stage 4 — build DPO preference pairs
    ├── recap_balance_pairs.py          Stage 5 — balance pairs across type/margin bins
    ├── recap_train_dpo.py              Stage 6 — DPO training (TRL DPOTrainer)
    ├── recap_train_grpo.py             Stage 7 — GRPO training (hand-rolled, NOT TRL)
    ├── recap_train_ppo.py              Stage 8 — PPO training (TRL classic PPOTrainer)
    ├── recap_evaluate.py               Stage 9 — test-set evaluation + deltas + CI
    ├── recap_infer.py                  Stage 10 — single reusable inference entrypoint
    ├── recap_checks.py                 9 pre-flight/runtime correctness checks
    ├── recap_utils.py                  shared helpers (seeding, DDP, resume, manifest)
    ├── recap_report_tables.py          Stage 11 — builds Tables 3-9 (CSV)
    ├── recap_report_plots.py           Stage 11 — builds Figures 1-4 (PNG)
    ├── recap_sample_for_human_eval.py  Stage 11 — samples pairs for human annotation
    ├── recap_human_eval_analysis.py    Stage 11 — Table 10 stats from labeled human eval
    ├── run_all_experiments.sh          sequential reference script (superseded by
    │                                    JOB_GUIDE.md for actual cluster use, kept for
    │                                    reference, header points to JOB_GUIDE.md)
    ├── JOB_GUIDE.md                    comprehensive HPC job-orchestration guide (§7 below)
    └── delete.py                       (stray/scratch file, 18 bytes, not part of pipeline)

Data/output directories (created at runtime by the pipeline, gitignored):
    recap_splits/<lang>/<direction>/{train,val,test}.csv, manifest.csv, rejected_rows.csv
    recap_calib/<lang>/<direction>/stats.json
    recap_rewards/<lang>/<direction>/rewards.csv
    recap_pairs/<lang>/<direction>/<experiment>/{pairs_all,pairs_balanced}.csv
    recap_dpo/<lang>/<direction>/<experiment>/seed_<N>/{checkpoint/, run_manifest.json, latest_state/}
    recap_grpo/...                    (same shape as recap_dpo/)
    recap_ppo/...                     (same shape as recap_dpo/)
    recap_eval/<lang>/<direction>/<experiment>/seed_<N>/report.json
    recap_eval/<lang>/<direction>/best_checkpoint.json
    recap_report/tables/table{3_5_<lang>,6,7,8,9,10}.csv
    recap_report/plots/figure1/<experiment>.png, figure2/<experiment>.png,
                       figure3/<experiment>.png, figure4_main_matrix_summary.png
    recap_human_eval/sample_<experiment>_<n>.csv
```

---

## 3. `config.py` — central config (single source of truth)

No stage script hardcodes a language/direction/model/path/hyperparameter — everything
flows from `config.py`. Key contents (exact values, confirmed by reading the file):

```python
LANGUAGES = ["Bhili", "Gondi", "Mundari"]
DIRECTIONS = ["hi2tgt", "tgt2hi"]
MODEL_NAMES = ["nllb", "mt5", "qwen", "llama"]

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLIT_SEED = 13
SEED = 13
SEEDS = [13, 42, 2026]           # for multi-seed runs (78 jobs -> 234)

CALIBRATION_QUANTITIES = ["bleu", "chrf", "comet", "rep", "len"]
REPETITION_NGRAM_SIZE = 4         # word n-grams, for P_rep
LENGTH_METRIC = "char"

MARGIN_DELTA = 0.10               # confidence-margin threshold for pair filtering
BALANCE_CAP_PER_TYPE = None       # None = no cap
BALANCE_MARGIN_BINS = 5
MAX_PAIRS_PER_SOURCE = 6          # C(4,2) unordered pairs per source, natural cap

DDP_BACKEND = "nccl"
DDP_TIMEOUT_SECONDS = 1800
DDP_ENV = {"NCCL_ASYNC_ERROR_HANDLING": "1"}
```

### Path root & helpers
`ROOT = Path(__file__).resolve().parent.parent` (i.e. the RECAP repo root, one
level up from `code/`). All stage I/O paths are built from `ROOT` via helper
functions — `maha_data_2_csv()`, `split_dir()`, `calib_path()`, `rewards_path()`,
`pairs_dir()`, `pairs_experiment_dir()`, `checkpoint_dir()`, `run_manifest_path()`,
`eval_report_path()`, `sft_checkpoint_path()`, `best_checkpoint_path()` — see the
directory tree in §2 for what each produces.

`sft_checkpoint_path(lang, direction)` →
`mt5_finetune/<Lang>/mt5-<lang-lower>-<direction>-finetuned` (no direction
subfolder — this matches the naming convention already used on the Pragya
server; different from `llama_finetune`, which does use a subfolder).

### Reward presets (`RewardConfig` dataclass)
Two groups, merged into `REWARD_PRESETS = {**PREFERENCE_ABLATIONS, **MAIN_MATRIX_REWARDS}`:

**(a) `PREFERENCE_ABLATIONS`** — 6-step cumulative ablation (paper §8.3, Table 8).
None use margin filtering until the last two steps — that's the point, it
isolates each reward component one at a time:
1. `ablation_quality_only` — w=(1.0, 0, 0), no margin, no balance
2. `ablation_quality_plus_rep` — w=(0.75, 0.25, 0), no margin, no balance
3. `ablation_quality_plus_len` — w=(0.75, 0, 0.25), no margin, no balance
4. `ablation_full_reward` — w=(0.60, 0.25, 0.15), no margin, no balance
5. `ablation_full_reward_margin` — same weights, **+margin filter**, no balance
6. `ablation_full_recap` — same weights, **+margin filter +balanced sampling**

**(b) `MAIN_MATRIX_REWARDS`** — 4 variants for the main 8-condition matrix
(paper §11.9, Table 7). Note `dpo_quality_only` here is NOT the same config as
`ablation_quality_only` above — deliberate paper distinction (this one keeps
margin filter + balancing on):
- `dpo_raw` — use_calibration=False, w=(1.0,0,0), no margin, no balance
- `dpo_quality_only` — calibrated, w=(1.0,0,0), **margin+balance ON**
- `dpo_no_confidence` — calibrated, w=(0.60,0.25,0.15), **no margin**, balance ON
- `recap_dpo` — calibrated, w=(0.60,0.25,0.15), **margin+balance ON** (= full method)

### Experiment registry (`Experiment` dataclass, `EXPERIMENTS` dict)
```python
EXPERIMENTS = {
    "sft":               Experiment(trainer="sft"),
    "dpo_raw":           Experiment(trainer="dpo", reward_preset="dpo_raw"),
    "dpo_quality_only":  Experiment(trainer="dpo", reward_preset="dpo_quality_only"),
    "dpo_no_confidence": Experiment(trainer="dpo", reward_preset="dpo_no_confidence"),
    "recap_dpo":         Experiment(trainer="dpo", reward_preset="recap_dpo"),
    "sft_ppo":           Experiment(trainer="ppo"),
    "sft_grpo":          Experiment(trainer="grpo", init_from=None),        # "Pure GRPO"
    "recap_dpo_grpo":    Experiment(trainer="grpo", init_from="recap_dpo"), # headline result
    # + 6 more: "ablation_*" (one per PREFERENCE_ABLATIONS key), trainer="dpo"
}
# = 14 total experiments
MAIN_MATRIX_ORDER = ["sft","dpo_raw","dpo_quality_only","dpo_no_confidence",
                      "recap_dpo","sft_ppo","sft_grpo","recap_dpo_grpo"]   # 8 conditions
PREFERENCE_ABLATION_ORDER = [...]  # the 6 ablation_* keys, in order
```

**GRPO semantics** (this came up early in the session as a clarifying Q&A):
- `sft_grpo` ("Pure GRPO"): policy AND frozen KL-reference **both start from
  the SFT checkpoint**. Every rollout is fresh generation from the *current*
  policy — it is genuinely on-policy RL, sampling different outputs from the
  same evolving policy each update, not reading static candidate files.
- `recap_dpo_grpo` (headline result): policy starts from the **`recap_dpo`
  checkpoint**, and the frozen KL-reference is a frozen copy of that SAME DPO
  checkpoint (not SFT) — GRPO refines on top of DPO's output.

**"NOT YET WIRED" (deliberately deferred, per explicit user instruction "Sep
scope")**: `GRPO_DATA_SIZE_ABLATION = [10_000, 25_000, 50_000]`,
`GRPO_GROUP_SIZE_ABLATION = [2, 3]`, and `Experiment.grpo_group_size` field
all exist in config.py but are not consumed anywhere — building this out
would need per-preset GRPO_SETTINGS overrides, a swept MARGIN_DELTA per
experiment, and new EXPERIMENTS/REWARD_PRESETS entries. This is one of **8
total targeted ablations** from IMPLEMENTATION_PLAN.md Central-Config #10
that are out of scope for the current 19-file pipeline (the other 6:
candidate-diversity, calibration-method, confidence/delta-grid, pair-selection-
strategy, weight-sensitivity — see `recap_report_tables.py`'s `build_table_9()`
docstring, which also flags this explicitly rather than silently returning
nothing).

### Training-settings dataclasses (real + smoke-test pairs)

```python
DPOSettings: beta=0.1, lr=5e-6, per_device_train_batch_size=8,
    gradient_accumulation_steps=4, num_train_epochs=3, max_length=256,
    max_prompt_length=128, warmup_ratio=0.1, max_grad_norm=1.0,
    eval_steps=500, save_steps=500, use_confidence_weighting=False
DPO_SETTINGS = DPOSettings()
DPO_SETTINGS_SMOKE_TEST = DPOSettings(num_train_epochs=1, eval_steps=5, save_steps=5)

GRPOSettings: group_size=2, lr=1e-6, kl_coef=0.05, clip_epsilon=0.2,
    temperature=1.0, top_p=0.95, max_new_tokens=128, num_updates=2000,
    per_device_batch_size=4, eval_steps=200, save_steps=200,
    source_subset_size=50_000
GRPO_SETTINGS = GRPOSettings()
GRPO_SETTINGS_SMOKE_TEST = GRPOSettings(num_updates=10, eval_steps=5, save_steps=5,
    source_subset_size=200)

PPOSettings: lr=1e-6, init_kl_coef=0.05, cliprange=0.2, cliprange_value=0.2,
    vf_coef=0.1, gamma=1.0, lam=0.95, batch_size=32, mini_batch_size=4,
    ppo_epochs=4, max_new_tokens=128, num_updates=2000, eval_steps=200,
    save_steps=200
PPO_SETTINGS = PPOSettings()
PPO_SETTINGS_SMOKE_TEST = PPOSettings(num_updates=10, eval_steps=5, save_steps=5,
    batch_size=8, mini_batch_size=2)
```

Smoke-test objects are **separate named config objects**, not hand-edits of the
real settings — nothing to remember to revert before a real run. Selected via
each trainer's `--smoke_test` CLI flag.

`InferenceConfig`: `strategy="greedy"`, `num_beams=1`, `max_new_tokens=128`
(shared by `recap_infer.py` and `recap_evaluate.py`, so eval and deployment
never drift apart).

---

## 4. Pipeline stages, one by one

| # | Script | Purpose | Key I/O |
|---|--------|---------|---------|
| 1 | `recap_split.py` | Source-level train(80%)/val(10%)/test(10%) split, dedup-aware (exact normalized-source duplicates stay in one split together), reject-and-log invalid rows (never silently drop) | in: `maha_data_2/<lang>/<direction>.csv`; out: `recap_splits/<lang>/<direction>/{train,val,test,manifest,rejected_rows}.csv` |
| 2 | `recap_calibrate.py` | Fits `RewardEngine` z-score calibration stats (μ/σ per quantity) from **train split only** | out: `recap_calib/<lang>/<direction>/stats.json` |
| 3 | `recap_score.py` | Computes RAW per-candidate metrics (bleu/chrf/comet/rep/len/valid) for every row × every model, long format, so Stage 4/5 never re-touch COMET under different reward presets | out: `recap_rewards/<lang>/<direction>/rewards.csv` |
| 4 | `recap_mine_pairs.py` | Builds DPO preference pairs from the 4-model candidate pool via `PreferenceBuilder`, one reward-preset/experiment at a time (different presets = non-interchangeable pair files) | out: `recap_pairs/<lang>/<direction>/<experiment>/pairs_all.csv` |
| 5 | `recap_balance_pairs.py` | Balances pairs across generator-pair-type AND reward-margin bins; reports per-model chosen/rejected participation (pathology-collapse check) | out: `.../pairs_balanced.csv` |
| 6 | `recap_train_dpo.py` | DPO training via TRL's `DPOTrainer`. π_θ and π_ref both start from the SFT checkpoint (never retrained). Checkpoint selection by validation composite (BLEU+ChrF+++COMET)/3, not training loss | out: `recap_dpo/<lang>/<direction>/<experiment>/seed_<N>/checkpoint/`, `run_manifest.json` |
| 7 | `recap_train_grpo.py` | **Hand-rolled** GRPO (NOT TRL's GRPOTrainer — that targets causal LMs, mT5 is encoder-decoder). Single inner epoch per rollout batch so importance ratio is exactly 1 at the gradient step (reduces clipped objective to plain KL-regularized policy gradient) | out: `recap_grpo/...` (same shape as DPO) |
| 8 | `recap_train_ppo.py` | PPO baseline via TRL's **classic** `PPOTrainer` API (`AutoModelForSeq2SeqLMWithValueHead` + `PPOConfig` + `PPOTrainer(config, model, ref_model, tokenizer)` + `.step()`) — independent RL baseline, starts from SFT (not DPO), same fixed reward as DPO/GRPO, no separate reward model | out: `recap_ppo/...` (same shape as DPO) |
| 9 | `recap_evaluate.py` | Test-set evaluation via `recap_infer.translate()` (same code path as deployment). Reports **true corpus-level** BLEU/ChrF++ (`sacrebleu.corpus_bleu/corpus_chrf`, not mean-of-sentence-scores) + segment-mean COMET (correct convention there) + repetition/length diagnostics + paired-bootstrap CI deltas vs SFT (aligned by `source_id`, not list position). Also picks each direction's deployed checkpoint by validation-time composite (never peeks at test set) | out: `recap_eval/<lang>/<direction>/<experiment>/seed_<N>/report.json`, `recap_eval/<lang>/<direction>/best_checkpoint.json` |
| 10 | `recap_infer.py` | Single reusable `translate()` entrypoint — used by CLI, by `recap_evaluate.py`, and (via `recap_utils.generate_batch` directly) by training-time validation callbacks. Resolves checkpoint via `--checkpoint` / `--experiment` / or the best-checkpoint registry | CLI: `--source "..."` or `--input_csv/--output_csv` |
| 11 | `recap_report_tables.py`, `recap_report_plots.py`, `recap_sample_for_human_eval.py`, `recap_human_eval_analysis.py` | Builds Tables 3-9 (CSV, from already-saved outputs only, never retrains/re-decodes), Figures 1-4 (PNG), samples pairs for human annotation, computes Table 10 stats (agreement rate, Spearman ρ, chosen-win-rate) once labels come back | out: `recap_report/tables/*.csv`, `recap_report/plots/**/*.png`, `recap_human_eval/sample_*.csv` |

`recap_checks.py` (9 pre-flight/runtime checks, wired into every stage/trainer's
`main()`/`process_one()`): reward-weights-sum-to-1, penalty-monotonicity,
calibration-finite, pairs-valid, split-leakage, reference-frozen (KL-ref model
never has `requires_grad=True` params), live-generation (data-driven: no-all-
identical-completions + no-copy-through-of-source — NOT a hardcoded-string
tautology), rollout-alignment, manifest-hashed.

`recap_utils.py` shared helpers: `set_seed()`, `is_main_process()`,
`wait_for_everyone()`, `configure_ddp_env()`, `make_accelerator()` (applies
`cfg.DDP_TIMEOUT_SECONDS` via `InitProcessGroupKwargs`), `config_hash()`,
`generate_batch()` (shared seq2seq generation used everywhere, restores
`model.train(was_training)` in a `finally` block), `save_training_state_meta()`/
`load_training_state_meta()` (resume support), `save_run_manifest()`.

---

## 5. Key design decisions (the "why")

- **Direction isolation**: 6 completely independent checkpoints (θ_d ∩ θ_d' = ∅)
  — one per (lang, direction) pair — never shared weights across directions.
- **mT5 as the ONE common backbone** across all 6 directions, selected by
  macro-averaged validation score (Table 6), NOT per-direction "pick whichever
  model wins individually" — this was explicitly discussed/pushed-back-on
  early in the session; the answer is it's an intentional controlled-experiment
  choice (isolates method effect from architecture effect), and all 4 models'
  outputs (including non-selected ones) still feed the DPO candidate pool.
- **Reward calibration is direction-specific**: μ/σ stats fit ONLY from that
  direction's own train-split candidates (`recap_calib/<lang>/<direction>/stats.json`)
  — never shared/pooled across directions.
- **`RewardEngine` split into expensive vs. cheap paths**: `compute_raw_metrics()`
  (sacrebleu + COMET, expensive) vs `score_raw_rows()` (standardize + weight
  only, cheap, no metric recompute) — lets the 10 reward-preset ablations all
  reuse Stage 3's cached raw metrics without ever re-touching COMET.
- **Config-driven, plug-and-play**: adding a new language/direction/ablation
  is a `config.py` entry, never a code change elsewhere — every stage script
  takes `--lang --direction --experiment` CLI args (matching `build_maha_data.py`'s
  established single-script-many-jobs pattern).
- **Corpus-level metrics, not mean-of-sentence-scores**: `sacrebleu.corpus_bleu`/
  `corpus_chrf` for BLEU/ChrF++ (mean-of-sentence is a well-known MT-eval
  malpractice sacrebleu itself warns against); COMET correctly stays
  segment-mean (that IS the standard "corpus COMET" convention).
- **Paired bootstrap CI aligned by `source_id`**, not list position — two
  conditions can flag different rows invalid, so a positional zip would
  silently misalign.
- **Invalid/garbage RL rollouts are EXCLUDED, not reward-substituted**: in
  GRPO/PPO, a completion failing validity checks is dropped from the
  advantage/loss computation entirely — NOT given `reward=0.0`. Rationale:
  calibrated rewards can be negative for legitimate-but-weak translations, so
  a fabricated 0.0 could look "better" than a genuine weak output and teach
  the policy to degenerate (reward-hacking risk).
- **Rank-aware sampling under DDP**: GRPO/PPO seed each rank's mini-batch
  sample with `seed + update*1000 + accelerator.process_index` — without the
  `process_index` offset, every DDP rank would sample the IDENTICAL batch,
  defeating data-parallelism (each extra GPU would just redo the same work).
- **DDP timeout actually applied**: `recap_utils.make_accelerator()` wraps
  `Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.DDP_TIMEOUT_SECONDS))])`
  — a bare `Accelerator()` silently ignores `cfg.DDP_TIMEOUT_SECONDS` and falls
  back to torch's 30-min default, defeating the "deadlock fails fast" goal.
  DPO uses `DPOConfig(ddp_timeout=cfg.DDP_TIMEOUT_SECONDS)`; PPO uses
  `PPOConfig(accelerator_kwargs={"kwargs_handlers": [InitProcessGroupKwargs(...)]})`.
- **Frozen reference models never `accelerator.prepare()`'d** — see
  IMPLEMENTATION_PLAN.md "Deadlock-free DDP" section; only the trainable
  policy goes through `accelerator.prepare()`.
- **Reproducibility**: every checkpoint gets a `run_manifest.json` (git commit
  hash, library versions, GPU info, seed, resolved config + `config_hash()`
  for drift detection); `recap_utils.set_seed()` seeds Python/NumPy/Torch/
  transformers consistently.

---

## 6. Code-review & bug-fix history (this project)

The user ran **"You are a senior RL Researcher and AI Scientist. Please do a
code review..."** verbatim **3 separate times** across the project's history
(each a fresh, rigorous pass finding new issues), plus one explicit
resolve-and-fix-errors pass. Cumulative fixes across all rounds:

1. DPO validation callback existed but was only ever called once AFTER
   `trainer.train()` finished (never mid-training) — fixed by wiring a real
   `transformers.TrainerCallback.on_step_end` adapter via `trainer.add_callback()`.
2. The 9 pre-flight checks in `recap_checks.py` were dead code (defined,
   never called except in one comment) — wired into every stage/trainer script.
3. GRPO/PPO invalid-rollout reward-hacking risk (see §5 above) — fixed via
   exclusion instead of substitution.
4. Paired-bootstrap CI misalignment (positional zip vs `source_id`-aligned) — fixed.
5. "Corpus BLEU" was actually mean-of-sentence-BLEU — fixed with real
   `sacrebleu.corpus_bleu`/`corpus_chrf`.
6. `generate_batch()` left the model permanently in `.eval()` mode after the
   first mid-training validation call (dropout silently disabled for the rest
   of training) — fixed with `was_training = model.training` + `finally: model.train(was_training)`.
7. DDP data-parallelism defeated by identical-seed sampling across ranks — fixed
   with rank-aware seeding (see §5).
8. COMET was being called on invalid/garbage candidates BEFORE the validity
   check (crash/waste risk) — fixed by filtering to `valid_idx` first.
9. Extreme-length validity check was an absolute character cap, not
   reference-relative — fixed with `EXTREME_LENGTH_RATIO = 10.0` (ratio-based)
   + `EXTREME_LENGTH_CHARS_ABSOLUTE = 4000` (backstop).
10. Reward-tie edge case (`delta == 0`) could form a spurious zero-margin pair
    — fixed with an explicit `if delta == 0: continue` guard.
11. DDP timeout declared in config but never actually applied anywhere — fixed
    via `recap_utils.make_accelerator()` (see §5).
12. `check_live_generation` was a tautology (hardcoded string matching exactly
    what it checked against, could never fail) — fixed to take real
    `(sources, completions)` and check two real signals (no-all-identical,
    no-copy-through-of-source).
13. DPO's fallback validation call (the "make sure at least one validation
    happened" safety net) wasn't `is_main_process()`-gated — every DDP rank
    would redundantly run COMET validation on short runs — fixed.
14. Dead `field` import in `config.py` — removed.
15. 8-targeted-ablations infrastructure (`GRPO_DATA_SIZE_ABLATION`,
    `GRPO_GROUP_SIZE_ABLATION`, `Experiment.grpo_group_size`) declared but
    never consumed — **user explicitly said "Sep scope"** — left as
    documented "NOT YET WIRED" future work, not implemented.
16. Figure 3 (delta heatmap) colormap bug: float-precision noise
    (`0.35-0.30 != 0.55-0.50` by ~1e-17 in float64) stretched across the full
    colormap range, making three visually-identical `+0.05` deltas render as
    red/red/green — fixed with `matplotlib.colors.TwoSlopeNorm(vcenter=0.0)`.
17. Resume-vs-"already done" check conflict: original logic
    (`if checkpoint_dir.exists() and any(iterdir()): skip`) would wrongly skip
    a job that crashed mid-run but had already saved one partial best
    checkpoint — fixed by switching the skip-check to `run_manifest.json`
    existing (only written on full success) across all three trainers.
18. Missing `Path` import in `recap_infer.py` (needed for the tqdm `desc=`
    label logic) — added.

All of these are committed on `main` (see §11 git log).

---

## 7. Resume system

Three structurally different trainers needed reconciling:

- **DPO** (`recap_train_dpo.py`): HF `Trainer`'s native
  `resume_from_checkpoint` against `trainer_state/checkpoint-*`, PLUS
  re-evaluates any existing best-checkpoint on resume to reseed
  `val_callback.best_composite` (so a resumed run doesn't start comparing
  against `-inf` again and overwrite a genuinely-better earlier checkpoint).
- **GRPO / PPO** (hand-rolled loops): `accelerator.save_state()`/`load_state()`
  (DDP-safe — accelerate handles unwrapping/gathering) into
  `<stage_root>/.../latest_state/accelerate/`, plus a small JSON via
  `recap_utils.save_training_state_meta()`/`load_training_state_meta()`
  recording `{"step": ..., "best_composite": ...}`, saved every `save_steps`,
  **deleted on successful completion** (it can be sizeable — full optimizer
  state — no need to keep it once `run_manifest.json` is written).
- **Completion check** (all three trainers): judged by **`run_manifest.json`
  existing** (written only at the very end of a successful run), NOT by
  `checkpoint_dir` having files — a resumed run legitimately has a partial
  best-checkpoint saved mid-flight while still genuinely in progress.

---

## 8. `torchrun` / multi-GPU support

All three trainers are `accelerate`-based (TRL's `DPOTrainer`/`PPOTrainer`,
our own GRPO loop uses `recap_utils.make_accelerator()`) and **auto-detect
`torchrun`'s distributed env vars — no code changes were needed** to support
multi-GPU. Confirmed and wrapped all 78 training-job commands with
`torchrun --standalone --nproc_per_node=$NGPUS_PER_JOB ...` in `JOB_GUIDE.md`.

**Stage 9 (`recap_evaluate.py`) explicitly does NOT support `torchrun`** — no
DDP sharding is built into it, so it should always be run as plain `python`.

---

## 9. `JOB_GUIDE.md` (in `code/`) — HPC job-orchestration guide

The most heavily-iterated document this session. Structure:
- Intro: `export NGPUS_PER_JOB=2` (or however many GPUs per job), torchrun-
  applicability note.
- §0 dependency graph between stages.
- §1 job-groups table.
- §2 Group A — data prep (Stages 1-5), 6 jobs, plain `python` (no DDP benefit).
- §3 Group B — DPO (Stage 6), 60 jobs (6 lang×direction × 10 DPO experiments),
  all `torchrun`.
- §4 Group C — `sft_grpo` (Pure GRPO), 6 jobs, `torchrun`.
- §5 Group D — `recap_dpo_grpo`, 6 jobs, `torchrun`, dependency-noted (needs
  Group B's `recap_dpo` checkpoints first).
- §6 Group E — `sft_ppo`, 6 jobs, `torchrun`.
- §7 PBS wrapping examples (multi-GPU and single-GPU variants,
  `ngpus=$NGPUS_PER_JOB` in the resource request).
- §8 Group F — evaluate (Stage 9), plain `python`, explicitly NOT `torchrun`.
- §9 Group G — reporting (Stage 11 tables/plots).
- §10 Group H — human-eval sampling/analysis.
- §11 resume explanation.
- §12 multi-seed note (78 → 234 jobs across `SEEDS = [13, 42, 2026]`).
- §13 explicitly-deferred 8-targeted-ablations scope note.
- §14 single-GPU `--n_samples` smoke test.
- §14c **"Full end-to-end smoke test — every stage, every experiment/ablation,
  ONE direction, `torchrun`"** — 22 commands, `--smoke_test` +
  `torchrun --nproc_per_node=$NGPUS_PER_JOB`, `NGPUS_PER_JOB=2`.
- §15 switching smoke-test → real run.

`run_all_experiments.sh` (in `code/`) — the earlier sequential reference
script, kept (not deleted) for reference; header now points to `JOB_GUIDE.md`
as the recommended parallel-job approach.

---

## 10. `--n_samples` smoke-test data flag

`recap_split.py --n_samples N` (requires `--lang`/`--direction` together — a
safety guard against accidentally bulk-shrinking every direction): samples N
rows from `maha_data_2` BEFORE splitting. Every downstream stage just reads
whatever's in `recap_splits/`, so shrinking here is enough to make the WHOLE
pipeline fast for a smoke test — no other script needs to change. To go back
to a full real run for the same (lang, direction), delete
`recap_splits/<lang>/<direction>/` first (otherwise the small test split just
gets skipped as "already exists").

---

## 11. tqdm instrumentation pass (most recent substantial code work)

User's request: *"Did you add enough tqdms in the entire code? Please
extensively add tqdms all throughout."* Motivated by earlier, strong
frustration in this project's history with repeating COMET/Lightning terminal
noise (paraphrased: "this keeps repeating, it's driving me crazy, fix it").

**Philosophy adopted**: `TQDM_MIN_ITEMS = 200` threshold constant (duplicated
locally in `recap_reward.py`, `recap_score.py`, `recap_utils.py`) — progress
bars are `disable`d below this size (so GRPO/PPO's tiny per-training-step
reward-scoring calls stay silent) but show real progress for genuinely bulk
operations. Multi-combo `main()` driver loops use `disable=len(combos) < 2`
(silent for a single-job run, visible when driving many combos at once).

**Files touched, what changed:**
- `recap_utils.py` — `generate_batch()` wrapped with tqdm (`show_progress =
  len(sources) >= TQDM_MIN_ITEMS and is_main_process()`), `desc=` parameter added.
- `recap_reward.py` — `compute_raw_metrics()` sacrebleu loop + row-assembly
  loop wrapped, `desc:str|None=None` param threaded through; `fit()` also
  gained a `desc:str="Calibrating"` param that forwards into
  `compute_raw_metrics()`. `score_raw_rows()` deliberately left untouched
  (always small per-source batches, would just be noise).
- `recap_preference.py` — `PreferenceBuilder.build_pairs()`'s main per-source
  loop wrapped unconditionally (`for _, row in tqdm(train_df.iterrows(), ...)`)
  — always bulk, no size threshold needed.
- `recap_split.py` — row-validation loop wrapped with per-(lang,direction)
  `desc`; `main()`'s outer combo loop also wrapped.
- `recap_score.py` — `compute_raw_metrics()` call passes
  `desc=f"{lang}/{direction} [{model}]"`; row-assembly loop wrapped; `main()`
  outer combo loop wrapped.
- `recap_calibrate.py` — `engine.fit(long_df, desc=f"{lang}/{direction}")`;
  `main()` outer combo loop wrapped.
- `recap_infer.py` — added missing `from pathlib import Path`; `translate()`
  now computes `label = experiment or Path(checkpoint_path).name` and passes
  `desc=f"Translating {lang}/{direction} [{label}]"` into `generate_batch()`.
- `recap_evaluate.py` — `evaluate_one()`'s `compute_raw_metrics()` call gets
  `desc=f"{lang}/{direction}/{experiment}"`; `main()`'s nested
  `lang_direction_jobs × experiments` loop (up to 84 combos) flattened into
  one `combos` list and wrapped in tqdm (`select_best_checkpoint()` now runs
  in a separate loop afterward — same net effect, cleaner progress reporting).
- `recap_mine_pairs.py`, `recap_balance_pairs.py` — `main()`'s outer combo
  loops wrapped (the actual heavy work inside `PreferenceBuilder` already had
  its own bar from the earlier edit).
- `recap_train_grpo.py`, `recap_train_ppo.py` — **highest-value fix**: the
  main `for update in range(...)` training loops (previously ZERO progress
  indication, only periodic `print` every 50 steps) wrapped in
  `tqdm(..., initial=start_update, total=settings.num_updates, disable=not
  recap_utils.is_main_process())`, with live `postfix` showing
  loss/kl/reward/n_invalid; all periodic `print()`s inside the loop switched
  to `tqdm.write()` so they don't corrupt the bar; validation-time
  `generate_batch()`/`compute_raw_metrics()` calls got descriptive `desc=`.
- `recap_train_dpo.py` — `ValidationCheckpointCallback` gained a `label: str
  = "DPO"` constructor param; `evaluate()` now builds `desc = f"{self.label}
  val@{step}"` and threads it into both `generate_batch()` and
  `compute_raw_metrics()`. The 3 call sites (`on_step_end` callback, the
  resumed-prior-best re-evaluation, and the end-of-training fallback
  validation) all automatically pick up the label since they all just call
  `val_callback.evaluate(...)`. Instantiation site passes
  `label=f"{lang}/{direction}/{experiment}"`.

**Deliberately left untouched** (too fast/small — pure CSV/JSON aggregation,
sub-second, would just be noise): `recap_report_tables.py`,
`recap_report_plots.py`, `recap_sample_for_human_eval.py`,
`recap_human_eval_analysis.py`, `recap_checks.py`.

All touched files were syntax-checked (`python3 -c "import ast; ast.parse(...)"`)
and import-checked (`importlib.import_module(...)`) successfully; no test
artifacts left behind.

---

## 12. HPC cluster environment-setup debugging saga (Pragya)

This was the last major thread of the session — getting an actual `torchrun`
smoke test to run on the cluster. **Every fix below is confirmed working by
the user's own terminal output**, in this exact order. Cluster identity:
`amilan0XX-eth0.head.cm.pragya.iitd.ac.in`, conda env name `mt5`
(`~/.conda/envs/mt5`, Python 3.10), code checked out at
`~/flash/RECAP/code` (also referenced as `/flash/scai/msr/aiy257590/RECAP/code`).

### Problem 1 — `ModuleNotFoundError: No module named 'trl'`
Root cause: genuinely missing. Fix:
```bash
pip install "trl==0.11.4" --no-deps
```
**Why `--no-deps`**: an earlier unpinned `pip install trl accelerate transformers
datasets unbabel-comet sacrebleu pandas numpy matplotlib tqdm sentencepiece
protobuf` (all together) forced pip's resolver to jointly satisfy all 12
packages at once. `unbabel-comet` pins very old constraints
(`pytorch-lightning==1.6.4`), and the resolver backtracked all the way down to
trying to install `pandas==1.1.5` (2020-era), which failed to build from
source on Python 3.10 (`ModuleNotFoundError: No module named 'pkg_resources'`
during the build-wheel step). Lesson: when only ONE package is actually
missing from an already-working env, install ONLY that one package, and use
`--no-deps` if its transitive deps are already satisfied by what's installed.

**Why `trl==0.11.4` specifically (not latest)**: `recap_train_ppo.py` uses
TRL's **classic** PPO API — `AutoModelForSeq2SeqLMWithValueHead`,
`PPOTrainer(config, model, ref_model, tokenizer)`, `ppo_trainer.step(...)`.
TRL rewrote `PPOTrainer` starting ~v0.12 to be causal-LM-only, dropping
seq2seq-value-head support. `trl==1.10.0` (what unpinned `pip install trl`
resolves to) would have broken PPO entirely. This exact risk was already
flagged in the code's own docstring comments before this was ever tested
("verify this against the exact pinned trl version before the first real run").

### Problem 2 — `torchrun` picks up the WRONG Python interpreter
Symptom: even after `trl` installed successfully into the `mt5` conda env,
`torchrun ... recap_train_dpo.py` still raised `ModuleNotFoundError: No
module named 'trl'`. Root cause (confirmed via `which torchrun` /
`echo $CONDA_PREFIX`): `~/.local/bin/torchrun` (a leftover user-site pip
install, likely from a `pip install --user` done outside the conda env at
some point) shadows the conda env's own `torchrun` in `$PATH` — `~/.local/bin`
resolves before `$CONDA_PREFIX/bin` — and that `~/.local/bin/torchrun` script
launches with `/apps/anaconda3/2025.06/bin/python3.13` (the system base
Anaconda's Python), which is a completely different interpreter without
`trl`/etc installed. A plain `python3 -c "..."` in the same shell worked
correctly because it resolved to the `mt5` env's own `python3.10`.

Fix (immediate, one-off): use the module invocation instead of the shadowed
binary —
```bash
python -m torch.distributed.run --standalone --nproc_per_node=$NGPUS_PER_JOB recap_train_dpo.py ...
```
Fix (permanent, for the shell session / put in `~/.bashrc` after
`conda activate mt5`):
```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
```
This makes bare `torchrun` (as used throughout `JOB_GUIDE.md`) resolve
correctly without editing every command.

### Problem 3 — `ValueError: Unsupported nproc_per_node value: ''`
After the PATH fix, `$NGPUS_PER_JOB` was empty in the new shell (not
re-exported after the PATH-fixing `export` reset context, or simply a fresh
shell). Fix: just re-export it —
```bash
export NGPUS_PER_JOB=2
```

### Problem 4 — `deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)`
Root cause: `trl==0.11.4`'s `trl/models/utils.py` does an **unconditional**
`import deepspeed` at module load (no `try/except`/optional-dependency
guard) as a side effect of importing `unwrap_model_for_generation`, which
`dpo_trainer.py` needs. `deepspeed`'s own `__init__.py` → `ops/__init__.py` →
`git_version_info.py` → `op_builder.builder.installed_cuda_version()` checks
`torch.utils.cpp_extension.CUDA_HOME`, which resolves via the `CUDA_HOME` env
var (or common default paths) — none were set in this shell (GPU driver was
present, but the full CUDA **toolkit** path wasn't configured), so it raised
hard instead of gracefully marking the op incompatible.

Confirmed: **uninstalling `deepspeed` would NOT have fixed this** — the
`import deepspeed` line has no guard, so removing the package just changes
the crash from `MissingCUDAException` to `ModuleNotFoundError`, still fatal.
The actual fix has to make `deepspeed`'s import succeed.

Diagnosis steps used:
```bash
module avail cuda 2>&1 | head -20     # -> cuda/11.6  cuda/11.8  cuda/12.6  cuda/13.0
python3 -c "import torch; print(torch.version.cuda)"   # -> 13.0 (exact match available)
module load cuda/13.0
echo $CUDA_HOME    # <- came back EMPTY, this module doesn't set it
module show cuda/13.0   # revealed: only prepends PATH/LD_LIBRARY_PATH/LD_INCLUDE_PATH,
                         # does NOT set a CUDA_HOME env var itself
which nvcc         # -> /apps/cuda/13.0/bin/nvcc  (confirms the real install root)
```
Fix:
```bash
export CUDA_HOME=/apps/cuda/13.0
```
Permanent version (add to `~/.bashrc`, after `conda activate mt5`):
```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
module load cuda/13.0
export CUDA_HOME=/apps/cuda/13.0
```

### Problem 5 — `TypeError: DPOTrainer.__init__() got an unexpected keyword argument 'processing_class'`
Root cause: our `recap_train_dpo.py` passed `processing_class=tokenizer` (the
newer, transformers-4.46-era kwarg name for what used to be called
`tokenizer=`). `trl==0.11.4` predates that rename in its own `DPOTrainer`
signature, so it doesn't accept `processing_class`.

**Fix applied directly in the codebase** (`recap_train_dpo.py`), forward/
backward-compatible rather than hardcoding one name — added `import inspect`
at the top, and at the `DPOTrainer(...)` call site:
```python
_tokenizer_kwarg = "processing_class" if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters else "tokenizer"
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    **{_tokenizer_kwarg: tokenizer},
)
```
This was committed to `main` by the user (commit `11897eb "error resolved"`)
and pulled onto the cluster via `git pull` (fast-forward `18fd008..11897eb`).
`recap_train_ppo.py` was checked and found NOT to have this risk — its
`PPOTrainer(ppo_config, model, ref_model, tokenizer)` call uses **positional**
args, not a `tokenizer=`/`processing_class=` keyword.

### Problem 6 — `AttributeError: 'generator' object has no attribute 'generate'` (current, UNRESOLVED as of this snapshot)
This is a **method-name collision** between `trl==0.11.4` and
`transformers==4.46.3`, discovered after Problem 5's fix let training
actually start:

- `transformers==4.46.3` added a NEW method to the base `Trainer` class,
  `get_batch_samples(self, epoch_iterator, num_batches)`, called from inside
  `_inner_training_loop` — part of a gradient-accumulation loss-averaging fix
  shipped in that release.
- `trl==0.11.4`'s `DPOTrainer` **already had its own, older, differently-
  purposed method with the exact same name**, `get_batch_samples(self, model,
  batch)` (used to generate sample completions for logging during training) —
  this predates and is unrelated to the new transformers feature, but the
  name collides.
- Because `DPOTrainer` subclasses `Trainer`, when the new transformers core
  loop calls `self.get_batch_samples(epoch_iterator, num_batches)`, Python
  method resolution order dispatches to **trl's old override instead**,
  passing `epoch_iterator` (a Python generator object) into what trl's old
  method treats as its `model` parameter. trl's code then does
  `model.generate(...)` → crashes with `'generator' object has no attribute
  'generate'`.

**Recommended fix (given to the user, not yet confirmed as of this
snapshot)**: downgrade `transformers` below the version that introduced this
`get_batch_samples` collision (4.46.0), since `trl==0.11.4` is pinned for the
PPO-classic-API reason from Problem 1 and shouldn't be casually bumped:
```bash
pip install "transformers==4.45.2" --no-deps
```
Then retry the same `torchrun` smoke-test command. If `4.45.2` surfaces
another incompatibility, the next fallback discussed was to go further back
(e.g. `4.44.2`). **This is the first open/unresolved item — check back on
this before assuming DPO training actually runs end-to-end.**

### Confirmed working version pin (once Problem 6 resolves)
```
trl==0.11.4          (--no-deps installed; needed for classic PPOTrainer API
                       AND to avoid the CUDA_HOME-at-import crash tradeoffs
                       of newer trl/deepspeed combos)
transformers==4.45.x  (or lower; must predate the get_batch_samples clash
                       introduced in 4.46.0)
accelerate==1.13.0    (already satisfied, untouched)
torch==2.13.0 (cu13)  (already satisfied, untouched; torch.version.cuda == "13.0")
CUDA_HOME=/apps/cuda/13.0   (module load cuda/13.0 does NOT set this itself —
                              must be exported manually)
```

### Recommended permanent `~/.bashrc` additions (cluster-side, after `conda activate mt5`)
```bash
export PATH="$CONDA_PREFIX/bin:$PATH"     # so torchrun/pip resolve to the mt5 env,
                                            # not ~/.local/bin's shadow copy
module load cuda/13.0                      # matches torch.version.cuda
export CUDA_HOME=/apps/cuda/13.0           # the cuda module doesn't set this itself
export NGPUS_PER_JOB=2                     # or however many GPUs allocated per job
```

---

## 13. Git history (most recent commits on `main`, newest first)

```
11897eb  error resolved              <- Problem 5 fix (processing_class/tokenizer
                                          inspect-based compat shim in recap_train_dpo.py)
18fd008  tqdm                        <- the full §11 tqdm instrumentation pass
b9c1bb7  added smoketest to job guide.md
92b1a05  added torchrun commands to job guide.md
eaa4cb2  Added smoke tests           <- *_SETTINGS_SMOKE_TEST config objects + --smoke_test flags
8d936c8  updated JOB_GUIDE.md to have jobs
ef57d47  Add training resume support, fix Figure 3 colormap bug, add job-clustering guide
d48e05d  Add RECAP pipeline implementation (config, data stages, DPO/GRPO/PPO training, eval, reporting)
        <- the original 19-file pipeline, after the 3 code-review rounds' fixes
```
Working tree was clean (no uncommitted changes) as of this snapshot — every
edit described in §6/§11/§12 above is already on `origin/main`.

---

## 14. Open items / where to pick back up

1. **[BLOCKING, unresolved]** Confirm Problem 6's fix
   (`pip install "transformers==4.45.2" --no-deps`) actually lets
   `recap_train_dpo.py --smoke_test` complete a full training run on the
   cluster. If it still fails, try further transformers downgrades, or
   consider whether trl needs a different pin entirely.
2. Once DPO smoke test passes end-to-end, run the equivalent smoke tests for
   GRPO (`recap_train_grpo.py --smoke_test`) and PPO
   (`recap_train_ppo.py --smoke_test`) — PPO especially, since it depends on
   the `AutoModelForSeq2SeqLMWithValueHead`/classic-`PPOTrainer` API that
   drove the `trl==0.11.4` pin in the first place; this has not yet been
   exercised at all on the cluster.
3. Consider adding a `requirements.txt` or `environment.yml` pinning the
   now-empirically-verified working versions (`trl==0.11.4`,
   `transformers==4.45.x`, plus whatever else is already satisfied) so this
   entire debugging saga doesn't have to be re-derived on a fresh cluster
   node/env — **not done yet, would need explicit user sign-off first**
   (per AGENTS.md: don't make unrequested changes).
4. 8 targeted ablations (candidate-diversity, calibration-method,
   confidence/delta-grid, pair-selection-strategy, weight-sensitivity,
   GRPO-data-size, GRPO-group-size — 2 of these are GRPO-specific and already
   have unwired `config.py` scaffolding) remain explicitly out of scope
   ("Sep scope") until the user asks for them.
5. Multi-seed runs (`SEEDS = [13, 42, 2026]`, 78 → 234 jobs) are documented in
   `JOB_GUIDE.md` §12 but not yet actually launched at scale.
