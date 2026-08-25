# RECAP — Running Conversation Summary

> Living document. Started 2026-08-23. Meant to be handed to a fresh Claude
> Code chat as a drop-in reference to pick this work back up with zero prior
> context. Updated as the conversation continues — if you're reading this in
> a new session, check the "Last updated" line below and ask for it to be
> refreshed if it looks stale relative to what's actually happened since.

**Last updated:** 2026-08-25 (through the Python-level `logs/` auto-logging rollout across all 15 pipeline scripts)

---

## 1. Quick pickup — current state in one look

- **Repo**: `/home/scai/msr/aiy237528/flash/final-climb-adivaani/RECAP` (also
  reachable as `/flash/scai/msr/aiy237528/final-climb-adivaani/RECAP` — same
  physical location, two paths). Git remote `origin` =
  `https://github.com/RECAP-Team/RECAP.git`. Local repo has **only a
  `master` branch**, tracking `origin/main` directly (no local `main`
  anymore — consolidated on request; nothing changed on the remote side).
- **Paper scope, as of this conversation: Bhili and Mundari only.** Gondi is
  explicitly excluded from the paper (not even mentioned) — decided because
  it's the weakest-performing language in the candidate pool and the user
  chose not to spend the compute/scope on it. `config.py`'s `LANGUAGES` is
  already updated to `["Bhili", "Mundari"]`. Gondi's raw data still exists on
  disk (`dataset/maha_data_2/gondi/`) and was included in the dataset-stats
  analysis "for completeness," but it is out of the paper's scope.
- **Conda env for everything RECAP-related: `recap`** (cloned from a
  general-purpose env called `starkai`, which should NOT be used for RECAP
  work — its `transformers` version conflicts with RECAP's pinned `trl`).
  Activate with `conda activate recap`. Pinned package versions and *why*
  each pin exists are documented in `requirements.txt` at the repo root.
- **Training status**: only **Bhili/hi2tgt** has real trained checkpoints —
  11 of 14 experiments (missing `ablation_full_reward` and
  `ablation_full_recap`). No other (language, direction) pair has been
  trained yet. All of this was done at **smoke-test scale** (1,998 rows,
  ~0.7% of the real 220k-row dataset) purely to validate the pipeline
  end-to-end — **the resulting BLEU/chrF/COMET numbers are not meaningful
  results and should not be cited or read as evidence about RECAP's method.**
- **`config.py` has been substantively changed since the smoke-test runs.**
  Current values differ from what the existing Bhili/hi2tgt smoke-test
  splits/checkpoints were built under — see §2.19. Existing artifacts are
  NOT invalidated (they're just under the old settings); anything trained
  fresh from now on uses the new ones:
  - `SPLIT_RATIOS`: was `{80, 10, 10}`, now `{99.25, 0.25, 0.5}` — a flat
    10% split gives ~22K val/test rows at Bhili/Mundari's real scale, which
    is extremely expensive for training-time validation AND for the new
    paired-bootstrap significance test. New ratios target ~500 val /
    ~1,050-1,100 test rows (FLORES-200 scale).
  - `DPOSettings.max_length`: `256` → `320` (Mundari's real token-length
    tail was close to the old ceiling — verified with the actual mT5
    tokenizer, not char counts).
  - `max_new_tokens` (`InferenceConfig`, `GRPOSettings`, `PPOSettings`):
    `128` → `176` (stored candidate translations were capped at exactly
    128 — a generation artifact, not a true length; real references
    already exceed 128 for ~1% of Mundari rows).
  - `LANGUAGES` is `["Bhili", "Mundari"]` (unchanged from before, restated
    here for completeness).
- **Checkpoint disk usage bug found and fixed** (see §2.21): DPO training
  was keeping *every* HF-Trainer checkpoint forever (no `save_total_limit`)
  — 8 completed smoke-test experiments had accumulated **1.26 TB** between
  them. Fixed (`save_total_limit=1` added) and the stale 1.26TB already
  deleted (verified safe first — all 8 experiments' real deployed
  checkpoints, which live separately, were confirmed untouched).
  GRPO/PPO never had this problem (they already overwrite-in-place and
  clean up on success).
- **A real bug was also found and fixed in `recap_reward.py`** while
  computing genuine z-scored calibration stats (see §2.20): `_is_valid()`
  didn't recognize a NaN candidate (pandas' representation of a blank CSV
  cell) as invalid, only `None` — crashed `_repetition_penalty()` further
  downstream. Real, would-hit-in-production (0.005%–0.175% of candidates
  across the real data, concentrated in Qwen/Llama). Fixed.
- **Reward weights (`w_quality=0.60`/`w_rep=0.25`/`w_len=0.15`) were
  examined against real z-scored calibration stats and deliberately left
  unchanged** — see §2.20 for the full reasoning: length mismatch is a
  common-but-mild, always-active signal; repetition is rare-but-severe and
  only meaningfully activates for Mundari (Mundari's repetition rate is
  5–10× Bhili's). The z-scoring is already doing the right thing;
  reweighting isn't warranted. Concrete, quantified, falsifiable prediction
  now on record for when Table 8 actually runs: **`ablation_quality_plus_rep`
  should show a much larger BLEU/COMET gain for Mundari than for Bhili** —
  if it doesn't, that's worth investigating.
- **Loss/validation curve logging + plotting is now implemented** (see
  §2.22) across all three trainers — this did NOT exist before this
  session and was a real gap against the paper's own §11.11 mandatory-plots
  spec. **This only captures data going forward** — the already-completed
  Bhili/hi2tgt runs predate this feature and have no `train_log.jsonl`/
  `val_log.jsonl`; `recap_report_plots.py`'s new Figures 5/6/7 will silently
  produce nothing for them until something is trained again.
- **The `recap_evaluate.py --lang Bhili --direction hi2tgt` run from §2.17
  did complete successfully** (confirmed via `git status` later showing
  `report.json` + `deltas_vs_sft.json` for all 11 trained experiments, plus
  `recap_report_tables.py`/`recap_report_plots.py` output for all of
  Tables 3-9 and Figures 1-4) — no longer a pending item.
- **Git remote/auth state — important, read before pushing anything**:
  - `origin` now uses **SSH**, not HTTPS:
    `git@github.com:RECAP-Team/RECAP.git`. This cluster blocks raw port 22
    AND direct port 443 — SSH only works because `~/.ssh/config` routes
    `github.com` → `ssh.github.com:443` through the campus HTTP proxy via
    `ncat --proxy ... --proxy-type http` (see §2.24). If SSH ever stops
    working, that's the first thing to check, not GitHub itself.
  - A dedicated key (`~/.ssh/id_ed25519_github`, not the cluster's own
    `id_rsa`) is registered on GitHub account `raja17021998` as
    `pragya-recap-cluster`.
  - **The remote's `main` branch is gone** — deleted at some point outside
    this session (not by me); `RECAP-Team/RECAP` now has only `master`,
    which is also the new default branch. Local `master` tracking was
    fixed to match (`origin/master`). **This may affect teammates** who
    still expect `main` to exist — flagged to the user, not yet resolved
    either way (recreating `main` on the remote is one `git push
    origin master:main` away if wanted).
  - There's an **open PR #1** on the repo (noticed incidentally via
    `git ls-remote`, pointing at commit `312bbab`) — not investigated,
    user was asked if they want it looked into.
  - The broad-scope GitHub PAT that was sitting in
    `flash/pedro/command.txt` in plaintext (`admin:enterprise`,
    `delete_repo`, etc. — far broader than needed) was flagged as a real
    security concern; recommended rotating it. Not rotated as of this
    writing (SSH auth setup means it's no longer needed for routine git
    operations, which reduces but doesn't eliminate the exposure).
- **`JOB_GUIDE.md` fully reordered + de-Gondi'd, smoke-test scripts added**
  (§2.25) — the "known follow-up" about stale Gondi content is now
  resolved. New `smoke_test_bhili.bash`/`smoke_test_mundari.bash` in
  `code/` run the full smoke-test flow for both directions of one language
  in a single command.
- **Known follow-ups not yet done** (offered, not actioned — ask the user
  before doing any of these):
  - `config.py`'s `SEEDS = [13, 42, 2026]` is an unused leftover constant —
    the actual plan uses one training seed (13) per experiment (**52 total
    training jobs**, not 156) and gets uncertainty entirely from paired
    bootstrap CI/p-values at evaluation time, not from retraining.
  - The repetition-rate table computed in §2.20 was offered to be folded
    into `dataset_statistics.csv`/the report artifact — not yet done, ask
    the user if they still want it.
  - `ablation_full_reward` and `ablation_full_recap` still need training for
    Bhili/hi2tgt.
  - Nothing has been trained yet for Mundari at all (any direction/experiment),
    or under the new `SPLIT_RATIOS`/`max_length`/`max_new_tokens` values for
    ANY language — the existing Bhili/hi2tgt smoke-test artifacts all
    predate this config.py revision.

---

## 2. Full chronological narrative

### 2.1 Environment/orientation questions (session start)
- Confirmed CLAUDE.md access: global `~/.claude/CLAUDE.md` imports
  `flash/final-climb-adivaani/AGENTS.md`.
- User asked which GitHub repo `RECAP/` points to →
  `github.com/raja17021998/RECAP.git`. User then asked to switch the remote
  to `github.com/RECAP-Team/RECAP` → done by editing `.git/config` directly
  (no `git` binary existed anywhere in `$PATH` at that point).
- User asked to consolidate to a single `master` branch (annoyed at typing
  `git checkout main`). Found `main` and `master` pointed to the identical
  commit already. Switched to `master`, deleted local `main`, set `master`
  to track `origin/main` directly — so `git pull`/`git push` "just work"
  without ever touching `main` locally again. Confirmed this doesn't touch
  the shared remote (teammates still see `main` there).
  - Along the way: **`git` was not installed anywhere accessible** on this
    HPC account. Installed it via `conda install -n starkai -c conda-forge
    git -y` (user's active general-purpose env at the time).

### 2.2 Reading `SESSION_CONTEXT.md` (a document from a *different*, earlier
Claude session, under a different cluster account, `aiy257590` — not this
session's account, `aiy237528`)
- That doc describes: RECAP = REward-Calibrated Alignment for Preference
  optimization, a DPO/GRPO/PPO pipeline for Hindi↔{Bhili,Gondi,Mundari} MT,
  19 Python files in `code/`, `config.py` as single source of truth, 3
  code-review rounds' worth of bug fixes already committed, and an
  **unresolved** `trl`/`transformers` compatibility saga (fixing that saga
  turned out to need re-doing from scratch in this session — see §2.4).
- Flagged then, confirmed since: that document is now stale in places (repo
  URL, branch name, and — turned out — the whole environment) because it was
  written by a different account/session than this one.

### 2.3 Early pipeline path bugs (before the environment saga)
- `python recap_split.py` from the repo root failed — scripts live in
  `code/`, not the repo root. Not a bug, just needed `cd code/` first.
- `[Skip] ... maha_data_2/bhili/hi2tgt.csv not found` — real bug:
  `config.py`'s `MAHA_DATA_2_ROOT` pointed at `RECAP/maha_data_2`, but the
  data actually lives one level deeper at `RECAP/dataset/maha_data_2/`.
  **Fixed**: `MAHA_DATA_2_ROOT = ROOT / "dataset" / "maha_data_2"`.
- User asked to hardcode an absolute path for portability "across any
  Pragya session." Explained `config.py`'s existing
  `ROOT = Path(__file__).resolve().parent.parent` already achieves that
  automatically (derives from wherever the code physically lives, not a
  string anyone has to edit) — hardcoding would actually be *less* portable
  for teammates. User agreed to keep it as-is.
- `JOB_GUIDE.md` had `cd /path/to/RECAP/code` placeholders — replaced with
  the real cluster path (`/flash/scai/msr/aiy237528/final-climb-adivaani/RECAP/code`)
  in all 3 occurrences.

### 2.4 The environment saga (this session's own version of it)
- `torchrun ... recap_train_dpo.py` → `ModuleNotFoundError: No module named
  'trl'`. Root cause: the active env (`starkai`) is a general-purpose env
  that never had RECAP's training libraries installed, and separately had
  `transformers==5.9.0`, far newer than the `trl==0.11.4` pin RECAP needs
  (old `trl`'s classic seq2seq `PPOTrainer` API + a transformers-4.46+
  method-name collision bug — see §2.5).
- **Decision**: rather than modifying the shared `starkai` env, created a
  new **dedicated `recap` conda env**, cloned from `starkai` (keeps its
  working GPU/CUDA-matched torch 2.6.0+cu124 setup intact).
- Installed pins **one package at a time** (not one joint `pip install`,
  which is what caused a resolver-backtracking disaster in the *original*
  `SESSION_CONTEXT.md` saga under the other account):
  `transformers==4.45.2` (must stay <4.46.0), `trl==0.11.4`, `tyro`
  (trl's PPOConfig needs it, wasn't obvious from RECAP's own imports),
  `sentencepiece`, `protobuf`, `unbabel-comet` (resolved cleanly this
  time — modern comet 2.2.7 + pytorch-lightning 2.6.5, not the ancient
  1.6.4 combo from the other session's saga), `setuptools<81` (newer
  setuptools dropped `pkg_resources`, which old `torchmetrics` — a comet
  transitive dep — still imports), and `numpy==1.26.4` (comet's own resolve
  requires this over the previously-installed 2.2.6).
- Verified: all 18 pipeline modules import cleanly under this env. Wrote
  **`requirements.txt`** at the repo root documenting the exact install
  order and *why* each pin exists (so this doesn't have to be re-derived on
  a fresh node/account again).

### 2.5 DPO smoke test — bug #1: autograd in-place-op crash
- First `recap_train_dpo.py --smoke_test` run crashed:
  `RuntimeError: one of the variables needed for gradient computation has
  been modified by an inplace operation`.
- **Root cause** (real bug in `trl==0.11.4`, not RECAP's own code):
  `DPOTrainer.get_batch_logps()` only `.clone()`s the `labels` tensor before
  mutating it in place on the *decoder-only* branch — the encoder-decoder
  branch (mT5's case) skips that clone, so it mutates the exact same tensor
  object the model already used (and saved for backward) when computing its
  own internal loss.
- **Fix**: a scoped monkeypatch in `recap_train_dpo.py::process_one()`,
  right after `from trl import DPOConfig, DPOTrainer` — wraps
  `DPOTrainer.get_batch_logps` to clone `labels` for the encoder-decoder
  case too, before calling the original function. (Caught and fixed two
  self-inflicted bugs in the first attempt at this patch: wrongly hardcoding
  `is_encoder_decoder=False` in the delegating call, and misusing `.__func__`
  on an already-plain `staticmethod`.) Verified with a standalone
  reproduction outside the real training loop — confirmed the bug reproduces
  unpatched and disappears patched.

### 2.6 DPO smoke test — bug #2: 30-minute DDP hang after training finished
- Retried the same command. Training completed successfully this time (all
  136 steps, best validation composite 0.4214 at step 100, checkpoint
  correctly saved) — but then the process hung for exactly 30 minutes and
  crashed with a NCCL `Watchdog caught collective operation timeout`.
- **Root cause**: `recap_train_dpo.py`, right after `trainer.train()`
  returns — a `wait_for_everyone()` barrier call was nested inside
  `if val_callback.best_step is None:`, but `best_step` is per-process state
  that's only ever populated on rank 0 (other ranks' copy stays `None`
  forever). So whenever validation succeeded during training (the normal
  case), rank 0 skipped the whole block — including the barrier — while
  other ranks entered it and waited at the barrier alone, forever.
- Confirmed via the actual `run_manifest.json` on disk that the *result*
  wasn't corrupted (the file-write itself is correctly `is_main_process()`-gated) — only 30 minutes of GPU time and a crash were lost, not the checkpoint.
- **Fix**: made the barrier unconditional (`recap_utils.wait_for_everyone()`
  called by every rank always; only the *decision* to run the fallback
  validation stays gated to rank 0).

### 2.7 Proactive audit — same hang-bug class found in two more places
*(User later explicitly asked: "check for all code files whether this hang
bug exists or not... correct it" — this section is that audit + fixes,
originally done for GRPO/PPO before they were even smoke-tested.)*
- Audited every file touching `accelerate`/DDP (only 3 do: the three
  trainers). `recap_train_grpo.py`'s existing barrier was already safe (its
  condition is the shared loop counter `update`, identical across ranks by
  construction).
- **Found a worse instance** in both `recap_train_grpo.py` and
  `recap_train_ppo.py`: each rank samples a *different* random batch and
  generates with sampling on, so whether a rank's `keep_idx` (valid,
  non-zero-advantage completions) is empty is a per-rank-random outcome that
  can differ **on any of the ~2000 training updates**, not just once at the
  end. A `continue` skipping `accelerator.backward()`/`ppo_trainer.step()`
  (both collective calls) only on the ranks that happened to get no valid
  data would leave other ranks waiting forever.
- **Fix (same pattern in both files)**: before deciding to skip an update,
  all ranks now exchange one `torch.distributed.all_reduce(..., op=MIN)` —
  "did *you* get usable data this round?" — and either all proceed or all
  skip together, never split. Guarded with `torch.distributed.is_initialized()`
  so single-GPU/non-`torchrun` runs are unaffected. Verified with a real
  2-process `torch.distributed` simulation reproducing the exact divergent
  scenario (rank 0 has data / rank 1 doesn't) — confirmed both ranks now
  agree to skip together.

### 2.8 `RESOLVE_ERRORS.md` — running error log (user-requested)
- User asked for a plain-language running history of errors + fixes,
  separate from code comments. Created `RESOLVE_ERRORS.md` at the repo root;
  it now has **8 entries** (as of this writing) covering: wrong working
  directory, `MAHA_DATA_2_ROOT` path bug, the whole environment saga, the
  `trl` autograd bug, the DPO DDP hang, the GRPO/PPO hang audit, the PPO
  DDP-unwrap bug (§2.9), and the missing-checkpoint crash in
  `recap_evaluate.py` (§2.10). **Keep appending new entries here as new
  errors come up** — this was an explicit standing instruction, not a
  one-time request.
- Also created `logs/2026-08-23_dpo_raw_bhili_hi2tgt_smoke_test.log` — the
  raw terminal output from the DPO smoke test run, per the same request
  ("dump all into logs file").

### 2.9 PPO smoke test — bug: DDP-wrapped model has no `.generate()`
- First `recap_train_ppo.py --smoke_test` run: trained 5 steps fine, then
  crashed at its first mid-training validation check:
  `AttributeError: 'DistributedDataParallel' object has no attribute
  'generate'`.
- **Root cause**: `ppo_trainer.model` is DDP-wrapped under 2-GPU training;
  `DistributedDataParallel` doesn't expose `.generate()`/`.save_pretrained()`
  (only its underlying `.module` does). `recap_train_grpo.py` already
  unwrapped correctly for its equivalent step; PPO's was just missing it.
- **Fix**: `unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)`
  in `recap_train_ppo.py`'s validation block, used for both the `.generate()`
  call and the `.save_pretrained()` call.

### 2.10 `recap_evaluate.py` — missing-checkpoint crash with a confusing error
- Running `python recap_evaluate.py --lang Bhili --direction hi2tgt`
  (evaluates all 14 experiments) got through 11 fine, then crashed on
  `ablation_full_reward` (not yet trained) with a Hugging Face Hub
  `HFValidationError` about "repo id" formatting — a real but genuinely
  misleading error, since it's actually "this checkpoint folder doesn't
  exist," not anything about Hugging Face Hub repo names.
- **Fix**: `recap_infer.py::resolve_checkpoint()` now checks the checkpoint
  path exists and raises a clear, specific `FileNotFoundError` naming the
  missing experiment if not; `recap_evaluate.py::evaluate_one()` catches
  that and prints a `[Skip]` line (matching the existing convention used
  elsewhere in the pipeline) instead of crashing the whole run. Confirmed
  the only two actually-missing experiments for Bhili/hi2tgt are
  `ablation_full_reward` and `ablation_full_recap` (all other 11/14 are
  trained).

### 2.11 Interpreting the smoke-test results — "how am I doing on my 2000 points"
- Compiled all 11 available `report.json`s into a comparison table. Key
  honest takeaways given to the user:
  - The smoke test used **1,998 rows (0.7%)** of the real 220k-row Bhili
    dataset, so the numbers aren't statistically meaningful either way.
  - All 12 conditions landed within a tight ±2% BLEU band — consistent with
    noise, not a real effect.
  - `recap_dpo` and `recap_dpo_grpo` produced **metric-identical** results —
    checked the actual checkpoint files (different `md5`, different
    validation composite: 0.4204 vs 0.4214) — GRPO's smoke-test config (10
    updates, `lr=1e-6`) genuinely trained but moved the weights far too
    little to flip a single greedy-decoded token across 200 test sentences.
  - **Bottom line given to user**: this run proves the *pipeline* is now
    correct end-to-end (which was the real open risk before this session);
    it says nothing about whether RECAP's method actually works. Recommended
    one full-scale (non-smoke) run before committing to the full sweep.

### 2.12 "Is it NAACL worthy?" / "Is the idea genuinely strong?"
- Actually read `RECAP.pdf` (installed `pypdf` into the `recap` env to
  extract text — no `pdftotext`/`poppler` available on this system).
- **Assessment given**: the experimental *design* is genuinely strong
  (careful CRPO differentiation stating "this must be demonstrated
  experimentally, not assumed," a real cumulative ablation ladder, Table 10
  human-validates the pseudo-preferences against actual humans, an honest
  Limitations section). But **every results table (6 through 10) is
  explicitly marked "All values are illustrative placeholders"** — the doc
  is a research protocol/spec (§11 is literally titled "Instructions to
  Coding Agent"), not a paper with real evidence yet. Combined with only
  1/6 directions being smoke-tested on <1% of data, concluded: not
  submission-ready *yet*, but the gap is compute/time, not design.
- On raw novelty (separate, harder question the user asked directly):
  honest answer was **no**, not a novel mechanism — the paper's own Table 1
  concedes the contribution is "the full combination," not any one piece,
  and CRPO already does confidence-filtered offline MT preference
  construction. The real strength is the *setting* (Bhili/Gondi/Mundari —
  genuinely unstudied) and the *rigor* of testing it, not a conceptual
  breakthrough — a legitimate but different path to acceptance than "new
  idea," worth being honest about in the paper's own framing.

### 2.13 Scope decision: drop Gondi
- User asked "what if I skip Gondi, show full analysis on Bhili and
  Mundari?" — flagged the real risk: Gondi is the *weakest*-performing
  language in the actual candidate-model data (checked Tables 3–5 in the
  PDF), so dropping it without a stated reason risks reading as
  cherry-picking to a sharp reviewer. Suggested keeping Gondi as a
  documented diagnostic exclusion (cheap, pre-empts the question) rather
  than silently omitting it.
- User declined the diagnostic middle-ground and decided definitively:
  **"I only do this paper on Bhili and Mundari"** — no mention of Gondi at
  all. Updated `config.py`: `LANGUAGES = ["Bhili", "Mundari"]  # Gondi
  excluded from the paper's scope`. Confirmed every script's `--lang`
  argument is built from `cfg.LANGUAGES`, so this alone enforces the scope
  everywhere in the Python code; only `JOB_GUIDE.md`'s ~26 lines of explicit
  Gondi commands/job-counts are stale (offered to fix, not yet done).
- Follow-up honest take when asked "is 2 languages still a good NAACL
  paper?": yes, if done fully (all main-matrix + ablations + human eval +
  significance testing) — reviewers reward depth over breadth. Noted Bhili
  is Indo-Aryan and Mundari is Munda/Austroasiatic — genuinely different
  families, so the typological-diversity argument survives Gondi's removal
  (loses "three families," keeps "meaningfully distinct families"). Flagged
  one open question only the user can answer: is a 3-language version of
  this dataset known/citable anywhere outside their own work? If not,
  omitting Gondi entirely needs no justification at all.

### 2.14 Correcting the seed plan
- User corrected an assumption from §2.13's discussion: the `SEEDS =
  [13, 42, 2026]` / "234 jobs" plan in `config.py`/`JOB_GUIDE.md` was for
  **evaluation**, not retraining 3×. Since `recap_evaluate.py` decodes with
  **greedy** (deterministic) decoding, 3 identical-seed eval runs would
  produce byte-identical output — so "3 eval seeds" can't mean re-decoding.
  Asked the user directly which of three interpretations was meant; answer:
  **the paired bootstrap CI already in `recap_evaluate.py` is what "3 seeds"
  was meant to cover** — no extra runs needed. Real implication: **78
  training jobs total** (not 234) — one seed (13) per experiment, with
  uncertainty coming entirely from bootstrap resampling at eval time.

### 2.15 Statistical significance testing — explained, then implemented
- User asked for 95% CI **and p-values** (reviewers expect both), and where
  in the pipeline this belongs. Explained in text first (as asked) before
  writing code:
  - `recap_evaluate.py::compute_deltas()` already had a paired-bootstrap CI,
    but two real gaps: no p-value, and the result was computed-then-discarded
    (never written to disk, so nothing downstream — including
    `recap_report_tables.py`'s Table 7/8 builders, checked directly — could
    ever read it).
  - Per the paper's own §11.10 ("show all direction-level results before
    any macro average"), significance has to be computed and reported
    **per direction**, never on a macro-average across directions (they
    don't share test sentences, so a pooled bootstrap isn't valid).
- **Implemented** (after the explanation, in a later message):
  - Found and fixed a **subtler pre-existing bug** while doing this: the old
    bootstrap resampled the *mean of per-sentence BLEU/chrF deltas*, while
    the actually-reported number was *corpus-level* BLEU/chrF (sacrebleu's
    real non-linear aggregation) — a different statistic than what the CI
    was attached to. Rewrote it as a proper Koehn (2004)-style paired
    bootstrap: resample sentence *indices*, recompute real
    `sacrebleu.corpus_bleu`/`corpus_chrf` (and mean COMET) on each of 1,000
    resamples for both systems using the same indices, derive both the CI
    and a two-sided p-value from that one set of resamples.
  - This required `evaluate_one()` to start storing per-sentence hypothesis
    and reference **text** (not just scores) in `report.json` — a schema
    change. **Old cached `report.json` files (all 11 for Bhili/hi2tgt) were
    deleted** since they predate this and would `KeyError` otherwise; they
    need to be regenerated by rerunning `recap_evaluate.py` (this is exactly
    the pending action in §1).
  - `compute_deltas()` now caches its own output to
    `recap_eval/<lang>/<direction>/<experiment>/seed_<N>/deltas_vs_sft.json`
    (new `cfg.deltas_path()` helper).
  - `recap_report_tables.py` gained `build_table_7_significance()` /
    `build_table_8_significance()` → `table7_significance.csv` /
    `table8_significance.csv`, one row per direction (never macro-averaged).
  - `JOB_GUIDE.md` updated (§8 evaluate, §9 reporting) with the new
    behavior, the migration `rm` command for stale caches, and suggested
    paper-table formatting (`delta [95% CI]` with `*`/`**` significance
    markers, or a plain `p=` value).
  - Verified: numerically (clear win → tight CI away from 0, p≈0; identical
    systems → CI=[0,0], p=1.0) and end-to-end through the full
    disk-write → cache-hit → table-read chain using synthetic data (no GPU
    needed for that). Did **not** verify against real Bhili checkpoints —
    this shell has no GPU (it's the cluster login node); that only happens
    when the user reruns it on their GPU node.

### 2.16 `table7_significance.csv`/`table8_significance.csv` showing 0 rows
- Diagnosed: `recap_report_tables.py` only *reads* `deltas_vs_sft.json`
  files; it never computes them. Confirmed by direct inspection that all 11
  cached `report.json` files were still the pre-migration schema (no
  `hyp`/`ref` fields) and zero `deltas_vs_sft.json` existed anywhere —
  meaning `recap_evaluate.py` hadn't successfully completed since the schema
  change (it likely crashed with `KeyError` on the first stale cache hit if
  it was run at all). **Deleted the 11 stale `report.json` files** (kept
  `best_checkpoint.json`, unaffected). Gave the exact rerun commands.

### 2.17 The "stuck" evaluate run → new significance test is just slow
- User reran `recap_evaluate.py --lang Bhili --direction hi2tgt`, got
  through 4 experiments (each showing real CI+p-value output, e.g.
  `dpo_no_confidence`: dCOMET=+0.0039 [-0.0005,+0.0092] p=0.086), then
  interrupted it thinking `recap_dpo`'s run was hung. **It wasn't** — the
  new bootstrap recomputes `sacrebleu.corpus_bleu`/`corpus_chrf` 2,000 times
  per experiment (1,000 resamples × 2 metrics), which is genuinely slow
  (~140s/experiment based on the observed timing, up from ~15–25s before).
  Explained this, then the user pivoted to a new task (§2.18) — **the eval
  rerun itself is still not finished** (see §1 pending action).

### 2.18 Dataset statistics + hyperparameter recommendations
- User asked for a full look at `dataset/maha_data_2` (all 3 languages),
  an exhaustive statistical analysis, a `dataset_statistics.csv`, and
  hyperparameter suggestions.
- Computed (via pandas over the real CSVs — 544,775 total rows across
  Bhili/Gondi/Mundari × 2 directions): corpus-level stats (row counts,
  duplicate-source rate, missing-value rate, source/reference length
  distributions, candidate-diversity metrics, oracle-best-of-4 gains) and
  per-model stats (BLEU/chrF/COMET mean/std/median, exact-match rate, empty-
  output rate, length ratio) — saved as **`dataset_statistics.csv`** (repo
  root, 30 rows).
- Also tokenized real samples with the actual mT5 tokenizer (from
  `mt5_finetune/Bhili/...`) to check **real** token-length percentiles
  against `config.py`'s `max_prompt_length=128`/`max_length=256` — found
  that stored candidate translations cap at *exactly* 128 tokens in every
  language (a generation artifact from `max_new_tokens=128`, not a true
  length), and Mundari references already exceed 128 tokens for ~1% of
  rows — a real, data-grounded truncation risk.
- Published a designed HTML report artifact (dataviz + artifact-design
  skills used; custom SVG grouped-bar charts with hover tooltips, light/dark
  theme support) at
  **https://claude.ai/code/artifact/650cda3d-d429-42a3-92ed-9ce4cc192224**
  — headline findings: mT5 wins 5/6 directions (matches the paper's own
  claim), +4–6 BLEU oracle gain from heterogeneous candidates (direct
  evidence for RECAP's core premise), ~2× BLEU swing between directions
  (justifies direction-specific reward calibration), and the 128-token
  truncation finding.
- **Hyperparameter recommendations given** (not applied to code — user's
  call):
  - Raise `max_new_tokens` (128 → 160–192) everywhere it's used for
    generation (`InferenceConfig`, `GRPOSettings`, `PPOSettings`).
  - Raise `DPOSettings.max_length` (256 → 320) — Mundari's combined
    prompt+completion tail sits close to the current ceiling.
  - **Keep** `max_prompt_length=128` (truncates <0.4% of sources, all
    languages) and `MAX_PAIRS_PER_SOURCE=6`/`MARGIN_DELTA=0.10` as-is.
  - Expect fewer retained DPO pairs for tgt2hi directions specifically
    (candidate models agree with each other more when translating *into*
    Hindi) — not a bug, just an expected asymmetry once Stage 4/5 run.

### 2.19 Full `config.py` review against the dataset stats
- User asked for a complete review of every `config.py` setting, grounded
  in the §2.18 statistics.
- **Critical finding, not previously flagged**: `SPLIT_RATIOS={80,10,10}`
  doesn't scale — at Bhili/Mundari's real size that's ~22K val AND test
  rows. Val gets decoded+scored on *every* validation event during
  training (dozens of times per job); the new paired-bootstrap significance
  test (§2.15) recomputes real corpus BLEU 1,000× per metric per
  experiment, so its cost scales with test-set size too. A val set this
  large could also genuinely take close to `DDP_TIMEOUT_SECONDS=1800`
  (30 min) to decode+score once, which would trip the DDP watchdog on a
  perfectly healthy run — indistinguishable from the real deadlock bugs
  fixed in §2.6/§2.7.
- Confirmed `max_new_tokens=128`/`DPOSettings.max_length=256` findings from
  §2.18 with the same reasoning.
- Also estimated (from the smoke test's own 8,714-pairs-from-1,600-sources
  ratio) that a full-scale Bhili DPO run could involve **~45,000 steps per
  experiment** — worth checking real `pairs_balanced.csv` size before
  committing to `num_train_epochs=3` across all 40 real DPO jobs; not
  changed, flagged as a "watch this" item.
- Confirmed everything else (`max_prompt_length`, `MAX_PAIRS_PER_SOURCE`,
  `MARGIN_DELTA`, `LENGTH_METRIC`, `REPETITION_NGRAM_SIZE`,
  `CALIBRATION_EPSILON`, reward weights) as fine, unchanged.
- User asked "can I justify this split to reviewers?" — answer: yes, but
  frame it as **absolute size matching FLORES-200 precedent (~1,000-1,100
  test sentences)**, not as a percentage (a bare "0.5%" invites the
  skepticism a cited, standard-sized test set doesn't). Refined the
  recommendation to **decouple val (small, only needs to support checkpoint
  selection) from test (sized to match precedent, since that's what's
  reported)**.
- User said "Yes" (apply it) — applied to `config.py`:
  - `SPLIT_RATIOS`: `{80, 10, 10}` → `{99.25, 0.25, 0.5}` (caught and fixed
    an arithmetic slip of my own along the way — first draft summed to
    1.003, not 1.0, which would have broken `recap_split.py`).
  - `DPOSettings.max_length`: `256` → `320`.
  - `max_new_tokens` in `InferenceConfig`/`GRPOSettings`/`PPOSettings`:
    `128` → `176` (all three kept synchronized).
  - Verified all three post-edit (`sum(SPLIT_RATIOS.values())==1.0`, real
    resulting split sizes computed: Bhili val≈551/test≈1,102, Mundari
    val≈525/test≈1,050).
  - **These changes only affect splits/checkpoints trained fresh from now
    on** — the existing Bhili/hi2tgt smoke-test artifacts predate this and
    are untouched (not invalidated, just under the old settings).

### 2.20 "What prompt is the DPO model using?" → reward-weight calibration deep-dive
- Answered the prompt-format question directly: traced every trainer/eval
  call site (`recap_train_dpo.py`, `recap_train_grpo.py`,
  `recap_train_ppo.py`, `recap_utils.generate_batch`, `recap_infer.py`) —
  **the prompt is the raw `source` sentence, verbatim, no instruction
  template or language-pair prefix anywhere in RECAP's own code.** Works
  because RECAP trains a fully separate checkpoint per (language,
  direction) — the model doesn't need a prefix to know which direction to
  translate, it *is* that direction's translator. Flagged one thing outside
  this codebase's visibility: whatever prompt convention the *original* SFT
  checkpoints were trained under (before RECAP) can't be verified from here.
- User then asked to "ponder... in the background of dataset stats" on the
  reward weights (`w_quality=0.60`/`w_rep=0.25`/`w_len=0.15`) — a follow-up
  to §2.19 where this was left alone without deep justification. Computed a
  real 4-gram repetition-rate statistic (matching `REPETITION_NGRAM_SIZE=4`)
  that wasn't in the original `dataset_statistics.csv`:
  - Repetition is rare overall (0.78–4.47% of candidates have any repeat)
    but **Mundari's rate is 5–10× Bhili's** — and this generalizes an
    example the paper's own PDF already cites (a naturally-repetitive
    Mundari reference).
  - mT5 (the actual trained backbone) has the *lowest* repetition rate of
    all 4 candidate models — its value is in disqualifying bad candidates
    from other models (esp. Qwen) during pair mining, not cleaning up its
    own behavior.
- User said "actual z-scored calibration stats — Do it." Ran RECAP's own
  real `RewardEngine.fit()` (not a reimplementation) on a genuine 20,000-row
  sample per direction — confirmed CPU-safe first (checked
  `recap_calibrate.py`: reuses precomputed BLEU/chrF/COMET from
  `maha_data_2`, only computes rep/len fresh from text, never touches the
  COMET model).
  - **Found and fixed a real, previously-unknown bug in `recap_reward.py`
    while doing this**: `_is_valid()` only checked `candidate is None`, not
    the NaN-float case pandas actually produces for a blank CSV cell —
    crashed `_repetition_penalty()` downstream. Confirmed 0.005%–0.175% of
    real candidates are affected (concentrated in Qwen/Llama) — this
    *would* have crashed any real Stage 2/3 run. Fixed: `_is_valid()` now
    also catches `isinstance(candidate, float) and math.isnan(candidate)`.
  - **The real finding**: even though `w_rep(0.25) > w_len(0.15)` nominally,
    for the *median* candidate in every direction, `len`'s typical z-score
    (~0.3–0.45) contributes more to the actual reward than `rep`'s
    (~0.05–0.13) — because repetition is rare enough that its z-score sits
    near zero for most rows. `rep` only "wakes up" in the tail, and almost
    exclusively for Mundari (p99 |z| = 4.5–4.8, vs Bhili's 0.05–1.08).
  - **Conclusion**: this is the z-scoring working correctly (Bhili genuinely
    doesn't have much of a repetition problem), not a bug to fix by
    reweighting — left `w_rep`/`w_len` unchanged. But it sharpens the
    Table 8 prediction from §2.19 into something quantified and falsifiable:
    the `+repetition` ablation step's improvement should be driven almost
    entirely by Mundari's tail cases.
  - Offered to fold the repetition-rate table into `dataset_statistics.csv`/
    the report artifact — **not yet done**, still open.

### 2.21 "Check if I am saving multiple checkpoints" → found and fixed a 1.26TB bug
- User asked directly whether multiple checkpoints were being saved, and to
  switch to "best checkpoint (by val) + last checkpoint" if so.
- Checked disk directly rather than just reading code: **8 completed
  Bhili/hi2tgt DPO smoke-test experiments (136 steps each) had accumulated
  1.26 TB of checkpoints between them** (18–28 `checkpoint-<step>`
  directories per experiment, 118–183GB each).
- **Root cause**: `DPOConfig(...)` in `recap_train_dpo.py` never set
  `save_total_limit` — Hugging Face's `Trainer` default is to keep every
  periodic checkpoint forever. **GRPO/PPO don't have this problem** —
  checked, zero `latest_state` dirs exist anywhere; they already
  overwrite a single fixed-path resume snapshot in place and delete it
  entirely on successful completion.
- **Fix**: added `save_total_limit=1` to `DPOConfig` — DPO now keeps
  exactly what was asked: the best-by-validation checkpoint (already a
  separate, single-path, overwrite-in-place location — unaffected by this
  bug) plus the single most recent step checkpoint (enough for resume).
- **Cleanup**: verified all 8 experiments had `run_manifest.json`
  (training genuinely complete, so their `trainer_state/` dirs serve no
  further purpose) and that each one's real deployed checkpoint was intact
  and separate, before deleting — with explicit user confirmation via
  AskUserQuestion given the size. Deleted all 8; `recap_dpo/` went from
  >1.26TB down to 18GB. Verified all 8 deployed checkpoints survived intact.
- Logged as `RESOLVE_ERRORS.md` entries #9 (the NaN bug from §2.20) and #10
  (this checkpoint bug) at the user's request.

### 2.22 "Where are loss plots?" → built train/val curve logging + plotting from scratch
- User asked where loss plots live. Checked directly rather than assuming:
  **nowhere** — `recap_report_plots.py` (Stage 11) only built Figures 1–4
  (margin distribution, model participation, delta-vs-SFT heatmap,
  main-matrix summary), none of which are training curves. Worse: the
  underlying per-step data wasn't even being saved anywhere. DPO's
  `DPOConfig(report_to=[])` explicitly disables every logging integration
  (no tensorboard/wandb), so the `{'loss':..., 'rewards/accuracies':...}`
  dicts only ever went to stdout. GRPO/PPO's per-update metrics were only
  ever `tqdm.write()`/progress-bar output, never persisted either. This is
  a real, previously-unflagged gap against the paper's own §11.11 mandatory
  plots (items 8–10: DPO/GRPO/PPO curves "versus update").
- User said "Yes. Train and Val" — built the full feature:
  - `recap_utils.append_jsonl()` (new) — shared, `is_main_process()`-gated,
    append-only JSONL writer (resume-safe: a resumed run's log just
    continues, no dedup logic needed).
  - `config.py` — `train_log_path()` / `val_log_path()` helpers, same shape
    as the existing `run_manifest_path()`.
  - **DPO**: added an `on_log` hook to the existing TRL callback adapter —
    persists whatever TRL's `DPOTrainer` already computes each
    `logging_steps` (loss, `rewards/accuracies`, `rewards/margins`,
    `logps/chosen`, `logps/rejected`) rather than deriving new metrics.
    `ValidationCheckpointCallback.evaluate()` (the single place ALL
    validation events already funnel through — periodic, resume-reseed,
    end-of-training fallback) now also appends to `val_log.jsonl`.
  - **GRPO**: logs loss/policy_loss/kl/mean_reward/advantage_std/completion-
    length/repetition/n_invalid per update; validation composite/BLEU/chrF/
    COMET per eval event.
  - **PPO**: logs policy_loss/value_loss/entropy/approx_kl/clip_fraction/
    explained_variance/response_len per update. Verified the exact stats
    dict key names (`ppo/loss/policy`, `ppo/policy/clipfrac`,
    `ppo/val/var_explained`, etc.) directly against this pinned trl
    version's actual `PPOTrainer.loss()`/`record_step_stats()` source —
    not guessed.
  - `recap_report_plots.py`: new `figure_5_dpo_curves` /
    `figure_6_grpo_curves` / `figure_7_ppo_curves` — one 2×2-panel PNG per
    `(lang, direction, experiment)` (3 train-metric panels + 1 always-
    present validation panel), wired into `main()`.
  - Caught and fixed a bug in my OWN first test attempt along the way
    (wrote synthetic data to a scratch path the figure functions never
    actually read from, since they use `cfg.DPO_ROOT`/etc. directly, not a
    parameter) before it could hide a real bug — redid the test correctly
    by overriding the root globals, then **rendered and visually inspected
    the actual PNG output** to confirm correct layout/legends/data, not
    just "no exception."

### 2.23 Git commit, then a tangled push
- Gave the user a copy-pasteable `git commit` command (staging `code/` as a
  whole plus the new docs/results) for them to run in their own terminal —
  didn't run it myself, since it was framed as "give me a command."
- User ran it and pushed; GitHub showed the commit landed
  (`312bbab`), but `git status` also showed a second, larger batch still
  staged-but-uncommitted, plus a puzzling `Your branch is based on
  'origin/main', but the upstream is gone` warning, plus an unexplained
  `test.txt`.
- Investigated directly (same shared cluster filesystem, so `git
  ls-remote`/`git branch -vv` from this session reflects the user's own
  terminal's reality exactly): confirmed the remote's `main` branch no
  longer exists at all — only `master` now, also the new GitHub default
  branch. Not something this session did. Fixed only the local tracking
  pointer (`git branch --set-upstream-to=origin/master master`) — did NOT
  touch the remote, since recreating `main` affects the whole team and
  wasn't this session's call to make unilaterally. Flagged `test.txt`
  (unexplained, not from this session) without touching it, and gave the
  user the one-line command for the still-pending second commit.

### 2.24 GitHub token check, then permanent SSH auth
- User selected the exact token text from `flash/pedro/command.txt`
  (`ghp_...`, a classic PAT sitting in plaintext) and asked whether it has
  access to `RECAP-Team/RECAP`. Checked directly against the GitHub API
  without ever printing the token value back: authenticates as
  `raja17021998`, full admin/push/pull on the repo, but the token's actual
  *scope* is far broader than the repo (`admin:enterprise`, `admin:org`,
  `delete_repo`, etc.) — flagged this as a real risk given it's sitting in
  a general notes file, recommended rotating to a narrower fine-grained PAT.
- User then hit the expected friction (`git push` prompting for
  username/password every time over HTTPS) and asked for a permanent fix.
  Offered two options; user picked **SSH** over permanently storing the
  broad token in plaintext.
- Setup hit a real environmental wall: this cluster blocks raw SSH (port
  22) *and* direct connections to GitHub's port-443 SSH fallback — only
  proxied HTTP(S) traffic works here (matches the proxy notes already in
  `pedro/command.txt`). Diagnosed this with actual connectivity tests
  (`/dev/tcp` probes) rather than assuming, found `ncat` (available, with
  `--proxy --proxy-type http` support) could tunnel SSH through the same
  HTTP proxy used for everything else, verified the tunneled connection
  actually authenticates before building anything permanent on top of it.
- Final setup: dedicated key `~/.ssh/id_ed25519_github` (kept separate from
  the cluster's own login key), added to the `raja17021998` GitHub account
  via the API (title `pragya-recap-cluster`), `~/.ssh/config` entry routing
  `github.com` → `ssh.github.com:443` through
  `proxy21.iitd.ernet.in:3128` via `ncat`, remote switched to
  `git@github.com:RECAP-Team/RECAP.git`. Verified end-to-end with a real
  `git ls-remote origin` — zero prompts, no token involved. This survives
  across sessions (not a per-session environment variable).

### 2.25 JOB_GUIDE.md reordered by priority + per-language smoke-test scripts
- User asked (in Hindi/English mixed) for two things: (1) reorder
  `JOB_GUIDE.md`'s command listings so main-matrix experiments come before
  ablations, grouped by language, with parallelization noted — e.g. "all
  Bhili main commands, all Mundari main commands, ..., then all Bhili
  ablations, all Mundari ablations..."; (2) one combined smoke-test script
  per language covering both directions.
- **Caught a real scope conflict before building anything**: the user's own
  example included Gondi throughout (main commands, ablations, and its own
  `smoke_test_gondi.bash`), but `config.py`'s `LANGUAGES` has excluded
  Gondi since §2.13 — any `--lang Gondi` command would fail immediately
  with an argparse error as the code currently stands. Asked explicitly
  rather than guessing; user confirmed: drop Gondi from this request
  entirely.
- Read the full existing `JOB_GUIDE.md` (733 lines) before touching it,
  rather than editing from memory — confirmed it still had Gondi throughout
  (job counts, all four command-listing groups) since the earlier
  "offered, not actioned" cleanup item from §2.13 had never actually been
  done.
- Full rewrite: fixed every job count for the 2-language scope (78→52 total
  training jobs; Group A 6→4, Group B 60→40, Groups C/D/E 6→4 each — all
  cross-checked arithmetically), and replaced the old flat
  "Group B, then C, then D, then E" command listing with a new §3
  structured exactly as requested: §3.1 Bhili main (14 jobs: 12
  launch-together + 2 that must wait on `recap_dpo`), §3.2 Mundari main
  (same shape), §3.3 Bhili ablations (12, all parallel), §3.4 Mundari
  ablations (12, all parallel) — each command tagged with its underlying
  Group letter so the dependency rules in §0/§1 still apply.
- This left a numbering gap (old §3 immediately followed by old §7, since
  §4/§5/§6 — the old Group C/D/E sections — were folded into the new §3).
  Judged that gap would read as an editing mistake to anyone opening the
  file cold, so renumbered every subsequent section (old §7-§15 → new
  §4-§12, including the `11b`/`11c`/`11d` subsections) — did this
  programmatically (one regex pass over `§N` cross-references plus header
  lines, using original-captured numbers so renamed-into-existing-number
  collisions couldn't happen, e.g. old §15→§12 racing old §12→§9) rather
  than by hand, then verified every single `§N` reference in the final
  file resolves to the right target.
- New `smoke_test_bhili.bash` / `smoke_test_mundari.bash` (`code/`,
  executable): each runs the full former-§14c flow — all 10 DPO
  experiments, both GRPO conditions, PPO, evaluate, reporting — for BOTH
  directions of one language in a single `bash` command, `set -euo
  pipefail` so it stops on first failure. Verified: `bash -n` syntax check
  on both, and cross-checked every `--experiment` name in both scripts
  against `config.py`'s real `EXPERIMENTS` dict keys (13/13 valid, no
  typos) — not just assumed correct from memory.

### 2.26 Whenever any code runs, everything gets logged to `logs/` automatically
- User first asked (bash-level): "Whenever I run some code, please add in
  logs" — the 3 bash entry-point scripts (`smoke_test_bhili.bash`,
  `smoke_test_mundari.bash`, `run_all_experiments.sh`) got `exec >
  >(tee -a "$LOG_FILE") 2>&1` wrapping. Hit a real race condition
  (background `tee` from process substitution can still be flushing when
  the script exits — demonstrated with an empty log file right after a
  "successful" run) — fixed with `TEE_PID=$!` + `trap 'exec 1>&- 2>&-;
  wait "$TEE_PID" 2>/dev/null || true' EXIT` (a `trap` is required, not
  just end-of-script code, because `set -e` failures jump straight to the
  `EXIT` trap and skip any code after them).
- User then asked "Where are the logs saved, just give dir?" — answered:
  `<repo-root>/logs/` (outside `code/`, matches `cfg.ROOT / "logs"`).
- User then broadened the request significantly: "Please add all loss
  plots and all metrics and also whichever code I run, please save each
  and everything in a log file in dir logs (outside code)." Bash-level
  `tee` only covers the 3 bash wrappers — it does nothing for a script run
  directly via `python` or `torchrun` (the normal way to run any single
  training/eval/report stage per `JOB_GUIDE.md`). Solved this at the
  Python level instead so logging is automatic regardless of invocation
  method.
- Added `start_run_logging(script_name, args)` to `code/recap_utils.py`,
  backed by a small `_Tee` class (writes to both the real
  stdout/stderr *and* a log file). Only runs on the main process
  (`is_main_process()` gate — correct even under `torchrun`, since
  `accelerate`'s `PartialState`/the `RANK` env var are both already set by
  torchrun before any script code runs, so this is safe to call at the very
  top of `main()`, before any DDP/`Accelerator` setup happens later inside
  `process_one()`). Builds a descriptive filename from a timestamp +
  script name + (`lang`/`direction`/`experiment` if present on `args`),
  e.g. `logs/2026-08-25_143012_recap_train_dpo_bhili_hi2tgt_recap_dpo.log`.
  Verified functionally in isolation before rolling out (fake
  `argparse.Namespace`, confirmed correct terminal passthrough + correct
  log file content + correct filename).
- Rolled out `recap_utils.start_run_logging(...)` to all **15** runnable
  pipeline scripts, called right after `args = parser.parse_args()` in each
  (or at the top of `main()` / the bare `__main__` block for the 3 scripts
  that take no CLI args): `recap_split.py`, `recap_calibrate.py`,
  `recap_score.py`, `recap_mine_pairs.py`, `recap_balance_pairs.py`,
  `recap_train_dpo.py`, `recap_train_grpo.py`, `recap_train_ppo.py`,
  `recap_infer.py`, `recap_evaluate.py`, `recap_report_tables.py`,
  `recap_report_plots.py`, `recap_sample_for_human_eval.py`,
  `recap_human_eval_analysis.py`, `recap_checks.py`. Since every training
  script (`recap_train_dpo/grpo/ppo.py`) already had per-step JSONL
  train/val curve logging from §2.22 (`train_log_path()`/`val_log_path()`,
  written to `recap_dpo/…`/`recap_grpo/…`/`recap_ppo/…`) and
  `recap_report_plots.py` already turns those into Figures 5/6/7 loss
  plots, this new layer's job is specifically the full raw terminal
  transcript — every print, every metric line, every stack trace — for
  literally any script, captured into `logs/` with zero extra flags needed
  at call time.
- `recap_report_tables.py`, `recap_report_plots.py`, and `recap_checks.py`
  needed `import recap_utils` added (they didn't previously import it) —
  checked for circular-import risk first (`recap_utils.py` only imports
  stdlib + `tqdm`, not `recap_checks`), confirmed clean.
- Verified all 7 newly-touched files this round
  (`recap_train_grpo.py`, `recap_train_dpo.py`, `recap_train_ppo.py`,
  `recap_split.py`, `recap_report_plots.py`, `recap_report_tables.py`,
  `recap_checks.py`) with `ast.parse` (syntax) and then a real
  `importlib.import_module()` pass inside the actual `recap` conda env
  (the bare system `python3` lacks `pandas`/`tqdm` — a real environment
  distinction worth remembering for any future verification step in this
  repo) — all 9 core modules imported cleanly with no errors.

---

## 3. Key files touched this session (for quick orientation)

| File | What changed |
|---|---|
| `code/config.py` | `MAHA_DATA_2_ROOT` fixed (missing `dataset/` prefix); `LANGUAGES` narrowed to `["Bhili", "Mundari"]`; added `deltas_path()`; `SPLIT_RATIOS`/`DPOSettings.max_length`/`max_new_tokens`×3 changed per §2.19; added `train_log_path()`/`val_log_path()` (§2.22) |
| `code/recap_utils.py` | new `append_jsonl()` helper for train/val curve logs (§2.22); new `_Tee` class + `start_run_logging(script_name, args)` — auto-logs every script's full terminal output to `logs/` (§2.26) |
| `code/recap_train_dpo.py` | trl autograd monkeypatch (§2.5); fixed the DDP hang (§2.6); `save_total_limit=1` to stop unbounded checkpoint accumulation (§2.21); `on_log` hook + val-log persistence for curves (§2.22); `start_run_logging()` call (§2.26) |
| `code/recap_train_grpo.py` | fixed the same-class DDP hang via `all_reduce` sync (§2.7); train/val curve logging (§2.22); `start_run_logging()` call (§2.26) |
| `code/recap_train_ppo.py` | same `all_reduce` fix (§2.7); DDP-unwrap fix for `.generate()`/`.save_pretrained()` (§2.9); train/val curve logging with verified real trl stats keys (§2.22); `start_run_logging()` call (§2.26) |
| `code/recap_infer.py` | `resolve_checkpoint()` now raises a clear error on a missing checkpoint instead of a confusing HF Hub error (§2.10); `start_run_logging()` call (§2.26) |
| `code/recap_evaluate.py` | proper Koehn-style paired bootstrap with p-values, deltas persisted to `deltas_vs_sft.json`, missing-checkpoint `[Skip]` handling (§2.10, §2.15); `start_run_logging()` call (§2.26) |
| `code/recap_report_tables.py` | new `table7_significance.csv`/`table8_significance.csv` builders (§2.15); `import recap_utils` + `start_run_logging()` call at top of `main()` (§2.26) |
| `code/recap_report_plots.py` | new Figures 5/6/7 — DPO/GRPO/PPO train+val curves, one per (lang, direction, experiment) (§2.22); `import recap_utils` + `start_run_logging()` call at top of `main()` (§2.26) |
| `code/recap_reward.py` | `_is_valid()` now catches NaN candidates, not just `None` (§2.20) |
| `code/recap_checks.py` | `import recap_utils` + `start_run_logging()` call in the bare `__main__` block (§2.26) |
| `code/recap_split.py` | `import recap_utils` + `start_run_logging()` call (§2.26) |
| `code/recap_score.py`, `recap_mine_pairs.py`, `recap_balance_pairs.py`, `recap_calibrate.py`, `recap_sample_for_human_eval.py`, `recap_human_eval_analysis.py` | each got `start_run_logging()` call after argparse (§2.26) |
| `code/JOB_GUIDE.md` | real cluster paths (§2.3); significance-testing instructions + migration note (§2.15); full rewrite — de-Gondi'd, job counts fixed, reordered by priority/language, renumbered §0-§12 (§2.25) |
| `code/smoke_test_bhili.bash` | new — full-pipeline smoke test, both directions, one command (§2.25); auto-logging via `exec`/`tee`/`trap` (§2.26) |
| `code/smoke_test_mundari.bash` | new — same, for Mundari (§2.25); auto-logging via `exec`/`tee`/`trap` (§2.26) |
| `code/run_all_experiments.sh` | auto-logging via `exec`/`tee`/`trap` added, command ordering/content otherwise untouched (§2.26) |
| `requirements.txt` | new — pinned package versions + why (§2.4) |
| `RESOLVE_ERRORS.md` | new — plain-language running error log, 10 entries so far (§2.8, §2.20, §2.21, append-only, keep going) |
| `dataset_statistics.csv` | new — full dataset statistical analysis (§2.18); repetition-rate table from §2.20 not yet folded in |
| `logs/2026-08-23_dpo_raw_bhili_hi2tgt_smoke_test.log` | new — raw smoke-test terminal output (§2.8) |
| `.gitignore` | added ignores for all regenerable pipeline-output dirs (`recap_splits/`, `recap_calib/`, `recap_rewards/`, `recap_pairs/`, `recap_dpo/`, `recap_grpo/`, `recap_ppo/`) — `recap_eval/`/`recap_report/`/`recap_human_eval/` deliberately left trackable, by user's choice |
| `CONVERSATION_SUMMARY.md` | this file |

**Disk state note**: `recap_dpo/bhili/hi2tgt/*/seed_13/trainer_state/` was
deleted for all 8 completed experiments as of §2.21 (was 1.26TB, now gone —
this was dead resume-state, not the deployed models, which are untouched).

**Outside the repo, but load-bearing for git to work at all**: `~/.ssh/config`
now routes all `github.com` SSH traffic through the campus proxy on port 443
(see §2.24) — this is machine/account state, not something `git clone`
elsewhere would reproduce. If cloning this repo fresh on a different
machine/account, HTTPS + a token (or a from-scratch SSH setup) is needed
there instead.

---

*(Append new sections below as the conversation continues — keep §1 "Quick
pickup" current, since that's what a fresh session will read first.)*
