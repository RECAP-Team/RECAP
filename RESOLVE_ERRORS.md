# RECAP — Errors Encountered & How They Were Resolved

A running log of real errors hit while setting up and running the RECAP
pipeline, in plain language. New entries get appended to the bottom as they
come up — this is not a one-time document.

---

## 2026-08-23

### 1. `python: can't open file '.../recap_split.py'`

**What happened:** Ran `python recap_split.py ...` from the RECAP repo root
(`RECAP/`), but the pipeline scripts actually live one folder deeper, in
`RECAP/code/`.

**Fix:** `cd` into `RECAP/code/` first, then run the script. All of
`JOB_GUIDE.md`'s commands already assume you're standing inside `code/`.

---

### 2. `[Skip] Bhili/hi2tgt: .../maha_data_2/bhili/hi2tgt.csv not found`

**What happened:** `recap_split.py` couldn't find the input data. The
dataset actually lives at `RECAP/dataset/maha_data_2/<lang>/<direction>.csv`,
but `code/config.py` was pointing one level too shallow, at
`RECAP/maha_data_2/...` (missing the `dataset/` folder).

**Fix:** Changed one line in `config.py`:
```python
MAHA_DATA_2_ROOT = ROOT / "dataset" / "maha_data_2"   # was: ROOT / "maha_data_2"
```

---

### 3. `ModuleNotFoundError: No module named 'trl'`

**What happened:** The conda environment being used (`starkai`) is a
general-purpose environment and never had RECAP's training libraries
(`trl`, `comet`, etc.) installed. On top of that, `starkai` had
`transformers==5.9.0`, which is far newer than what RECAP's pinned
`trl==0.11.4` can work with (old `trl` needs `transformers` below `4.46`,
otherwise DPO training crashes for an unrelated reason — see error 4 below).

**Fix:** Rather than changing `starkai` (which could break other unrelated
work in that environment), created a **new dedicated conda environment
called `recap`**, cloned from `starkai` so it keeps the working
GPU/CUDA-matched PyTorch setup, then installed the missing packages into it
one at a time (installing them all together in one command was avoided
on purpose — doing that can make pip silently pick broken, years-old
versions of everything to satisfy all packages at once):

- `transformers==4.45.2` (downgraded from 5.9.0)
- `trl==0.11.4`
- `tyro` (needed by `trl`, wasn't obvious just from reading RECAP's own code)
- `sentencepiece`, `protobuf` (needed for mT5's tokenizer)
- `unbabel-comet` (for the COMET metric) — this also pulled `numpy` down to
  `1.26.4` and required pinning `setuptools<81`, because a very new
  `setuptools` removed something (`pkg_resources`) that `comet`'s older
  dependencies still need.

All of this is written down with reasons in `requirements.txt` at the repo
root, so it doesn't need to be re-figured-out next time.

**Going forward:** use `conda activate recap` (not `starkai`) for anything
RECAP-related.

---

### 4. `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

**What happened:** This one showed up only once actual GPU training started
(everything above just affects whether the code can *load* — this is a bug
in how training math is done). It's a genuine bug inside the `trl==0.11.4`
library itself (not code we wrote), and it only affects encoder-decoder
models like mT5 (not the more common decoder-only/GPT-style models `trl` is
usually tested against).

In plain terms: `trl` reuses the same batch of "labels" (the correct
translation tokens) twice — once by handing them to the model to compute its
own internal loss, and once inside its own DPO-specific math. The second use
edits those labels "in place" (modifies the same piece of memory) instead of
making a safe copy first. Editing something in place after PyTorch already
used it once to compute gradients corrupts the gradient computation, and
PyTorch refuses to proceed rather than silently give a wrong answer.

**Fix:** Instead of editing the installed `trl` library directly (which
would be lost the next time the environment is rebuilt), added a small
patch inside `recap_train_dpo.py` itself: right after importing
`DPOTrainer`, it swaps in a corrected version of the one function
(`get_batch_logps`) that makes a safe copy of the labels before editing
them, only for encoder-decoder models. Everything else about that function
is untouched. This was verified with a small standalone reproduction
(outside of RECAP entirely) confirming the exact same error happens without
the patch and disappears with it.

---

### 5. 30-minute hang, then crash: `Watchdog caught collective operation timeout` (SIGABRT)

**What happened:** This one was sneaky because *training itself finished
successfully* first — all 136 steps ran, validation scores were computed
correctly, the best checkpoint was saved, and `[Done]` printed. Then, instead
of the program just exiting, it froze for exactly 30 minutes and then
crashed hard.

Full log saved at `logs/2026-08-23_dpo_raw_bhili_hi2tgt_smoke_test.log`.

In plain terms: this pipeline runs on 2 GPUs at once (one "rank" per GPU),
and both GPUs are supposed to stay in lockstep — whenever one of them does a
synchronization step ("everyone wait here until both of us are ready"), the
other one has to do the exact same synchronization step at the exact same
point, or one of them waits forever for a partner that never shows up.

Our own code (`recap_train_dpo.py`) had exactly this mismatch, right after
training finished. It decided whether to do one extra "just in case"
validation check based on a value (`best_step`) that was only ever filled in
on GPU 0 — GPU 1's copy of that value stayed empty the whole time. So after
training, GPU 0 (which had a real value) skipped the synchronization step
entirely and moved on to finish up and exit. GPU 1 (whose value was still
empty) went the other way and stopped to wait at the synchronization point —
and waited, and waited, because GPU 0 was never coming. After 30 minutes
(the maximum time we tell it to wait before giving up), GPU 1 gave up and
the whole job crashed.

The important part: because the actual result-saving step in this code was
already correctly restricted to "only GPU 0 is allowed to save results," GPU
1 being wrong and stuck didn't corrupt anything — the real result
(`best_validation_composite=0.4214`) was already safely saved to
`run_manifest.json` before the hang started. Confirmed by reading that file
directly. So this bug wasted 30 minutes of GPU time and ended in an ugly
crash, but it did not produce a wrong or corrupted result.

**Fix:** Changed the code so that the synchronization step always happens
for both GPUs, no matter what — only the *decision of whether to run the
extra validation check* is allowed to differ between GPU 0 and GPU 1, not
whether they show up to the synchronization point together
(`recap_train_dpo.py`, inside `process_one()`, right after
`trainer.train()` returns). Verified the same style of bug does *not* exist
in the GRPO trainer (`recap_train_grpo.py`) — its equivalent synchronization
step is gated on the training-loop counter, which is identical on both GPUs
by construction, so it can't drift apart the way `best_step` did here. The
PPO trainer (`recap_train_ppo.py`) doesn't have any synchronization steps
like this yet at all — worth watching when that one gets its own smoke test.

---

### 6. Same hang bug (error 5), found by audit in two more places before they ever crashed

**What happened:** After fixing error 5, we went through every single code
file in `code/` on purpose, specifically looking for anything else that
could cause the same "two GPUs disagree about whether to synchronize"
problem. Found two more, both in files that hadn't been smoke-tested yet
(`recap_train_grpo.py` and `recap_train_ppo.py`) — so these were caught
*before* they ever caused a real crash, not from a log.

These two were actually a more serious version of the same idea. Error 5
only happened once, right at the very end of a run. These new ones live
inside the **main training loop itself** — the one that repeats thousands of
times per job. Here's why: during GRPO/PPO training, each GPU independently
makes up its own random batch of examples and generates its own random
translations of them (that's normal and intentional — it's how the two GPUs
avoid wastefully doing identical work). But every so often, by pure chance,
one GPU's random batch might turn out to have nothing usable in it (e.g. the
model happened to produce only garbage for that particular random batch),
while the other GPU's batch is fine. The code's old behavior was: "if my
batch has nothing usable, skip my turn and move to the next one" — but the
GPU whose batch *was* fine did NOT skip, and went ahead to do a training
update, which is a step that requires both GPUs to participate together.
One GPU shows up, the other doesn't — same freeze-then-crash as error 5,
except this could happen unpredictably at any point during any of the ~2000
training updates in a job, not just once at the end.

**Fix:** Before either GPU is allowed to decide "skip this one," they now
compare notes first (a quick, cheap message: "did you get anything usable
this round?"). If even one of them says no, **both** skip that round
together. If both got something usable, both proceed together. Either way,
they stay in agreement, so this can't freeze. Verified this actually works
with a standalone 2-GPU-process simulation reproducing the exact
disagreement scenario, confirming both sides now agree to skip together
instead of drifting apart.

Files fixed: `recap_train_grpo.py` and `recap_train_ppo.py` (same idea as
each other, since both scripts sample/generate their own rollouts the same
way). Audited every other file in `code/` for this same class of bug —
confirmed no other file talks to more than one GPU at all, so there was
nowhere else to check.

---

## 2026-08-23 (continued) — PPO smoke test

### 7. `AttributeError: 'DistributedDataParallel' object has no attribute 'generate'`

**What happened:** First run of the PPO smoke test (`recap_train_ppo.py`).
Training itself started fine and ran 5 update steps, then crashed the
moment it tried to run its first mid-training validation check.

In plain terms: when training uses 2 GPUs, PyTorch wraps the model in a
special container (`DistributedDataParallel`, aka "DDP") that keeps both
GPUs' copies of the model in sync automatically. That wrapper knows how to
handle normal training (forward/backward), but it does NOT know how to do
other model actions like "generate a translation" or "save yourself to
disk" — those calls have to go to the real model *inside* the wrapper, not
the wrapper itself. PPO's validation-check code was calling `.generate()`
and (further down, unreached because it crashed first) `.save_pretrained()`
directly on the wrapper instead of unwrapping it first. GRPO's equivalent
code (`recap_train_grpo.py`) already did this unwrapping correctly — PPO's
was just missing it.

**Fix:** Added the missing unwrap step (`ppo_trainer.accelerator.unwrap_model(...)`)
in `recap_train_ppo.py`, right before both the `.generate()` call and the
`.save_pretrained()` call in the per-step validation block — matching the
exact pattern `recap_train_grpo.py` already uses correctly.

---

## 2026-08-23 (continued) — full evaluation run

### 8. `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**What happened:** Ran `recap_evaluate.py --lang Bhili --direction hi2tgt`,
which evaluates all 14 experiments for that language/direction one after
another. The first 11 worked fine and printed real BLEU/chrF++/COMET
numbers. The 12th one, `ablation_full_reward`, crashed the whole run with a
confusing error about a Hugging Face "repo id."

In plain terms: `ablation_full_reward` (and, it turned out, one other
experiment, `ablation_full_recap`) simply **haven't been trained yet** — no
checkpoint folder exists for them. That by itself is completely normal and
not a bug (training happens experiment-by-experiment as separate jobs, so
it's expected that some haven't run yet). The actual problem was how the
code *reacted* to a missing checkpoint: instead of noticing "this folder
doesn't exist" and saying so clearly, it handed the (nonexistent) folder
path to a library function that tries two things — "is this a real folder?"
and "is this the name of a public model on the Hugging Face Hub?" — and
since our local folder path obviously isn't a valid Hub model name either,
it failed with a message about "repo id" formatting that has nothing to do
with the real issue and looks like a completely unrelated bug.

Worth noting: this crash also stopped the ENTIRE evaluation run partway
through, meaning even the lang/direction combinations after this one
(if evaluating all languages at once) would never have gotten a chance to
run.

**Fix:** Two small changes, both making the code fail the way it already
does elsewhere in this project (`[Skip] ... not found -- run Stage X
first`) instead of crashing with someone else's confusing error message:

- `recap_infer.py`, `resolve_checkpoint()`: now checks whether the
  checkpoint folder actually exists before returning its path, and raises a
  clear, specific error naming exactly which experiment is missing and
  which training script to run, if not.
- `recap_evaluate.py`, `evaluate_one()`: catches that error and prints a
  `[Skip]` line (matching the message style already used a few lines above
  it for a missing test split), then moves on to the next experiment
  instead of crashing the whole run.

Nothing needs to be redone — already-finished evaluations are cached to
disk and are skipped (instantly) on rerun; only the two genuinely-missing
ones now print a clear one-line skip message instead of a crash.

---

## 2026-08-23 (continued) — reward calibration & checkpoint disk usage

### 9. `AttributeError: 'float' object has no attribute 'split'` (reward code)

**What happened:** While actually running RECAP's own reward-calibration
code for real (not smoke-test data) to check whether the reward weights
(`w_quality`/`w_rep`/`w_len`) looked reasonable, it crashed partway through
scoring real candidate translations.

In plain terms: a few candidate translations in the raw dataset are
genuinely blank (never generated). When that blank cell gets read from the
CSV, Python doesn't read it as "nothing" — it reads it as a special
"not-a-number" placeholder that *looks* like a number, not like text. The
code that decides "is this candidate usable or should I skip it" only knew
how to recognize a MISSING candidate, not this NOT-A-NUMBER-shaped one, so
a genuinely blank candidate slipped through as "looks fine to me" — and
then crashed a few steps later when something tried to actually read it as
text (repetition-checking code trying to split text into words, on
something that isn't text).

This is rare — well under 0.2% of candidates, mostly from the weaker models
(Qwen, Llama) — but with datasets this size (200K+ rows), rare things
still happen, guaranteed, every time this code runs for real.

**Fix:** `recap_reward.py`'s "is this candidate usable" check
(`_is_valid()`) now also recognizes the not-a-number case, not just a
genuinely-missing one, and correctly marks those rows as unusable (skipped
from scoring) rather than letting them through and crashing later.

---

### 10. Training was silently saving hundreds of gigabytes of checkpoints per experiment

**What happened:** Not a crash — the user asked directly, "check if I'm
saving multiple checkpoints." Checked, and the answer was yes, badly:
**8 already-finished smoke-test DPO experiments had accumulated 1.26
terabytes of checkpoint files between them, just from smoke-scale runs of
136 training steps each.**

In plain terms: while training, the code periodically saves a snapshot of
the model so training can resume if the job gets killed partway through
(walltime limit, crash, etc.). The library RECAP uses for this
(`transformers`) defaults to keeping *every single snapshot it ever
saves*, forever, unless explicitly told otherwise — nobody had told it
otherwise. Every few hundred training steps, it saved a brand new multi-
gigabyte copy of the entire model and never deleted the old ones. At real
full-dataset scale (tens of thousands of steps instead of 136), this would
have used an amount of disk space the project likely could not have
afforded.

Worth noting: this only affected DPO. GRPO and PPO already saved their
resume-snapshot to the same fixed location every time (overwriting the old
one) and deleted it entirely once a run finished successfully — exactly
the right behavior. DPO was just missing the equivalent setting.

**Fix:** Added one setting (`save_total_limit=1`) to DPO's training
configuration, so it now only ever keeps the single most recent snapshot
(enough to resume from) instead of every snapshot it has ever saved. The
actual deployed model — the best-by-validation checkpoint, saved
separately — was never part of this problem and is unaffected.

**Cleanup done the same day:** deleted the 1.26TB of now-pointless old
snapshots from the 8 already-finished experiments (verified first that
each one had already fully completed and that its real deployed model,
stored separately, was untouched). `recap_dpo/` went from over a terabyte
down to 18GB.

---

*(Append new entries below this line as new errors come up.)*
