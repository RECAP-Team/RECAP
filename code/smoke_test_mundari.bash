#!/usr/bin/env bash
# Full end-to-end smoke test for Mundari, BOTH directions (hi2tgt + tgt2hi),
# every stage, every experiment/ablation, torchrun multi-GPU path included.
# This is JOB_GUIDE.md section 14c, run for both directions in one shot
# instead of typed out by hand -- see that section for what each stage does
# and why (--n_samples shrinks the data, --smoke_test swaps in the small
# named *_SETTINGS_SMOKE_TEST objects, config.py's real settings are never
# touched).
#
# Run from a GPU-allocated session (JOB_GUIDE.md section 7's qsub -I
# pattern), from RECAP/code/:
#
#   export NGPUS_PER_JOB=2
#   bash smoke_test_mundari.bash
#
# Stops immediately on the first failing command -- a CheckFailure or crash
# partway through won't silently continue into later stages on broken state.

set -euo pipefail

LANG_NAME="Mundari"
N_SAMPLES=200
DIRECTIONS=(hi2tgt tgt2hi)
MAIN_DPO_EXPERIMENTS=(dpo_raw dpo_quality_only dpo_no_confidence recap_dpo)
ABLATION_EXPERIMENTS=(ablation_quality_only ablation_quality_plus_rep ablation_quality_plus_len ablation_full_reward ablation_full_reward_margin ablation_full_recap)

: "${NGPUS_PER_JOB:?Set NGPUS_PER_JOB first, e.g. export NGPUS_PER_JOB=2}"

log() { echo; echo "=== [$LANG_NAME] $* ==="; }

for DIR in "${DIRECTIONS[@]}"; do
  log "$DIR -- Stage 1-5: data prep (--n_samples $N_SAMPLES)"
  python recap_split.py --lang "$LANG_NAME" --direction "$DIR" --n_samples "$N_SAMPLES"
  python recap_calibrate.py --lang "$LANG_NAME" --direction "$DIR"
  python recap_score.py --lang "$LANG_NAME" --direction "$DIR"
  python recap_mine_pairs.py --lang "$LANG_NAME" --direction "$DIR"
  python recap_balance_pairs.py --lang "$LANG_NAME" --direction "$DIR"

  log "$DIR -- Stage 6: DPO, main-matrix experiments (${#MAIN_DPO_EXPERIMENTS[@]})"
  for EXP in "${MAIN_DPO_EXPERIMENTS[@]}"; do
    torchrun --standalone --nproc_per_node="$NGPUS_PER_JOB" recap_train_dpo.py --lang "$LANG_NAME" --direction "$DIR" --experiment "$EXP" --smoke_test
  done

  log "$DIR -- Stage 6: DPO, ablation experiments (${#ABLATION_EXPERIMENTS[@]})"
  for EXP in "${ABLATION_EXPERIMENTS[@]}"; do
    torchrun --standalone --nproc_per_node="$NGPUS_PER_JOB" recap_train_dpo.py --lang "$LANG_NAME" --direction "$DIR" --experiment "$EXP" --smoke_test
  done

  log "$DIR -- Stage 7: GRPO (sft_grpo, recap_dpo_grpo)"
  torchrun --standalone --nproc_per_node="$NGPUS_PER_JOB" recap_train_grpo.py --lang "$LANG_NAME" --direction "$DIR" --experiment sft_grpo --smoke_test
  torchrun --standalone --nproc_per_node="$NGPUS_PER_JOB" recap_train_grpo.py --lang "$LANG_NAME" --direction "$DIR" --experiment recap_dpo_grpo --smoke_test

  log "$DIR -- Stage 8: PPO (sft_ppo)"
  torchrun --standalone --nproc_per_node="$NGPUS_PER_JOB" recap_train_ppo.py --lang "$LANG_NAME" --direction "$DIR" --experiment sft_ppo --smoke_test

  log "$DIR -- Stage 11: Table 10 sampling (structural check only)"
  python recap_sample_for_human_eval.py --experiment recap_dpo --n 20
done

log "Stage 9: evaluate (all 14 experiments, both directions)"
for DIR in "${DIRECTIONS[@]}"; do
  python recap_evaluate.py --lang "$LANG_NAME" --direction "$DIR"
done

log "Stage 11: reporting"
python recap_report_tables.py
python recap_report_plots.py

log "smoke test complete -- both directions, every stage, no errors"
echo "Next: JOB_GUIDE.md section 15 to clean up this language's test output and switch to the real full-data run."
