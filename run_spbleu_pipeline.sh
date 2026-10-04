#!/bin/bash
# Runs the full spBLEU/chrF++ benchmarking pipeline end-to-end: inference for
# all 5 models (mt5, NLLB, Qwen, Llama, Sarvam) across Bhili/Mundari/Gondi,
# val+test, all 4 directions (2 for Sarvam until its English checkpoints
# finish), then scores everything into spBLEU_scores/*.csv.
#
# Each model's infer.py/infer_sarvam.py is itself resumable (skips rows/jobs
# already done), so this is safe to re-run after an interruption -- it'll
# just pick up wherever it left off instead of redoing finished work.
#
# This is a LOT of real GPU inference (96+ jobs) and will likely run for
# hours -- run it inside tmux so a disconnect doesn't kill it:
#   tmux new -s spbleu
#   bash run_spbleu_pipeline.sh
#   (Ctrl+b, d to detach; tmux attach -t spbleu to come back)
#
# Run (from the RECAP repo root):
#   bash run_spbleu_pipeline.sh

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "mt5 + NLLB (env: mt5)"
echo "=============================================="
for lang in Bhili Mundari Gondi; do
    echo "--- mt5 / $lang ---"
    (cd mt5_finetune/$lang && conda run -n mt5 --no-capture-output python infer.py)
done
for lang in Bhili Mundari Gondi; do
    echo "--- NLLB / $lang ---"
    (cd NLLB-finetune/$lang && conda run -n mt5 --no-capture-output python infer.py)
done

echo "=============================================="
echo "Qwen + Llama (env: qwen_llama)"
echo "=============================================="
for lang in Bhili Mundari Gondi; do
    echo "--- Qwen / $lang ---"
    (cd qwen_finetune/$lang && conda run -n qwen_llama --no-capture-output python infer.py)
done
for lang in Bhili Mundari Gondi; do
    echo "--- Llama / $lang ---"
    (cd llama_finetune/$lang && conda run -n qwen_llama --no-capture-output python infer.py)
done

echo "=============================================="
echo "Sarvam (env: sarvam)"
echo "=============================================="
(cd sarvam_finetune && conda run -n sarvam --no-capture-output python infer_sarvam.py)

echo "=============================================="
echo "Scoring (env: mt5)"
echo "=============================================="
(cd spBLEU_scores && conda run -n mt5 --no-capture-output python compute_spbleu_scores.py)

echo "=============================================="
echo "ALL DONE. Scores in spBLEU_scores/*.csv"
echo "=============================================="
