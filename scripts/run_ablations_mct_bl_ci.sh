#!/bin/bash
# Ablation runs for PATROL on RandomSearch.
#   Sweep A: vary max_check_tokens     {4,8,16,32,64,128,256}  with ci=4, bl=20
#   Sweep B: vary backtrack_length     {8,12,16,20,24,28}      with ci=4, mct=128
#   Sweep C: vary check_interval       {2,4,6,8}               with bl=20, mct=128
# Models: llama2, llama3, vicuna   →   total 21+18+12 = 51 runs.
# Usage (from SafeDecoding/exp/):
#   nohup bash run_ablations_mct_bl_ci.sh > ablations.out 2>&1 &

set -e

MODELS="llama2 llama3 vicuna"
COMMON="--attacker RandomSearch \
        --defender PATROL \
        --patrol_configs 1.0-1.0-50 \
        --max_new_tokens 400 \
        --seed 42 \
        --disable_GPT_judge"

run() {
    local model=$1 ci=$2 bl=$3 mct=$4
    echo "============================================================"
    echo "  $model  ci=$ci  bl=$bl  mct=$mct"
    echo "============================================================"
    python defense_patrol.py \
        --model_name $model \
        $COMMON \
        --patrol_check_interval $ci \
        --patrol_backtrack_length $bl \
        --patrol_max_check_tokens $mct
}

# ── Sweep A: max_check_tokens ablation (ci=4, bl=20) ─────────────────────────
for MODEL in $MODELS; do
    for MCT in 4 8 16 32 64 128 256; do
        run $MODEL 4 20 $MCT
    done
done

# ── Sweep B: backtrack_length ablation (ci=4, mct=128) ───────────────────────
for MODEL in $MODELS; do
    for BL in 8 12 16 20 24 28; do
        run $MODEL 4 $BL 128
    done
done

# ── Sweep C: check_interval ablation (bl=20, mct=128) ────────────────────────
for MODEL in $MODELS; do
    for CI in 2 4 6 8 16; do
        run $MODEL $CI 20 128
    done
done

echo ""
echo "All ablations done."

# ── GPT-4.1 judging of every newly produced JSON in results/ ─────────────────
# Skips files that already have a *_gpt41_judged.json sibling.
echo ""
echo "============================================================"
echo "  Judging ablation results with GPT-4.1"
echo "============================================================"

cd "$(dirname "$0")/../.."   # repo root

python judge_all.py \
    --judge_backend openai \
    --openai_model gpt-4.1 \
    --include_dirs results \
    --model_filter llama2 llama3 vicuna \
    --no_strict_filter \
    --spreadsheet results_summary_ablations_gpt41.xlsx

echo ""
echo "Ablation judging complete."
