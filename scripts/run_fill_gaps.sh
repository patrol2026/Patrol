#!/bin/bash
# Fill missing (defender × model × attacker) cells in the main results table.
#   1. SafeDecoding × Vicuna × {6 attacks + alpaca_eval}      (7 runs)
#   2. {PPL, Self-Reminder, ICD, Retokenization, Self-Exam,
#      SafeDecoding} × LLaMA-2 × alpaca_eval                  (6 runs)
#
# Usage from SafeDecoding/exp/:
#   nohup bash run_fill_gaps.sh > fill_gaps.out 2>&1 &
#
# Estimated time: 3-4 hours total on a single GPU.

set -e

ALPACA_FLAG="--max_prompts 100 --random_sample"

run() {
    local defender=$1 model=$2 attacker=$3
    echo "============================================================"
    echo "  $defender  ×  $model  ×  $attacker"
    echo "============================================================"

    local extra=""
    [ "$attacker" = "alpaca_eval" ] && extra="$ALPACA_FLAG"

    python defense_patrol.py \
        --model_name "$model" \
        --attacker "$attacker" \
        --defender "$defender" \
        --max_new_tokens 400 \
        --seed 42 \
        --disable_GPT_judge \
        $extra
}

# ─────────────────────────────────────────────────────────────────────────────
# (1) SafeDecoding × Vicuna × all attacks
# ─────────────────────────────────────────────────────────────────────────────
for ATK in AdvBench GCG AutoDAN PAIR DeepInception RandomSearch alpaca_eval; do
    run SafeDecoding vicuna "$ATK"
done

# ─────────────────────────────────────────────────────────────────────────────
# (2) All baseline defenders × LLaMA-2 × alpaca_eval
# ─────────────────────────────────────────────────────────────────────────────
for DEF in PPL Self-Reminder ICD Retokenization Self-Exam SafeDecoding; do
    run "$DEF" llama2 alpaca_eval
done

echo ""
echo "All gap-filling generation runs done."

# ─────────────────────────────────────────────────────────────────────────────
# (3) GPT-4.1 ASR judging on the new safety-attack JSONs.
#     judge_all.py auto-skips alpaca_eval files (handled by AlpacaEval 2.0 below).
#     Files already judged with gpt-4.1 (have a *_gpt41_judged.json sibling)
#     are skipped automatically.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  GPT-4.1 ASR judging (judge_all.py)"
echo "============================================================"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "  [warn] OPENAI_API_KEY not set; skipping judging steps."
    exit 0
fi

cd "$(dirname "$0")/../.."   # repo root

python judge_all.py \
    --judge_backend openai \
    --openai_model gpt-4.1 \
    --include_dirs results \
    --model_filter llama2 vicuna \
    --no_strict_filter \
    --spreadsheet results_summary_fillgaps_gpt41.xlsx

# ─────────────────────────────────────────────────────────────────────────────
# (4) Official AlpacaEval 2.0 win-rate judging on the new alpaca_eval JSONs,
#     using the weighted_alpaca_eval_gpt41 annotator config (length-controlled,
#     position-swap, GPT-4.1 judge). --skip_existing avoids re-judging earlier
#     runs whose leaderboard CSV is already in place.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  AlpacaEval 2.0 judging (run_alpaca_eval_official.py)"
echo "============================================================"
python run_alpaca_eval_official.py \
    --output_dir results_alpaca_eval_official \
    --annotators_config weighted_alpaca_eval_gpt41 \
    --skip_existing \
    --input \
        "results/SafeDecoding_vicuna_alpaca_eval_*.json" \
        "results/PPL_llama2_alpaca_eval_*.json" \
        "results/Self-Reminder_llama2_alpaca_eval_*.json" \
        "results/ICD_llama2_alpaca_eval_*.json" \
        "results/Retokenization_llama2_alpaca_eval_*.json" \
        "results/Self-Exam_llama2_alpaca_eval_*.json" \
        "results/SafeDecoding_llama2_alpaca_eval_*.json"

echo ""
echo "All gap-filling runs and judging complete."
