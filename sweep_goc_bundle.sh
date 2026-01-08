#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="$(date +%Y%m%d_%H%M%S)"

DATASET="${DATASET:-graph_needle_test_branch.jsonl}"
RESULTS="${RESULTS:-experiment_goc_runs.jsonl}"
MODEL="${MODEL:-gpt-4o-mini}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-2}"
SIZES="${SIZES:-2 3 5 8 10}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

RUN_LOG="${RUN_LOG:-$LOG_DIR/goc_sweep_${DATE_TAG}.log}"

{
  echo "GoC bundle size sweep"
  echo "date_tag=${DATE_TAG}"
  echo "dataset=${DATASET}"
  echo "results=${RESULTS}"
  echo "model=${MODEL}"
  echo "temperature=${TEMPERATURE}"
  echo "top_k=${TOP_K}"
  echo "sizes=${SIZES}"
} | tee -a "$RUN_LOG"

for size in $SIZES; do
  LLM_LOG="${LOG_DIR}/llm_goc_b${size}_${DATE_TAG}.jsonl"
  {
    echo ""
    echo "=== bundle_size=${size} ==="
    echo "llm_log=${LLM_LOG}"
  } | tee -a "$RUN_LOG"

  python experiment_goc.py run \
    --dataset "$DATASET" \
    --results "$RESULTS" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --goc-only \
    --goc-bundle-size "$size" \
    --goc-top-k "$TOP_K" \
    --llm-log "$LLM_LOG" | tee -a "$RUN_LOG"
done

echo "done" | tee -a "$RUN_LOG"
