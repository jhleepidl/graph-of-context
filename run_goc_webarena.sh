set -euo pipefail

WCA_ROOT="${WCA_ROOT:-/home/jhlee/WebChoreArena}"
TASKS_JSON="${TASKS_JSON:-$WCA_ROOT/BrowserGym/config_files/test_shopping.raw.json}"
SMALL_IDS="${SMALL_IDS:-$WCA_ROOT/BrowserGym/config_files/small_ids_10.txt}"
ENV_ID="${ENV_ID:-browsergym/webarena.310}"

MODEL="${MODEL:-gpt-4.1-mini}"          # 또는 gpt-5-mini
API_MODE="${API_MODE:-auto}"            # auto|chat|responses

MAX_STEPS="${MAX_STEPS:-60}"
OBS_TRUNCATE_CHARS="${OBS_TRUNCATE_CHARS:-8000}"

# GoC: fold 과다 방지 + unfold 강화
BUDGET_ACTIVE="${BUDGET_ACTIVE:-12000}"
BUDGET_UNFOLD="${BUDGET_UNFOLD:-2500}"
UNFOLD_K="${UNFOLD_K:-10}"

GOC_FOLD_POLICY="${GOC_FOLD_POLICY:-pef_url}"
GOC_PEF_HI="${GOC_PEF_HI:-1.8}"
GOC_PEF_LO="${GOC_PEF_LO:-0.8}"
GOC_PEF_BACKSTOP="${GOC_PEF_BACKSTOP:-3.2}"
GOC_PEF_KEEP_LAST="${GOC_PEF_KEEP_LAST:-16}"

TRACE_DIR="${TRACE_DIR:-trace}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
TRACE_TAG="GoC_${MODEL}_ba${BUDGET_ACTIVE}_bu${BUDGET_UNFOLD}_k${UNFOLD_K}_${RUN_ID}"
OUT_DIR="results"
mkdir -p "$TRACE_DIR" "$OUT_DIR"

OUT_FILE="${OUT_DIR}/results_${TRACE_TAG}.jsonl"

EXTRA_LLM_ARGS=()
if [[ "$MODEL" == gpt-5* ]]; then
  EXTRA_LLM_ARGS+=( --api_mode "${API_MODE}" --reasoning_effort medium --verbosity low --max_output_tokens 900 )
else
  EXTRA_LLM_ARGS+=( --api_mode "${API_MODE}" --temperature 0.4 )
fi

python run_webchorearena_browsergym.py \
  --tasks_json "$TASKS_JSON" \
  --small_set_ids "$SMALL_IDS" \
  --method GoC \
  --task_config_mode config_file \
  --env_id "$ENV_ID" \
  --new_env_per_task \
  --max_steps "$MAX_STEPS" --headless \
  --budget_active "$BUDGET_ACTIVE" --budget_unfold "$BUDGET_UNFOLD" \
  --unfold_k "$UNFOLD_K" \
  --goc_fold_policy "$GOC_FOLD_POLICY" \
  --goc_pef_hi_mult "$GOC_PEF_HI" \
  --goc_pef_lo_mult "$GOC_PEF_LO" \
  --goc_pef_backstop_mult "$GOC_PEF_BACKSTOP" \
  --goc_pef_roll_keep_last "$GOC_PEF_KEEP_LAST" \
  --obs_truncate_chars "$OBS_TRUNCATE_CHARS" \
  --loop_guard --loop_guard_force_action \
  --loop_guard_repeat_threshold 2 \
  --loop_guard_noop_threshold 1 \
  --loop_guard_ttl 15 \
  --trace_dir "$TRACE_DIR" --trace_tag "$TRACE_TAG" \
  --trace_include_obs --trace_obs_chars 12000 \
  --trace_include_prompt --trace_prompt_chars 2000 \
  --model "$MODEL" \
  "${EXTRA_LLM_ARGS[@]}" \
  --out "$OUT_FILE"

echo "Wrote: $OUT_FILE"