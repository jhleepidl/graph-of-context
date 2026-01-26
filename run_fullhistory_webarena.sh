set -euo pipefail

WCA_ROOT="${WCA_ROOT:-/home/jhlee/WebChoreArena}"
TASKS_JSON="${TASKS_JSON:-$WCA_ROOT/BrowserGym/config_files/test_shopping.raw.json}"
SMALL_IDS="${SMALL_IDS:-$WCA_ROOT/BrowserGym/config_files/small_ids_10.txt}"
ENV_ID="${ENV_ID:-browsergym/webarena.310}"

MODEL="${MODEL:-gpt-4.1-mini}"          # 또는 gpt-5-mini
API_MODE="${API_MODE:-auto}"            # auto|chat|responses

MAX_STEPS="${MAX_STEPS:-60}"
OBS_TRUNCATE_CHARS="${OBS_TRUNCATE_CHARS:-8000}"

TRACE_DIR="${TRACE_DIR:-trace}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
TRACE_TAG="FullHistory_${MODEL}_${RUN_ID}"
OUT_DIR="results"
mkdir -p "$TRACE_DIR" "$OUT_DIR"

OUT_FILE="${OUT_DIR}/results_${TRACE_TAG}.jsonl"

# GPT-5*면 Responses 옵션 권장
EXTRA_LLM_ARGS=()
if [[ "$MODEL" == gpt-5* ]]; then
  EXTRA_LLM_ARGS+=( --api_mode "${API_MODE}" --reasoning_effort medium --verbosity low --max_output_tokens 900 )
else
  EXTRA_LLM_ARGS+=( --api_mode "${API_MODE}" --temperature 0.4 )
fi

python run_webchorearena_browsergym.py \
  --tasks_json "$TASKS_JSON" \
  --small_set_ids "$SMALL_IDS" \
  --method FullHistory \
  --task_config_mode config_file \
  --env_id "$ENV_ID" \
  --new_env_per_task \
  --max_steps "$MAX_STEPS" --headless \
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