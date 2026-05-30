#!/usr/bin/env bash
#
# Shared helpers for agent_systems runners.
# Assumes execution from the repo root.
#

set -o pipefail

agsys_die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Load default AGSYS_* env vars from Python config (env vars override defaults).
if [[ -z "${AGSYS_CONFIG_LOADED:-}" ]]; then
  [[ -f "agent_systems/config.py" ]] || agsys_die "Missing agent_systems/config.py"
  eval "$(python3 agent_systems/config.py --print-bash)" || agsys_die "Failed to load agent_systems/config.py"
  export AGSYS_CONFIG_LOADED=1
fi

# Runtime judge fallbacks (only used if config.py did not already export them).
# Public default: GPT judge only. The optional GLM judge is OFF and ships with no
# endpoint baked in -- set AGSYS_EVAL_ENABLE_GLM_JUDGE=1 and AGSYS_EVAL_GLM_ENDPOINT
# yourself to evaluate with a local GLM/vLLM judge.
: "${AGSYS_EVAL_ENABLE_GLM_JUDGE:=0}"
: "${AGSYS_EVAL_GLM_PROVIDER:=vllm}"
: "${AGSYS_EVAL_GLM_MODEL:=GLM-4.7}"
: "${AGSYS_EVAL_GLM_ENDPOINT:=}"
: "${AGSYS_EVAL_GLM_THINKING:=disabled}"
: "${AGSYS_EVAL_GLM_MAX_WORKERS:=2}"
: "${AGSYS_EVAL_ENABLE_GPT_JUDGE:=1}"
: "${AGSYS_EVAL_GPT_PROVIDER:=openai}"
: "${AGSYS_EVAL_GPT_MODEL:=gpt-5-mini}"
: "${AGSYS_EVAL_GPT_MAX_WORKERS:=2}"
export \
  AGSYS_EVAL_ENABLE_GLM_JUDGE \
  AGSYS_EVAL_GLM_PROVIDER \
  AGSYS_EVAL_GLM_MODEL \
  AGSYS_EVAL_GLM_ENDPOINT \
  AGSYS_EVAL_GLM_THINKING \
  AGSYS_EVAL_GLM_MAX_WORKERS \
  AGSYS_EVAL_ENABLE_GPT_JUDGE \
  AGSYS_EVAL_GPT_PROVIDER \
  AGSYS_EVAL_GPT_MODEL \
  AGSYS_EVAL_GPT_MAX_WORKERS

agsys_require_eval_root() {
  [[ -d "${AGSYS_EVAL_ROOT}" ]] || agsys_die "Missing ${AGSYS_EVAL_ROOT}. Run: python3 agent_systems/prepare_sandbox.py"
  [[ -d "${AGSYS_EVAL_ROOT}/memory" ]] || agsys_die "Missing ${AGSYS_EVAL_ROOT}/memory. Run: python3 agent_systems/prepare_sandbox.py"
  [[ -d "${AGSYS_EVAL_ROOT}/qas" ]] || agsys_die "Missing ${AGSYS_EVAL_ROOT}/qas. Run: python3 agent_systems/prepare_sandbox.py"
  [[ -d "${AGSYS_EVAL_ROOT}/prompts" ]] || agsys_die "Missing ${AGSYS_EVAL_ROOT}/prompts. Run: python3 agent_systems/prepare_sandbox.py"
  [[ -f "${AGSYS_EVAL_ROOT}/question_ids.txt" ]] || agsys_die "Missing ${AGSYS_EVAL_ROOT}/question_ids.txt. Run: python3 agent_systems/prepare_sandbox.py"
  [[ -f "${AGSYS_SYSTEM_PROMPT}" ]] || agsys_die "Missing ${AGSYS_SYSTEM_PROMPT}. Check AGSYS_SYSTEM_PROMPT or agent_systems/config.py"
  [[ -f "${AGSYS_QA_SCHEMA}" ]] || agsys_die "Missing ${AGSYS_QA_SCHEMA}. Check AGSYS_QA_SCHEMA or agent_systems/config.py"
  local requested_mode
  requested_mode="$(agsys_normalize_memory_mode "${AGSYS_MEMORY_MODE:-sgm}")"
  if [[ -f "${AGSYS_EVAL_ROOT}/memory/memory_variant.json" ]]; then
    local prepared_mode
    prepared_mode="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("memory_mode", "sgm"))' "${AGSYS_EVAL_ROOT}/memory/memory_variant.json")" || agsys_die "Could not read ${AGSYS_EVAL_ROOT}/memory/memory_variant.json"
    [[ "${prepared_mode}" == "${requested_mode}" ]] || agsys_die "${AGSYS_EVAL_ROOT} memory mode is ${prepared_mode}, but AGSYS_MEMORY_MODE=${requested_mode}. Rebuild with: AGSYS_MEMORY_MODE=${requested_mode} AGSYS_EVAL_ROOT=${AGSYS_EVAL_ROOT} python3 agent_systems/prepare_sandbox.py --force"
  elif [[ "${requested_mode}" != "sgm" ]]; then
    agsys_die "Missing memory variant manifest for AGSYS_MEMORY_MODE=${requested_mode} at ${AGSYS_EVAL_ROOT}/memory/memory_variant.json. Rebuild with: AGSYS_MEMORY_MODE=${requested_mode} AGSYS_EVAL_ROOT=${AGSYS_EVAL_ROOT} python3 agent_systems/prepare_sandbox.py --force"
  fi
  if [[ "${requested_mode}" == "raw" ]]; then
    [[ -d "${AGSYS_EVAL_ROOT}/memory/raw_images" ]] || agsys_die "Missing raw image directory in ${AGSYS_EVAL_ROOT}. Rebuild raw sandbox."
    [[ -d "${AGSYS_EVAL_ROOT}/memory/raw_videos" ]] || agsys_die "Missing raw video directory in ${AGSYS_EVAL_ROOT}. Rebuild raw sandbox."
  fi
}

agsys_sanitize_tag() {
  local raw="${1:-default}"
  if [[ -z "${raw}" ]]; then
    raw="default"
  fi
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//:/_}"
  raw="${raw//,/}"
  echo "${raw}"
}

agsys_normalize_memory_mode() {
  local mode="${1:-sgm}"
  mode="${mode,,}"
  mode="${mode//-/_}"
  case "${mode}" in
    baseline|full)
      mode="sgm"
      ;;
    description|dm|descriptive_memory)
      mode="descriptive"
      ;;
    raw_entries|raw_media)
      mode="raw"
      ;;
  esac
  echo "${mode}"
}

agsys_memory_tag_suffix() {
  local mode
  mode="$(agsys_normalize_memory_mode "${1:-sgm}")"
  case "${mode}" in
    raw)
      echo "raw"
      ;;
    descriptive)
      echo "dm"
      ;;
    *)
      echo ""
      ;;
  esac
}

agsys_tag_with_memory_mode() {
  local base_tag="${1:-default}"
  local mode="${2:-sgm}"
  local suffix
  suffix="$(agsys_memory_tag_suffix "${mode}")"
  if [[ -n "${suffix}" ]]; then
    case "${base_tag}" in
      *-"${suffix}")
        ;;
      *)
        base_tag="${base_tag}-${suffix}"
        ;;
    esac
  fi
  agsys_sanitize_tag "${base_tag}"
}

agsys_list_qids() {
  local single_qid="${1:-}"
  if [[ -n "${single_qid}" ]]; then
    echo "${single_qid}"
    return 0
  fi
  cat "${AGSYS_EVAL_ROOT}/question_ids.txt"
}

agsys_run_dir() {
  local agent="$1"
  local model_tag="$2"
  local qid="$3"
  echo "${AGSYS_RUNS_ROOT}/${AGSYS_RUN_TAG}/${agent}/${model_tag}/${qid}"
}

agsys_setup_run_dir() {
  local agent="$1"
  local model_tag="$2"
  local qid="$3"

  local run_dir
  run_dir="$(agsys_run_dir "${agent}" "${model_tag}" "${qid}")"
  mkdir -p "${run_dir}/output"

  # Symlink shared dataset inputs into the per-question workspace (all within eval_root).
  local rel_memory rel_qjson rel_qtxt
  rel_memory="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${AGSYS_EVAL_ROOT}/memory" "${run_dir}")"
  rel_qjson="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${AGSYS_EVAL_ROOT}/qas/${qid}/question.json" "${run_dir}")"
  rel_qtxt="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${AGSYS_EVAL_ROOT}/qas/${qid}/question.txt" "${run_dir}")"

  ln -sfn "${rel_memory}" "${run_dir}/memory"
  ln -sfn "${rel_qjson}" "${run_dir}/question.json"
  ln -sfn "${rel_qtxt}" "${run_dir}/question.txt"

  echo "${run_dir}"
}

agsys_answer_valid() {
  local answer_file="$1"
  local expected_id="$2"
  python3 -c 'import json,sys; p=sys.argv[1]; exp=sys.argv[2]; d=json.load(open(p,"r",encoding="utf-8")); assert set(d.keys())=={"id","question","answer"}; assert str(d.get("id"))==str(exp); assert isinstance(d.get("answer"), str)' "${answer_file}" "${expected_id}" >/dev/null 2>&1
}

agsys_array_append_flag_if_value() {
  local -n target_array="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    target_array+=("${flag}" "${value}")
  fi
}

agsys_run_eval_command() {
  local judge_label="$1"
  shift

  echo "=== Evaluating ATM (${judge_label}) ==="
  PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}." python memqa/utils/evaluator/evaluate_qa.py "$@"
  local eval_exit=$?
  if [[ ${eval_exit} -ne 0 ]]; then
    echo "WARN: evaluate_qa.py failed for ${judge_label} (exit=${eval_exit})."
  fi
  return "${eval_exit}"
}

agsys_evaluate_predictions() {
  local predictions="$1"
  local eval_dir="$2"

  if [[ ! -f "${predictions}" ]]; then
    echo "WARN: predictions file not found for evaluation: ${predictions}"
    return 1
  fi

  local -a common_flags=(
    --ground-truth "${AGSYS_QA_SOURCE}"
    --predictions "${predictions}"
    --output-dir "${eval_dir}"
    --metrics em atm
  )
  local -a shared_extra_flags=()
  local -a glm_extra_flags=()
  local -a gpt_extra_flags=()
  local failures=0
  local ran_any=0

  read -r -a shared_extra_flags <<< "${AGSYS_EVAL_EXTRA_FLAGS:-}"
  read -r -a glm_extra_flags <<< "${AGSYS_EVAL_GLM_FLAGS:-}"
  read -r -a gpt_extra_flags <<< "${AGSYS_EVAL_GPT_FLAGS:-}"

  if [[ "${AGSYS_EVAL_ENABLE_GLM_JUDGE:-1}" != "0" ]]; then
    local -a glm_flags=(
      "${common_flags[@]}"
      --judge-provider "${AGSYS_EVAL_GLM_PROVIDER}"
      --judge-model "${AGSYS_EVAL_GLM_MODEL}"
      --max-workers "${AGSYS_EVAL_GLM_MAX_WORKERS}"
    )
    agsys_array_append_flag_if_value glm_flags --judge-endpoint "${AGSYS_EVAL_GLM_ENDPOINT:-}"
    agsys_array_append_flag_if_value glm_flags --judge-thinking "${AGSYS_EVAL_GLM_THINKING:-}"
    agsys_array_append_flag_if_value glm_flags --judge-reasoning-effort "${AGSYS_EVAL_GLM_REASONING_EFFORT:-}"
    agsys_array_append_flag_if_value glm_flags --request-delay "${AGSYS_EVAL_GLM_REQUEST_DELAY:-}"
    agsys_array_append_flag_if_value glm_flags --judge-max-retries "${AGSYS_EVAL_GLM_MAX_RETRIES:-}"

    agsys_run_eval_command "${AGSYS_EVAL_GLM_MODEL}" \
      "${glm_flags[@]}" \
      "${shared_extra_flags[@]}" \
      "${glm_extra_flags[@]}" || failures=$((failures + 1))
    ran_any=1
  fi

  if [[ "${AGSYS_EVAL_ENABLE_GPT_JUDGE:-1}" != "0" ]]; then
    local -a gpt_flags=(
      "${common_flags[@]}"
      --judge-provider "${AGSYS_EVAL_GPT_PROVIDER}"
      --judge-model "${AGSYS_EVAL_GPT_MODEL}"
      --max-workers "${AGSYS_EVAL_GPT_MAX_WORKERS}"
    )
    agsys_array_append_flag_if_value gpt_flags --judge-endpoint "${AGSYS_EVAL_GPT_ENDPOINT:-}"
    agsys_array_append_flag_if_value gpt_flags --judge-thinking "${AGSYS_EVAL_GPT_THINKING:-}"
    agsys_array_append_flag_if_value gpt_flags --judge-reasoning-effort "${AGSYS_EVAL_GPT_REASONING_EFFORT:-}"
    agsys_array_append_flag_if_value gpt_flags --request-delay "${AGSYS_EVAL_GPT_REQUEST_DELAY:-}"
    agsys_array_append_flag_if_value gpt_flags --judge-max-retries "${AGSYS_EVAL_GPT_MAX_RETRIES:-}"

    agsys_run_eval_command "${AGSYS_EVAL_GPT_MODEL}" \
      "${gpt_flags[@]}" \
      "${shared_extra_flags[@]}" \
      "${gpt_extra_flags[@]}" || failures=$((failures + 1))
    ran_any=1
  fi

  if [[ ${ran_any} -eq 0 ]]; then
    echo "SKIP: evaluation (both AGSYS_EVAL_ENABLE_GLM_JUDGE and AGSYS_EVAL_ENABLE_GPT_JUDGE are disabled)"
    return 0
  fi

  if [[ ${failures} -ne 0 ]]; then
    return 1
  fi
  return 0
}
