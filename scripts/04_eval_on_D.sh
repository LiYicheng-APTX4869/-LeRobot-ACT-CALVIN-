#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

D_ROOT="${DATA_ROOT}/splitD_v30"
if [[ ! -d "${D_ROOT}" ]]; then
  D_ROOT="${DATA_ROOT}/splitD"
fi
if [[ ! -d "${D_ROOT}/data" || ! -f "${D_ROOT}/meta/info.json" ]]; then
  echo "[评估] 缺少 splitD 数据: ${D_ROOT}" >&2
  exit 1
fi

run_eval() {
  local exp_name="$1"
  local repo_id="$2"
  local exp_dir="${OUTPUT_ROOT}/${exp_name}"
  local eval_dir="${exp_dir}/eval_splitD"
  mkdir -p "${eval_dir}/logs" "${eval_dir}/videos" "${eval_dir}/rollouts"

  local policy_path="${POLICY_PATH:-${exp_dir}/checkpoints/last/pretrained_model}"
  if [[ ! -e "${policy_path}" ]]; then
    policy_path="${exp_dir}/checkpoints/last"
  fi

  echo "[评估:${exp_name}] policy=${policy_path}"
  echo "[评估:${exp_name}] dataset=${D_ROOT}"

  if [[ -n "${COURSE_EVAL_SCRIPT:-}" ]]; then
    echo "[评估:${exp_name}] 使用课程评估脚本: ${COURSE_EVAL_SCRIPT}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${COURSE_EVAL_SCRIPT}" \
      --policy-path "${policy_path}" \
      --dataset-root "${D_ROOT}" \
      --output-dir "${eval_dir}" \
      ${COMMON_EVAL_ARGS} 2>&1 | tee "${eval_dir}/logs/eval.log" || {
        if [[ "${STRICT_EVAL:-0}" == "1" ]]; then
          exit 1
        fi
        echo "[评估:${exp_name}] 课程评估失败，写入占位汇总。" >&2
        "${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_eval.py" --eval-dir "${eval_dir}" --write-placeholder
      }
  elif [[ "${EVAL_MODE:-offline_action_error}" == "offline_action_error" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${ROOT_DIR}/tools/eval_offline_action_error.py" \
      --policy-path "${policy_path}" \
      --dataset-root "${D_ROOT}" \
      --repo-id "${repo_id}_splitD_eval" \
      --output-dir "${eval_dir}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-batches "${EVAL_MAX_BATCHES:-200}" \
      --episodes "${EVAL_EPISODES:-0}" \
      --episode-start "${EVAL_EPISODE_START:-0}" \
      ${COMMON_EVAL_ARGS} 2>&1 | tee "${eval_dir}/logs/eval.log" || {
        if [[ "${STRICT_EVAL:-0}" == "1" ]]; then
          exit 1
        fi
        echo "[评估:${exp_name}] 离线动作误差评估失败，写入占位汇总。" >&2
        "${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_eval.py" --eval-dir "${eval_dir}" --write-placeholder
      }
  elif command -v "${LEROBOT_EVAL_BIN}" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${LEROBOT_EVAL_BIN}" \
      "--policy.path=${policy_path}" \
      "--dataset.repo_id=${repo_id}_splitD_eval" \
      "--dataset.root=${D_ROOT}" \
      "--output_dir=${eval_dir}" \
      "--batch_size=${BATCH_SIZE}" \
      "--num_workers=${NUM_WORKERS}" \
      ${COMMON_EVAL_ARGS} 2>&1 | tee "${eval_dir}/logs/eval.log" || {
        if [[ "${STRICT_EVAL:-0}" == "1" ]]; then
          exit 1
        fi
        echo "[评估:${exp_name}] lerobot-eval 失败，写入占位汇总。" >&2
        "${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_eval.py" --eval-dir "${eval_dir}" --write-placeholder
      }
  else
    echo "[评估] 未找到 lerobot-eval，也没有课程评估脚本。" >&2
    if [[ "${STRICT_EVAL:-0}" == "1" ]]; then
      exit 1
    fi
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_eval.py" --eval-dir "${eval_dir}" --write-placeholder
  fi

  "${PYTHON_BIN}" "${ROOT_DIR}/tools/summarize_eval.py" --eval-dir "${eval_dir}"
}

if [[ "$#" -gt 0 ]]; then
  EVAL_TARGETS=("$@")
else
  EVAL_TARGETS=(act_calvin_A_only act_calvin_ABC_joint)
fi

for target in "${EVAL_TARGETS[@]}"; do
  case "${target}" in
    act_calvin_A_only|A|a|a_only|A-only)
      run_eval "act_calvin_A_only" "${BASE_REPO_ID}"
      ;;
    act_calvin_ABC_joint|ABC|abc|ABC-joint)
      run_eval "act_calvin_ABC_joint" "${JOINT_REPO_ID}_v30"
      ;;
    *)
      echo "[评估] 未知目标: ${target}" >&2
      echo "[评估] 可用目标: act_calvin_A_only, act_calvin_ABC_joint" >&2
      exit 1
      ;;
  esac
done
