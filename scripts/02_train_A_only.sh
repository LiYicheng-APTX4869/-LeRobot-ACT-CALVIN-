#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

EXP_NAME="act_calvin_A_only"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SCRIPT_LOG_DIR="${OUTPUT_ROOT}/_script_logs/${EXP_NAME}"
if [[ -d "${DATA_ROOT}/splitA_v30" ]]; then
  DATASET_ROOT="${DATA_ROOT}/splitA_v30"
  DATASET_REPO_ID="local/calvin_splitA_v30"
elif [[ -d "${DATA_ROOT}/_lerobot_convert_home/local/calvin_splitA_v30" ]]; then
  DATASET_ROOT="${DATA_ROOT}/_lerobot_convert_home/local/calvin_splitA_v30"
  DATASET_REPO_ID="local/calvin_splitA_v30"
else
  DATASET_ROOT="${DATA_ROOT}/splitA"
  DATASET_REPO_ID="${BASE_REPO_ID}"
fi
mkdir -p "${OUTPUT_ROOT}" "${SCRIPT_LOG_DIR}"
mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_datasets" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"

if [[ ! -d "${DATASET_ROOT}/data" || ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "缺少数据集 splitA。请先运行 scripts/01_download_data.sh。" >&2
  exit 1
fi

if [[ "${DATASET_ROOT}" == "${DATA_ROOT}/splitA" ]] && grep -q '"codebase_version"[[:space:]]*:[[:space:]]*"v2' "${DATASET_ROOT}/meta/info.json"; then
  echo "[训练:A] 当前 splitA 仍是 LeRobot v2 格式，当前 LeRobot 需要 v3 格式。" >&2
  echo "[训练:A] 请先运行: FORCE_RECONVERT=1 bash scripts/01_convert_data_v30.sh splitA" >&2
  echo "[训练:A] 成功后应出现目录: ${DATA_ROOT}/splitA_v30" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/patch_v30_stats.py" --dataset-root "${DATASET_ROOT}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/patch_v30_features.py" --dataset-root "${DATASET_ROOT}"
mapfile -t POLICY_FEATURE_ARGS < <("${PYTHON_BIN}" "${ROOT_DIR}/tools/build_policy_feature_args.py" --dataset-root "${DATASET_ROOT}")

if [[ -d "${EXP_DIR}" && "${FORCE_RESTART:-0}" == "1" ]]; then
  echo "[训练:A] FORCE_RESTART=1，正在删除旧输出目录: ${EXP_DIR}"
  rm -rf "${EXP_DIR}"
elif [[ -d "${EXP_DIR}" && "${RESUME:-false}" != "true" ]]; then
  echo "[训练:A] 输出目录已存在: ${EXP_DIR}" >&2
  echo "[训练:A] 如需重新开始，请加 FORCE_RESTART=1；如需续训，请加 RESUME=true。" >&2
  exit 1
fi

cat > "${SCRIPT_LOG_DIR}/run_config.env" <<EOF
EXP_NAME=${EXP_NAME}
DATASET_ROOT=${DATASET_ROOT}
GPU_IDS=${GPU_IDS}
NUM_GPUS=${NUM_GPUS}
SEED=${SEED}
TRAIN_STEPS=${TRAIN_STEPS}
BATCH_SIZE=${BATCH_SIZE}
LEARNING_RATE=${LEARNING_RATE}
ACT_CHUNK_SIZE=${ACT_CHUNK_SIZE}
ACT_N_ACTION_STEPS=${ACT_N_ACTION_STEPS}
EOF

if [[ "${USE_COMPAT_TRAIN}" == "1" ]]; then
  TRAIN_ENTRY=("${PYTHON_BIN}" "${ROOT_DIR}/tools/lerobot_train_compat.py")
else
  TRAIN_ENTRY=("${LEROBOT_TRAIN_BIN}")
fi

CMD=(
  "${TRAIN_ENTRY[@]}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--policy.type=act"
  "--policy.repo_id=${BASE_REPO_ID}_act_policy"
  "--policy.push_to_hub=false"
  '--rename_map={"actions":"action","state":"observation.state","image":"observation.images.image","wrist_image":"observation.images.wrist_image"}'
  "--output_dir=${EXP_DIR}"
  "--job_name=${EXP_NAME}"
  "--seed=${SEED}"
  "--steps=${TRAIN_STEPS}"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--eval_freq=${EVAL_FREQ}"
  "--optimizer.lr=${LEARNING_RATE}"
  "--policy.optimizer_lr=${LEARNING_RATE}"
  "--policy.optimizer_lr_backbone=${LEARNING_RATE}"
  "--policy.chunk_size=${ACT_CHUNK_SIZE}"
  "--policy.n_action_steps=${ACT_N_ACTION_STEPS}"
  "--wandb.enable=false"
)
CMD+=("${POLICY_FEATURE_ARGS[@]}")

if [[ "${RESUME:-false}" == "true" ]]; then
  CMD+=("--resume=true")
fi

echo "[训练:A] ${CMD[*]} ${COMMON_TRAIN_ARGS}" | tee "${SCRIPT_LOG_DIR}/command.txt"
echo "[训练:A] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" | tee -a "${SCRIPT_LOG_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}" ${COMMON_TRAIN_ARGS} 2>&1 | tee "${SCRIPT_LOG_DIR}/train.log"

if [[ -d "${EXP_DIR}" ]]; then
  mkdir -p "${EXP_DIR}/logs"
  cp -f "${SCRIPT_LOG_DIR}/command.txt" "${SCRIPT_LOG_DIR}/train.log" "${SCRIPT_LOG_DIR}/run_config.env" "${EXP_DIR}/logs/" 2>/dev/null || true
fi
