#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

EXP_NAME="act_calvin_ABC_joint"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
SCRIPT_LOG_DIR="${OUTPUT_ROOT}/_script_logs/${EXP_NAME}"
MERGED_V2_ROOT="${DATA_ROOT}/splitABC_merged_v2"
MERGED_ROOT="${DATA_ROOT}/splitABC_merged_v30"
DATASET_REPO_ID="${JOINT_REPO_ID}_v30"
mkdir -p "${OUTPUT_ROOT}" "${SCRIPT_LOG_DIR}"
mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_datasets" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"

if [[ -d "${EXP_DIR}" && "${FORCE_RESTART:-0}" == "1" ]]; then
  echo "[训练:ABC] FORCE_RESTART=1，正在删除旧输出目录: ${EXP_DIR}"
  rm -rf "${EXP_DIR}"
elif [[ -d "${EXP_DIR}" && "${RESUME:-false}" != "true" ]]; then
  echo "[训练:ABC] 输出目录已存在: ${EXP_DIR}" >&2
  echo "[训练:ABC] 如需重新开始，请加 FORCE_RESTART=1；如需续训，请加 RESUME=true。" >&2
  exit 1
fi

for split in splitA splitB splitC; do
  if [[ ! -d "${DATA_ROOT}/${split}/data" || ! -f "${DATA_ROOT}/${split}/meta/info.json" ]]; then
    echo "[训练:ABC] 缺少原始 ${split} 数据目录: ${DATA_ROOT}/${split}" >&2
    echo "[训练:ABC] 请先确认 splitA/splitB/splitC 原始 v2 数据仍在本地。" >&2
    exit 1
  fi
done

if [[ ! -d "${MERGED_ROOT}/data" || ! -f "${MERGED_ROOT}/meta/info.json" || "${FORCE_REMERGE:-0}" == "1" ]]; then
  if [[ "${FORCE_REMERGE:-0}" == "1" ]]; then
    rm -rf "${MERGED_V2_ROOT}" "${MERGED_ROOT}" "${DATA_ROOT}/splitABC_merged_v2_v30"
  fi

  echo "[训练:ABC] 正在合并原始 v2 splitA/splitB/splitC: ${MERGED_V2_ROOT}"
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/merge_lerobot_splits.py" \
    --input-root "${DATA_ROOT}" \
    --splits splitA splitB splitC \
    --output-root "${MERGED_V2_ROOT}" \
    --repo-id "${DATASET_REPO_ID}"
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/check_episode_indices.py" \
    --dataset-root "${MERGED_V2_ROOT}" \
    --limit 20

  echo "[训练:ABC] 正在将 ABC 联合集转换为 v30。"
  FORCE_RECONVERT=1 bash "${ROOT_DIR}/scripts/01_convert_data_v30.sh" splitABC_merged_v2
  if [[ -d "${DATA_ROOT}/splitABC_merged_v2_v30" ]]; then
    rm -rf "${MERGED_ROOT}"
    mv "${DATA_ROOT}/splitABC_merged_v2_v30" "${MERGED_ROOT}"
  fi
fi

if [[ ! -d "${MERGED_ROOT}/data" || ! -f "${MERGED_ROOT}/meta/info.json" ]]; then
  echo "[训练:ABC] 没有找到 ABC v30 联合数据集: ${MERGED_ROOT}" >&2
  exit 1
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/check_episode_indices.py" \
  --dataset-root "${MERGED_ROOT}" \
  --limit 20

"${PYTHON_BIN}" "${ROOT_DIR}/tools/patch_v30_stats.py" --dataset-root "${MERGED_ROOT}"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/patch_v30_features.py" --dataset-root "${MERGED_ROOT}"
mapfile -t POLICY_FEATURE_ARGS < <("${PYTHON_BIN}" "${ROOT_DIR}/tools/build_policy_feature_args.py" --dataset-root "${MERGED_ROOT}")

cat > "${SCRIPT_LOG_DIR}/run_config.env" <<EOF
EXP_NAME=${EXP_NAME}
DATASET_ROOT=${MERGED_ROOT}
SOURCE_SPLITS=splitA,splitB,splitC
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
  "--dataset.root=${MERGED_ROOT}"
  "--policy.type=act"
  "--policy.repo_id=${DATASET_REPO_ID}_act_policy"
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

echo "[训练:ABC] ${CMD[*]} ${COMMON_TRAIN_ARGS}" | tee "${SCRIPT_LOG_DIR}/command.txt"
echo "[训练:ABC] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" | tee -a "${SCRIPT_LOG_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}" ${COMMON_TRAIN_ARGS} 2>&1 | tee "${SCRIPT_LOG_DIR}/train.log"

if [[ -d "${EXP_DIR}" ]]; then
  mkdir -p "${EXP_DIR}/logs"
  cp -f "${SCRIPT_LOG_DIR}/command.txt" "${SCRIPT_LOG_DIR}/train.log" "${SCRIPT_LOG_DIR}/run_config.env" "${EXP_DIR}/logs/" 2>/dev/null || true
fi
