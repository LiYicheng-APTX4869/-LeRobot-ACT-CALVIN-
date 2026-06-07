#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

mkdir -p "${DATA_ROOT}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ "$#" -gt 0 ]]; then
  SPLITS=("$@")
else
  SPLITS=(splitA splitB splitC splitD)
fi

echo "[数据下载] 数据集: ${HF_DATASET_ID}"
echo "[数据下载] 保存路径: ${DATA_ROOT}"
echo "[数据下载] splits: ${SPLITS[*]}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "[数据下载] Hugging Face Endpoint: ${HF_ENDPOINT}"
fi
if [[ -n "${https_proxy:-}" || -n "${HTTPS_PROXY:-}" ]]; then
  echo "[数据下载] 检测到 HTTPS 代理配置。"
fi

if command -v hf >/dev/null 2>&1; then
  DOWNLOAD_CMD=(hf download)
else
  DOWNLOAD_CMD=(huggingface-cli download)
fi

INCLUDE_ARGS=()
for split in "${SPLITS[@]}"; do
  INCLUDE_ARGS+=("${split}/*")
done

"${DOWNLOAD_CMD[@]}" "${HF_DATASET_ID}" \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}" \
  --include "${INCLUDE_ARGS[@]}"

"${PYTHON_BIN}" - "${SPLITS[@]}" <<PY
import sys
from pathlib import Path

root = Path(r"${DATA_ROOT}")
required = sys.argv[1:]
for split in required:
    for rel in ["data", "meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
        path = root / split / rel
        if not path.exists():
            raise SystemExit(f"缺少文件或目录: {path}")
print("数据下载完整性检查通过")
PY
