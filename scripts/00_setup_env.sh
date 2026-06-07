#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

TMP_ROOT="${TMP_ROOT:-${ROOT_DIR}/.tmp}"
mkdir -p "${TMP_ROOT}"

export TMPDIR="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"

echo "[环境配置] Python: $(${PYTHON_BIN} --version)"
echo "[环境配置] 临时目录: ${TMP_ROOT}"
echo "[环境配置] pip 缓存: 已默认禁用，避免大 wheel 写满用户目录。"

echo "[环境配置] 磁盘空间检查:"
df -h "${ROOT_DIR}" || true
df -h "${TMP_ROOT}" || true

if [[ "${INSTALL_CONDA_BUILD_TOOLS}" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "[环境配置] 正在安装 conda 编译工具链，用于编译 evdev 等源码包。"
    conda install -y -c conda-forge c-compiler cxx-compiler binutils
  else
    echo "[环境配置] 未找到 conda，跳过 conda 编译工具链安装。"
  fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]]; then
    export CC="${CC:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc}"
  fi
  if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
    export CXX="${CXX:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++}"
  fi
  if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-as" ]]; then
    export AS="${AS:-${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-as}"
  fi
fi

echo "[环境配置] CC=${CC:-系统默认}"
echo "[环境配置] CXX=${CXX:-系统默认}"
echo "[环境配置] AS=${AS:-系统默认}"

echo "[环境配置] 正在升级 pip。"
"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade pip

echo "[环境配置] 正在安装 LeRobot 和实验工具依赖。"
echo "[环境配置] LeRobot 安装包: ${LEROBOT_PACKAGE}"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "${LEROBOT_PACKAGE}" \
  huggingface_hub \
  hf_transfer \
  tensorboard \
  pandas \
  pyarrow \
  matplotlib \
  seaborn \
  opencv-python \
  tqdm \
  ${PIP_EXTRA_ARGS:-}

echo "[环境配置] 正在验证 Python 包导入。"
"${PYTHON_BIN}" - <<'PY'
import importlib
for name in ["lerobot", "torch", "tensorboard", "pandas", "matplotlib", "cv2"]:
    importlib.import_module(name)
    print(f"导入成功: {name}")
PY

echo "[环境配置] GPU 检查:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "未找到 nvidia-smi；只有在你确认当前环境不需要 GPU 检查时才继续。"
fi
