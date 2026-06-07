#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/configs/act_calvin_common.env"

if [[ "$#" -gt 0 ]]; then
  SPLITS=("$@")
else
  SPLITS=(splitA splitB splitC splitD)
fi

CONVERT_HOME="${CONVERT_HOME:-${DATA_ROOT}/_lerobot_convert_home}"
mkdir -p "${CONVERT_HOME}/local"

convert_one() {
  local split="$1"
  local split_root="${DATA_ROOT}/${split}"
  local repo_id=""

  case "${split}" in
    splitA) repo_id="${BASE_REPO_ID}" ;;
    splitB) repo_id="local/calvin_splitB" ;;
    splitC) repo_id="local/calvin_splitC" ;;
    splitD) repo_id="local/calvin_splitD" ;;
    *) repo_id="local/calvin_${split}" ;;
  esac

  if [[ ! -f "${split_root}/meta/info.json" ]]; then
    echo "[数据转换] 跳过 ${split}，未找到 ${split_root}/meta/info.json。"
    return
  fi

  if grep -q '"codebase_version"[[:space:]]*:[[:space:]]*"v3' "${split_root}/meta/info.json"; then
    echo "[数据转换] ${split} 已经是 v3 格式，跳过。"
    return
  fi

  "${PYTHON_BIN}" - "${split_root}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

split_root = Path(sys.argv[1])
episodes_path = split_root / "meta" / "episodes.jsonl"
stats_path = split_root / "meta" / "episodes_stats.jsonl"
if not episodes_path.exists() or not stats_path.exists():
    raise SystemExit(0)

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

episodes = load_jsonl(episodes_path)
length_by_idx = {}
for row in episodes:
    idx = row.get("episode_index", row.get("episode_idx", row.get("index")))
    length = row.get("length", row.get("episode_length", row.get("num_frames")))
    if length is None and "from" in row and "to" in row:
        length = row["to"] - row["from"]
    if idx is not None and length is not None:
        length_by_idx[idx] = int(length)

stats_rows = load_jsonl(stats_path)
changed = False

def add_count(obj, count):
    global changed
    if not isinstance(obj, dict):
        return
    looks_like_stats = any(k in obj for k in ("mean", "std", "min", "max"))
    if looks_like_stats and ("count" not in obj or not isinstance(obj.get("count"), list)):
        obj["count"] = [count]
        changed = True
    for value in obj.values():
        if isinstance(value, dict):
            add_count(value, count)

for row in stats_rows:
    idx = row.get("episode_index", row.get("episode_idx", row.get("index")))
    count = length_by_idx.get(idx)
    if count is None:
        continue
    add_count(row, count)

if changed:
    backup = stats_path.with_suffix(".jsonl.before_count_patch")
    if not backup.exists():
        shutil.copy2(stats_path, backup)
    with stats_path.open("w", encoding="utf-8") as f:
        for row in stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[数据转换] 已为缺失的 episodes_stats count 字段打补丁: {stats_path}")
else:
    print("[数据转换] episodes_stats count 字段无需补丁。")
PY

  local link_path="${CONVERT_HOME}/${repo_id}"
  local new_root="${link_path}_v30"
  mkdir -p "$(dirname "${link_path}")"
  if [[ -L "${link_path}" || -f "${link_path}" ]]; then
    rm -f "${link_path}"
  elif [[ -d "${link_path}" ]]; then
    rm -rf "${link_path}"
  fi
  mkdir -p "${link_path}"
  if ! cp -al "${split_root}/." "${link_path}/" 2>/dev/null; then
    echo "[数据转换] 硬链接复制失败，退回普通复制；这会占用更多磁盘空间。"
    cp -a "${split_root}/." "${link_path}/"
  fi
  if [[ -e "${new_root}" ]]; then
    if [[ "${FORCE_RECONVERT:-0}" == "1" ]]; then
      echo "[数据转换] FORCE_RECONVERT=1，正在删除旧转换目录: ${new_root}"
      rm -rf "${new_root}"
    else
      echo "[数据转换] 旧转换目录已存在: ${new_root}" >&2
      echo "[数据转换] 如需重新转换，请加 FORCE_RECONVERT=1。" >&2
      exit 1
    fi
  fi

  echo "[数据转换] 正在转换 ${split}: ${split_root}"
  echo "[数据转换] repo_id=${repo_id}"
  echo "[数据转换] HF_LEROBOT_HOME=${CONVERT_HOME}"

  local help_text
  help_text="$("${PYTHON_BIN}" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --help 2>&1 || true)"

  local args=(--repo-id="${repo_id}")
  if grep -q -- "--push-to-hub" <<< "${help_text}"; then
    args+=(--push-to-hub=false)
  fi
  if grep -q -- "--root" <<< "${help_text}"; then
    args+=(--root="${CONVERT_HOME}")
  fi

  HF_HUB_OFFLINE=1 HF_LEROBOT_HOME="${CONVERT_HOME}" \
    "${PYTHON_BIN}" -m lerobot.datasets.v30.convert_dataset_v21_to_v30 "${args[@]}"

  if [[ -d "${new_root}" ]]; then
    local final_root="${DATA_ROOT}/${split}_v30"
    rm -rf "${final_root}"
    mv "${new_root}" "${final_root}"
    echo "[数据转换] v3 数据已保存到: ${final_root}"
  elif [[ ! -L "${link_path}" && -f "${link_path}/meta/info.json" ]] && grep -q '"codebase_version"[[:space:]]*:[[:space:]]*"v3' "${link_path}/meta/info.json"; then
    local final_root="${DATA_ROOT}/${split}_v30"
    rm -rf "${final_root}"
    mv "${link_path}" "${final_root}"
    echo "[数据转换] v3 数据已保存到: ${final_root}"
  else
    echo "[数据转换] 转换命令结束，但没有找到预期 v3 输出目录: ${new_root}" >&2
    echo "[数据转换] 请检查以下目录中是否存在 *_v30:" >&2
    find "${CONVERT_HOME}" -maxdepth 4 -type d -name '*v30*' 2>/dev/null || true
    exit 1
  fi
}

for split in "${SPLITS[@]}"; do
  convert_one "${split}"
done

echo "[数据转换] 完成。"
