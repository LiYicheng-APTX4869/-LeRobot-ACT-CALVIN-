#!/usr/bin/env python3
"""检查 LeRobot parquet 内部 episode_index 是否和文件名/metadata 对齐。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def read_unique_episode_indices(path: Path) -> list[int]:
    table = pq.read_table(path, columns=["episode_index"])
    values = table.column("episode_index").to_pylist()
    return sorted(set(int(v) for v in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    files = sorted((args.dataset_root / "data").rglob("episode_*.parquet"))
    if not files:
        files = sorted((args.dataset_root / "data").rglob("*.parquet"))
    print(f"dataset_root={args.dataset_root}")
    print(f"num_parquet={len(files)}")
    if not files:
        raise SystemExit("没有找到 parquet 文件")

    sample_files = files[: args.limit // 2] + files[-args.limit // 2 :]
    bad = []
    for path in sample_files:
        expected = None
        match = EPISODE_RE.search(path.name)
        if match:
            expected = int(match.group(1))
        unique = read_unique_episode_indices(path)
        ok = expected is None or unique == [expected]
        print(f"{path.relative_to(args.dataset_root)} expected={expected} actual={unique[:5]} ok={ok}")
        if not ok:
            bad.append((path, expected, unique))

    if bad:
        raise SystemExit(f"发现 {len(bad)} 个 parquet 的 episode_index 未对齐")

    all_indices = set()
    for path in files:
        all_indices.update(read_unique_episode_indices(path))

    expected_total = None
    info_path = args.dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        expected_total = info.get("total_episodes")

    if all_indices:
        min_idx = min(all_indices)
        max_idx = max(all_indices)
        print(f"unique_episode_indices={len(all_indices)} min={min_idx} max={max_idx}")
        if expected_total is not None:
            print(f"meta.total_episodes={expected_total}")
            missing = set(range(int(expected_total))) - all_indices
            if missing:
                preview = sorted(missing)[:20]
                raise SystemExit(f"缺少 {len(missing)} 个 episode_index，前几个: {preview}")
            print("episode_index 全量连续性检查通过")
    print("episode_index 检查通过")


if __name__ == "__main__":
    main()
