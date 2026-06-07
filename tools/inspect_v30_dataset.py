#!/usr/bin/env python3
"""检查 LeRobot v30 数据集的 feature、parquet schema、视频和样本 keys。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/calvin_splitA_v30")
    args = parser.parse_args()

    root = args.dataset_root
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"

    print("== paths ==")
    print("root:", root)
    print("info exists:", info_path.exists())
    print("stats exists:", stats_path.exists())

    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        print("\n== info.features ==")
        for key, value in info.get("features", {}).items():
            print(key, value)

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print("\n== stats keys ==")
        print(list(stats.keys()))

    print("\n== data parquet schema ==")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    print("num parquet:", len(parquet_files))
    if parquet_files:
        try:
            import pyarrow.parquet as pq

            print("first parquet:", parquet_files[0])
            print(pq.read_schema(parquet_files[0]))
        except Exception as exc:
            print("failed to read parquet schema:", repr(exc))

    print("\n== videos ==")
    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        video_files.extend((root / "videos").rglob(ext))
    print("num videos:", len(video_files))
    for path in sorted(video_files)[:10]:
        print(path)

    print("\n== LeRobotDataset sample keys ==")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, root=root)
        sample = ds[0]
        print("len:", len(ds))
        print("keys:", list(sample.keys()))
        for key, value in sample.items():
            shape = getattr(value, "shape", None)
            print(key, type(value), shape)
    except Exception as exc:
        import traceback

        print("failed to load sample:", repr(exc))
        traceback.print_exc()


if __name__ == "__main__":
    main()
