#!/usr/bin/env python3
"""将 TensorBoard event 中的 scalar 导出为整洁 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise SystemExit(f"需要安装 tensorboard: {exc}")

    event_files = sorted(args.log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"在 {args.log_dir} 下没有找到 TensorBoard event 文件")

    rows = []
    for event_file in event_files:
        acc = EventAccumulator(str(event_file.parent))
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for ev in acc.Scalars(tag):
                rows.append(
                    {
                        "run": str(event_file.parent.relative_to(args.log_dir)),
                        "tag": tag,
                        "step": ev.step,
                        "value": ev.value,
                        "wall_time": ev.wall_time,
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value", "wall_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {len(rows)} 行 scalar 到 {args.output_csv}")


if __name__ == "__main__":
    main()
