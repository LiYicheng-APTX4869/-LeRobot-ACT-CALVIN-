#!/usr/bin/env python3
"""汇总 splitD 分块离线动作误差评估结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    paths = sorted(args.eval_dir.glob("offline_action_error_ep*.json"))
    if not paths:
        raise SystemExit(f"没有找到分块结果: {args.eval_dir}/offline_action_error_ep*.json")

    weighted_sums: dict[str, float] = {}
    total_weight = 0
    chunks = []
    total_frames = 0
    total_batches = 0

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        weight = int(data.get("num_eval_batches", 0) or 0)
        if weight <= 0:
            continue
        total_weight += weight
        total_batches += weight
        total_frames += int(data.get("num_frames", 0) or 0)
        chunks.append(
            {
                "path": str(path),
                "episode_start": data.get("episode_start"),
                "num_eval_batches": weight,
                "num_frames": data.get("num_frames"),
                "metrics": metrics,
            }
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                weighted_sums[key] = weighted_sums.get(key, 0.0) + float(value) * weight

    if total_weight <= 0:
        raise SystemExit("分块结果中没有可汇总的 batch")

    summary = {
        "eval_dir": str(args.eval_dir),
        "num_chunks": len(chunks),
        "num_eval_batches": total_batches,
        "num_frames_sum_over_chunks": total_frames,
        "metrics": {key: value / total_weight for key, value in sorted(weighted_sums.items())},
        "chunks": chunks,
    }
    if "action_l1" in summary["metrics"]:
        summary["action_l1"] = summary["metrics"]["action_l1"]
    if "action_mse" in summary["metrics"]:
        summary["action_mse"] = summary["metrics"]["action_mse"]

    output = args.output_json or (args.eval_dir / "offline_action_error_aggregated.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
