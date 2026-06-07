#!/usr/bin/env python3
"""修复 LeRobot v30 stats.json 中缺失的图像/重命名统计键。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "min": [0.0],
    "max": [1.0],
    "count": [1],
}


def is_image_key(key: str, feature: dict | None = None) -> bool:
    lowered = key.lower()
    if "image" in lowered or "visual" in lowered:
        return True
    if feature:
        dtype = str(feature.get("dtype", "")).lower()
        shape = feature.get("shape", [])
        return dtype in {"image", "video"} or (isinstance(shape, list) and len(shape) == 3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    info_path = args.dataset_root / "meta" / "info.json"
    stats_path = args.dataset_root / "meta" / "stats.json"
    if not info_path.exists() or not stats_path.exists():
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if not isinstance(features, dict) or not isinstance(stats, dict):
        return

    changed = False

    image_sources = [key for key in stats if is_image_key(key, features.get(key))]
    image_source = image_sources[0] if image_sources else None
    if image_source is None:
        stats["image"] = IMAGENET_STATS
        image_source = "image"
        print("[stats修复] 未找到图像统计量，已写入 ImageNet 默认统计到 image")
        changed = True

    for key, feature in features.items():
        if key not in stats and is_image_key(key, feature):
            stats[key] = stats[image_source]
            print(f"[stats修复] 为缺失图像键 {key} 复制统计量来源: {image_source}")
            changed = True

    aliases = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
        "state": "observation.state",
        "actions": "action",
    }
    for src, dst in aliases.items():
        if dst not in stats and src in stats:
            stats[dst] = stats[src]
            print(f"[stats修复] 为重命名键 {dst} 复制统计量来源: {src}")
            changed = True

    if "wrist_image" not in stats and "image" in stats:
        stats["wrist_image"] = stats["image"]
        print("[stats修复] 为缺失图像键 wrist_image 复制统计量来源: image")
        changed = True
    if "observation.images.wrist_image" not in stats and "wrist_image" in stats:
        stats["observation.images.wrist_image"] = stats["wrist_image"]
        print("[stats修复] 为重命名键 observation.images.wrist_image 复制统计量来源: wrist_image")
        changed = True

    if changed:
        backup = stats_path.with_suffix(".json.before_patch")
        if not backup.exists():
            shutil.copy2(stats_path, backup)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stats修复] 已更新 {stats_path}")


if __name__ == "__main__":
    main()
