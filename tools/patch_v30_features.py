#!/usr/bin/env python3
"""为 LeRobot v3 info.json 补齐 ACT 训练需要的 feature 类型。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def classify_feature(key: str, feature: dict) -> str | None:
    lowered = key.lower()
    dtype = str(feature.get("dtype", "")).lower()
    shape = feature.get("shape", [])
    if "action" in lowered:
        return "ACTION"
    if "image" in lowered or "visual" in lowered or dtype in {"image", "video"}:
        return "VISUAL"
    if "state" in lowered or "proprio" in lowered or "robot" in lowered:
        return "STATE"
    if isinstance(shape, list) and len(shape) == 3:
        return "VISUAL"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    info_path = args.dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    if not isinstance(features, dict):
        return

    changed = False
    for key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        inferred = classify_feature(key, feature)
        if inferred and feature.get("type") != inferred:
            feature["type"] = inferred
            print(f"[feature修复] {key} -> {inferred}")
            changed = True

    if changed:
        backup = info_path.with_suffix(".json.before_feature_patch")
        if not backup.exists():
            shutil.copy2(info_path, backup)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[feature修复] 已更新 {info_path}")


if __name__ == "__main__":
    main()

