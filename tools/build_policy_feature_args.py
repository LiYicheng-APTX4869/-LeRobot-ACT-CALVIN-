#!/usr/bin/env python3
"""从 LeRobot v3 info.json 生成 ACT policy feature CLI 参数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RENAME_MAP = {
    "state": "observation.state",
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def feature_type(key: str, feature: dict) -> str | None:
    explicit = feature.get("type")
    if explicit in {"ACTION", "STATE", "VISUAL"}:
        return explicit
    lowered = key.lower()
    dtype = str(feature.get("dtype", "")).lower()
    shape = feature.get("shape", [])
    if "action" in lowered:
        return "ACTION"
    if "state" in lowered or "proprio" in lowered:
        return "STATE"
    if "image" in lowered or "visual" in lowered or dtype in {"image", "video"}:
        return "VISUAL"
    if isinstance(shape, list) and len(shape) == 3:
        return "VISUAL"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    info_path = args.dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})

    input_features = {}
    output_features = {}
    for raw_key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        ftype = feature_type(raw_key, feature)
        shape = feature.get("shape")
        if ftype is None or shape is None:
            continue
        key = raw_key if ftype == "ACTION" else RENAME_MAP.get(raw_key, raw_key)
        spec = {"type": ftype, "shape": shape}
        if ftype == "ACTION":
            output_features[key] = spec
        elif ftype in {"STATE", "VISUAL"}:
            input_features[key] = spec

    if not output_features:
        raise SystemExit("无法从 info.json 推断 ACTION feature，请检查 meta/info.json")

    print("--policy.input_features=" + json.dumps(input_features, separators=(",", ":")))
    print("--policy.output_features=" + json.dumps(output_features, separators=(",", ":")))


if __name__ == "__main__":
    main()
