#!/usr/bin/env python3
"""在未见过的 LeRobot split 上离线评估策略动作误差。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.lerobot_train_compat  # noqa: F401,E402  # 应用本项目的 LeRobot 兼容补丁


def add_aliases(batch: dict) -> dict:
    batch = dict(batch)
    aliases = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
        "state": "observation.state",
        "actions": "action",
    }
    for src, dst in aliases.items():
        if src in batch and (dst not in batch or batch[dst] is None):
            batch[dst] = batch[src]
    return batch


def move_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def first_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 3:
        return action[:, 0, :]
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/calvin_splitD_v30")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", choices=["select_action", "forward_loss"], default="select_action")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    episodes = (
        list(range(args.episode_start, args.episode_start + args.episodes))
        if args.episodes > 0
        else None
    )
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=episodes)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = move_to_device(add_aliases(batch), device)
            target = first_action(batch["action"])
            if args.method == "forward_loss":
                was_training = policy.training
                policy.train()
                loss, output = policy.forward(batch)
                if not was_training:
                    policy.eval()
                metrics = {"batch": batch_idx, "loss": scalar(loss)}
                if isinstance(output, dict):
                    for key, value in output.items():
                        val = scalar(value)
                        if val is not None:
                            metrics[key] = val
            else:
                obs_batch = {key: value for key, value in batch.items() if key not in {"action", "actions"}}
                if hasattr(policy, "reset"):
                    policy.reset()
                pred = policy.select_action(obs_batch)
                if isinstance(pred, dict):
                    pred = pred.get("action", None)
                    if pred is None:
                        pred = pred.get("actions", None)
                pred = first_action(pred)
                if pred.shape[0] != target.shape[0]:
                    n = min(pred.shape[0], target.shape[0])
                    pred = pred[:n]
                    target = target[:n]
                if pred.shape != target.shape:
                    pred = pred.reshape_as(target)
                diff = pred - target
                metrics = {
                    "batch": batch_idx,
                    "action_l1": float(diff.abs().mean().detach().cpu()),
                    "action_mse": float(diff.pow(2).mean().detach().cpu()),
                    "action_l2": float(diff.norm(dim=-1).mean().detach().cpu()),
                }
            rows.append(metrics)
            for key, value in metrics.items():
                if key == "batch" or value is None:
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    summary = {
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "policy_path": str(args.policy_path),
        "num_frames": len(dataset),
        "episodes": episodes,
        "episode_start": args.episode_start,
        "num_eval_batches": len(rows),
        "batch_size": args.batch_size,
        "method": args.method,
        "metrics": {key: sums[key] / counts[key] for key in sorted(sums)},
    }

    preferred_l1_keys = ["action_l1", "l1_loss", "action_l1_loss", "loss/action_l1"]
    for key in preferred_l1_keys:
        if key in summary["metrics"]:
            summary["action_l1"] = summary["metrics"][key]
            break
    if "action_l1" not in summary and "loss" in summary["metrics"]:
        summary["action_l1"] = summary["metrics"]["loss"]

    suffix = ""
    if args.episodes > 0:
        suffix = f"_ep{args.episode_start:05d}_{args.episode_start + args.episodes - 1:05d}"

    csv_path = args.output_dir / f"offline_action_error_batches{suffix}.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    (args.output_dir / f"offline_action_error{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "offline_action_error.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
