#!/usr/bin/env python3
"""从训练日志和 splitD 离线评估 JSON 生成报告图表。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"step:(?P<step>\d+).*?loss:(?P<loss>[0-9.eE+-]+).*?grdn:(?P<grad>[0-9.eE+-]+).*?lr:(?P<lr>[0-9.eE+-]+)"
)


def parse_train_log(path: Path, model: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "model": model,
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad")),
                "lr": float(match.group("lr")),
            }
        )
    return rows


def load_eval(path: Path, model: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "model": model,
        "action_l1": data.get("action_l1", metrics.get("action_l1")),
        "action_mse": data.get("action_mse", metrics.get("action_mse")),
        "action_l2": metrics.get("action_l2"),
        "num_eval_batches": data.get("num_eval_batches"),
        "num_frames": data.get("num_frames_sum_over_chunks", data.get("num_frames")),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        plt.plot([r["step"] for r in subset], [r["loss"] for r in subset], label=model)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss From Logs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_eval(rows: list[dict], output_dir: Path) -> None:
    rows = [row for row in rows if row]
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["action_l1", "action_mse", "action_l2"]:
        usable = [row for row in rows if row.get(metric) is not None]
        if not usable:
            continue
        plt.figure(figsize=(5.5, 4.2))
        plt.bar([row["model"] for row in usable], [float(row[metric]) for row in usable])
        plt.ylabel(metric)
        plt.title(f"splitD {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"splitD_{metric}_comparison.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    train_rows = []
    train_rows += parse_train_log(args.output_root / "_script_logs/act_calvin_A_only/train.log", "A-only")
    train_rows += parse_train_log(args.output_root / "_script_logs/act_calvin_ABC_joint/train.log", "ABC-joint")
    save_csv(args.output_dir / "training_log_scalars.csv", train_rows)
    plot_loss(train_rows, args.output_dir / "training_loss_from_logs.png")

    eval_rows = [
        load_eval(args.output_root / "act_calvin_A_only/eval_splitD/offline_action_error_aggregated.json", "A-only"),
        load_eval(args.output_root / "act_calvin_ABC_joint/eval_splitD/offline_action_error_aggregated.json", "ABC-joint"),
    ]
    eval_rows = [row for row in eval_rows if row is not None]
    save_csv(args.output_dir / "splitD_offline_action_error_summary.csv", eval_rows)
    plot_eval(eval_rows, args.output_dir)
    print(f"已从日志和离线评估结果生成图表: {args.output_dir}")


if __name__ == "__main__":
    main()
