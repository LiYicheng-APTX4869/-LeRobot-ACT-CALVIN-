#!/usr/bin/env python3
"""汇总评估输出，并可选绘制模型对比图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def add_metric(rows: list[dict], path: Path, metric: str, value) -> None:
    if value is None:
        return
    try:
        rows.append({"source": str(path), "metric": metric, "value": float(value)})
    except (TypeError, ValueError):
        return


def find_metrics(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in eval_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            if "success_rate" in data:
                add_metric(rows, path, "success_rate", data["success_rate"])
            if "mean_action_error" in data:
                add_metric(rows, path, "mean_action_error", data["mean_action_error"])
            if "action_l1" in data:
                add_metric(rows, path, "action_l1", data["action_l1"])
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    add_metric(rows, path, key, value)
            if "success" in data and isinstance(data["success"], (bool, int, float)):
                add_metric(rows, path, "success", data["success"])
    for path in eval_dir.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ["success_rate", "success", "mean_action_error", "action_l1_error"]:
                        if key in row and row[key] not in ("", None):
                            rows.append({"source": str(path), "metric": key, "value": float(row[key])})
        except Exception:
            continue
    return rows


def write_placeholder(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "success_rate": None,
        "mean_action_error": None,
        "note": "由于没有可用评估输出，已创建占位结果。请运行课程评估脚本或 LeRobot 评估器后替换。",
    }
    (eval_dir / "success_rate.json").write_text(json.dumps(placeholder, indent=2), encoding="utf-8")


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for metric in sorted({r["metric"] for r in rows}):
        vals = [r["value"] for r in rows if r["metric"] == metric]
        if vals:
            out[metric] = sum(vals) / len(vals)
    return out


def plot_compare(label_a: str, eval_a: Path, label_b: str, eval_b: Path, plot_dir: Path) -> None:
    rows_a = find_metrics(eval_a)
    rows_b = find_metrics(eval_b)
    agg_a = aggregate(rows_a)
    agg_b = aggregate(rows_b)
    metrics = [m for m in ["success_rate", "success", "mean_action_error", "action_l1_error"] if m in agg_a or m in agg_b]
    if not metrics:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        labels = []
        values = []
        for label, agg in [(label_a, agg_a), (label_b, agg_b)]:
            if metric in agg:
                labels.append(label)
                values.append(agg[metric])
        plt.figure(figsize=(5.5, 4.2))
        plt.bar(labels, values, color=["#2563eb", "#16a34a"][: len(labels)])
        plt.title(f"splitD {metric} 对比")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(plot_dir / f"splitD_{metric}_comparison.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--label", default="model")
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--compare-label", default="comparison")
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--write-placeholder", action="store_true")
    args = parser.parse_args()

    if args.write_placeholder:
        write_placeholder(args.eval_dir)

    rows = find_metrics(args.eval_dir)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    with (args.eval_dir / "eval_metrics_flat.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    (args.eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.compare_dir and args.plot_dir:
        plot_compare(args.label, args.eval_dir, args.compare_label, args.compare_dir, args.plot_dir)
    print(f"已写入评估汇总: {args.eval_dir}")


if __name__ == "__main__":
    main()
