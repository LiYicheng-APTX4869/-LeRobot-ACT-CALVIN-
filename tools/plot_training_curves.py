#!/usr/bin/env python3
"""根据导出的 TensorBoard CSV 绘制 A-only 与 ABC-joint 训练曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PREFERRED_TAGS = [
    "train/action_l1_loss",
    "action_l1_loss",
    "loss/action_l1",
    "train/loss",
    "loss",
    "lr",
    "learning_rate",
]


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["tag", "step", "value", "model"])
    df = pd.read_csv(path)
    if not df.empty:
        df["model"] = label
    return df


def choose_tags(df: pd.DataFrame) -> list[str]:
    tags = list(df["tag"].dropna().unique()) if not df.empty else []
    chosen = [tag for tag in PREFERRED_TAGS if tag in tags]
    if chosen:
        return chosen[:4]
    loss_like = [tag for tag in tags if "loss" in tag.lower() or "l1" in tag.lower()]
    return loss_like[:4] or tags[:4]


def plot_tag(df: pd.DataFrame, tag: str, output: Path) -> None:
    subset = df[df["tag"] == tag].sort_values("step")
    if subset.empty:
        return
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(data=subset, x="step", y="value", hue="model")
    plt.title(tag)
    plt.xlabel("训练 step")
    plt.ylabel("数值")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-csv", required=True, type=Path)
    parser.add_argument("--abc-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    df = pd.concat(
        [
            load_csv(args.a_csv, "A-only"),
            load_csv(args.abc_csv, "ABC-joint"),
        ],
        ignore_index=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        (args.output_dir / "NO_TENSORBOARD_DATA.txt").write_text(
            "尚未找到 TensorBoard scalar。请先完成训练。\n",
            encoding="utf-8",
        )
        print("没有找到 scalar 数据")
        return

    tags = choose_tags(df)
    for tag in tags:
        safe = tag.replace("/", "_").replace(" ", "_")
        plot_tag(df, tag, args.output_dir / f"{safe}.png")

    summary = df.groupby(["model", "tag"], as_index=False)["value"].agg(["min", "max", "last"]).reset_index()
    summary.to_csv(args.output_dir / "training_scalar_summary.csv", index=False)
    print(f"已绘制 {len(tags)} 个指标到 {args.output_dir}")


if __name__ == "__main__":
    main()
