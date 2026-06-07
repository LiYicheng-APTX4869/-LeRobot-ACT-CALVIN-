#!/usr/bin/env python3
"""Build integrity checks and report-ready figures from ACT/CALVIN outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


MODELS = {
    "A-only": "act_calvin_A_only",
    "ABC-joint": "act_calvin_ABC_joint",
}

COLORS = {
    "A-only": "#2563EB",
    "ABC-joint": "#16A34A",
    "neutral": "#64748B",
    "dark": "#0F172A",
    "warn": "#D97706",
}

LOG_RE = re.compile(
    r"\bstep:(?P<step>\d+(?:\.\d+)?[KMB]?)\s+"
    r"smpl:(?P<samples>\d+(?:\.\d+)?[KMB]?)\s+"
    r"ep:(?P<episodes>\d+(?:\.\d+)?[KMB]?)\s+"
    r"epch:(?P<epoch>[0-9.]+)\s+"
    r"loss:(?P<loss>[0-9.eE+-]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.eE+-]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.eE+-]+)\s+"
    r"data_s:(?P<data_s>[0-9.eE+-]+)"
)


def parse_compact_number(value: str) -> int:
    suffix = value[-1].upper()
    if suffix in {"K", "M", "B"}:
        scale = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
        return int(round(float(value[:-1]) * scale))
    return int(float(value))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_train_log(output_root: Path, label: str, exp: str) -> list[dict]:
    candidates = [
        output_root / exp / "logs" / "train.log",
        output_root / "_script_logs" / exp / "train.log",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []

    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        for match in LOG_RE.finditer(line):
            rows.append(
                {
                    "model": label,
                    "step": parse_compact_number(match.group("step")),
                    "samples": parse_compact_number(match.group("samples")),
                    "episodes_seen": parse_compact_number(match.group("episodes")),
                    "epoch": float(match.group("epoch")),
                    "loss": float(match.group("loss")),
                    "grad_norm": float(match.group("grad_norm")),
                    "lr": float(match.group("lr")),
                    "update_s": float(match.group("update_s")),
                    "data_s": float(match.group("data_s")),
                }
            )

    rows.sort(key=lambda row: row["step"])
    deduped = {}
    for row in rows:
        deduped[row["step"]] = row
    return list(deduped.values())


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    smoothed = []
    total = 0.0
    queue = []
    for value in values:
        total += value
        queue.append(value)
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def load_eval(output_root: Path, label: str, exp: str) -> dict | None:
    path = output_root / exp / "eval_splitD" / "offline_action_error_aggregated.json"
    if not path.exists():
        return None
    data = read_json(path)
    metrics = data.get("metrics", {})
    return {
        "model": label,
        "exp": exp,
        "path": str(path),
        "num_chunks": data.get("num_chunks"),
        "num_eval_batches": data.get("num_eval_batches"),
        "num_frames": data.get("num_frames_sum_over_chunks"),
        "action_l1": float(metrics.get("action_l1")),
        "action_l2": float(metrics.get("action_l2")),
        "action_mse": float(metrics.get("action_mse")),
        "chunks": data.get("chunks", []),
    }


def build_integrity(output_root: Path, train_rows: dict[str, list[dict]], evals: dict[str, dict | None]) -> tuple[list[dict], list[str]]:
    rows = []
    notes = []
    expected_ckpts = [f"{step:06d}" for step in range(10_000, 100_001, 10_000)] + ["last"]

    for label, exp in MODELS.items():
        exp_root = output_root / exp
        train_log = exp_root / "logs" / "train.log"
        if not train_log.exists():
            train_log = output_root / "_script_logs" / exp / "train.log"
        log_text = train_log.read_text(errors="ignore") if train_log.exists() else ""

        rows.append({"item": f"{label} training log exists", "status": "OK" if train_log.exists() else "MISSING", "detail": str(train_log)})
        rows.append({"item": f"{label} training reached end", "status": "OK" if "End of training" in log_text else "CHECK", "detail": "End of training marker"})
        rows.append({"item": f"{label} training traceback/error", "status": "OK" if ("Traceback" not in log_text and "ERROR" not in log_text) else "CHECK", "detail": "No Traceback/ERROR in train.log"})

        missing = []
        for ckpt in expected_ckpts:
            model_path = exp_root / "checkpoints" / ckpt / "pretrained_model" / "model.safetensors"
            state_path = exp_root / "checkpoints" / ckpt / "training_state" / "training_step.json"
            if not model_path.exists() or (ckpt != "last" and not state_path.exists()):
                missing.append(ckpt)
        rows.append({"item": f"{label} checkpoints 010000-100000 + last", "status": "OK" if not missing else "MISSING", "detail": ",".join(missing) if missing else "all expected checkpoints present"})

        parsed = train_rows.get(label, [])
        final_step = max((row["step"] for row in parsed), default=None)
        rows.append({"item": f"{label} parsed train scalars", "status": "OK" if final_step == 100_000 else "CHECK", "detail": f"rows={len(parsed)}, final_step={final_step}"})

        run_config = exp_root / "logs" / "run_config.env"
        rows.append({"item": f"{label} run config", "status": "OK" if run_config.exists() else "MISSING", "detail": str(run_config)})

        eval_data = evals.get(label)
        if eval_data is None:
            rows.append({"item": f"{label} splitD aggregated eval", "status": "MISSING", "detail": "offline_action_error_aggregated.json"})
        else:
            rows.append({"item": f"{label} splitD aggregated eval", "status": "OK", "detail": f"chunks={eval_data['num_chunks']}, batches={eval_data['num_eval_batches']}, frames={eval_data['num_frames']}"})

        success_placeholder = exp_root / "eval_splitD" / "success_rate.json"
        if success_placeholder.exists():
            data = read_json(success_placeholder)
            if data.get("success_rate") is None:
                notes.append(f"{label}: success_rate.json is a placeholder and should be ignored for this offline-action-error report.")

    available_evals = [data for data in evals.values() if data is not None]
    if len(available_evals) == 2:
        a, b = available_evals
        same = (
            a["num_chunks"] == b["num_chunks"]
            and a["num_eval_batches"] == b["num_eval_batches"]
            and a["num_frames"] == b["num_frames"]
            and [c.get("episode_start") for c in a["chunks"]] == [c.get("episode_start") for c in b["chunks"]]
        )
        rows.append({"item": "A-only vs ABC splitD eval coverage", "status": "OK" if same else "CHECK", "detail": "same chunks/batches/frames" if same else "coverage differs"})

    return rows, notes


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": COLORS["dark"],
            "axes.titlecolor": COLORS["dark"],
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.65,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def kfmt(x: float, _pos: int) -> str:
    if abs(x) >= 1000:
        return f"{x / 1000:.0f}K"
    return f"{x:.0f}"


def plot_training_loss(train_rows: dict[str, list[dict]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for label, rows in train_rows.items():
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        losses = [r["loss"] for r in rows]
        smooth = moving_average(losses, 25)
        ax.plot(steps, losses, color=COLORS[label], alpha=0.18, linewidth=1.0)
        ax.plot(steps, smooth, color=COLORS[label], linewidth=2.4, label=label)
    ax.set_title("ACT Training Loss Convergence")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action L1 training loss")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "01_training_loss_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for label, rows in train_rows.items():
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        losses = moving_average([r["loss"] for r in rows], 25)
        ax.plot(steps, losses, color=COLORS[label], linewidth=2.4, label=label)
    ax.set_yscale("log")
    ax.set_title("Training Loss on Log Scale")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss, log scale")
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "02_training_loss_log_scale.png", dpi=300)
    plt.close(fig)


def plot_eval_bars(eval_rows: list[dict], out_dir: Path) -> None:
    metrics = [("action_l1", "Action L1"), ("action_mse", "Action MSE"), ("action_l2", "Action L2")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (key, title) in zip(axes, metrics):
        values = [row[key] for row in eval_rows]
        labels = [row["model"] for row in eval_rows]
        ax.bar(labels, values, color=[COLORS[label] for label in labels], width=0.55)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.set_ylim(0, max(values) * 1.22)
        for idx, value in enumerate(values):
            ax.text(idx, value + max(values) * 0.03, f"{value:.4f}", ha="center", va="bottom", color=COLORS["dark"], fontweight="bold")
        if len(values) == 2 and values[0] > 0:
            gain = (values[0] - values[1]) / values[0] * 100
            ax.text(0.5, max(values) * 1.12, f"{gain:.1f}% lower", ha="center", color="#15803D", fontweight="bold")
    fig.suptitle("Zero-shot splitD Offline Action Error", fontsize=15, fontweight="bold", color=COLORS["dark"])
    fig.tight_layout()
    fig.savefig(out_dir / "03_splitD_action_error_grouped_bars.png", dpi=300)
    plt.close(fig)


def chunk_rows(evals: dict[str, dict]) -> list[dict]:
    rows = []
    for label, data in evals.items():
        if data is None:
            continue
        for chunk in data["chunks"]:
            metrics = chunk["metrics"]
            start = int(chunk["episode_start"])
            rows.append(
                {
                    "model": label,
                    "episode_start": start,
                    "episode_range": f"{start}-{start + 199}",
                    "num_eval_batches": chunk.get("num_eval_batches"),
                    "num_frames": chunk.get("num_frames"),
                    "action_l1": metrics.get("action_l1"),
                    "action_l2": metrics.get("action_l2"),
                    "action_mse": metrics.get("action_mse"),
                }
            )
    return rows


def load_batch_rows(output_root: Path) -> list[dict]:
    rows = []
    for label, exp in MODELS.items():
        eval_dir = output_root / exp / "eval_splitD"
        for path in sorted(eval_dir.glob("offline_action_error_batches_ep*.csv")):
            range_match = re.search(r"ep(\d+)_(\d+)", path.name)
            episode_range = f"{range_match.group(1)}-{range_match.group(2)}" if range_match else path.stem
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(
                        {
                            "model": label,
                            "episode_range": episode_range,
                            "batch": int(row["batch"]),
                            "action_l1": float(row["action_l1"]),
                            "action_l2": float(row["action_l2"]),
                            "action_mse": float(row["action_mse"]),
                        }
                    )
    return rows


def plot_chunk_lines(chunks: list[dict], out_dir: Path, metric: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.1))
    for label in MODELS:
        rows = sorted([r for r in chunks if r["model"] == label], key=lambda r: r["episode_start"])
        ax.plot(
            [r["episode_start"] for r in rows],
            [r[metric] for r in rows],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=COLORS[label],
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("splitD episode start")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=300)
    plt.close(fig)


def plot_improvement(chunks: list[dict], out_dir: Path) -> None:
    by_start = {}
    for row in chunks:
        by_start.setdefault(row["episode_start"], {})[row["model"]] = row
    rows = []
    for start, pair in sorted(by_start.items()):
        if "A-only" in pair and "ABC-joint" in pair:
            a = pair["A-only"]["action_l1"]
            b = pair["ABC-joint"]["action_l1"]
            rows.append((start, (a - b) / a * 100))
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    colors = ["#16A34A" if value >= 0 else "#DC2626" for _, value in rows]
    ax.bar([start for start, _ in rows], [value for _, value in rows], width=135, color=colors)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_title("ABC-joint Improvement over A-only by splitD Chunk")
    ax.set_xlabel("splitD episode start")
    ax.set_ylabel("Action L1 reduction (%)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "06_splitD_chunk_l1_improvement.png", dpi=300)
    plt.close(fig)


def plot_chunk_heatmap(chunks: list[dict], out_dir: Path) -> None:
    starts = sorted({row["episode_start"] for row in chunks})
    labels = list(MODELS)
    values = []
    for label in labels:
        by_start = {row["episode_start"]: row["action_l1"] for row in chunks if row["model"] == label}
        values.append([by_start[start] for start in starts])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu_r")
    ax.set_title("splitD Action L1 Heatmap by Evaluation Chunk")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(starts)), [str(start) for start in starts], rotation=45, ha="right")
    ax.set_xlabel("episode start")
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="#0F172A", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Action L1")
    fig.tight_layout()
    fig.savefig(out_dir / "08_splitD_chunk_action_l1_heatmap.png", dpi=300)
    plt.close(fig)


def plot_batch_distribution(batch_rows: list[dict], out_dir: Path) -> None:
    if not batch_rows:
        return
    labels = list(MODELS)
    values = [[row["action_l1"] for row in batch_rows if row["model"] == label] for label in labels]
    fig, ax = plt.subplots(figsize=(7.8, 5))
    parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(COLORS[label])
        body.set_edgecolor(COLORS[label])
        body.set_alpha(0.22)
    box = ax.boxplot(values, widths=0.32, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.72)
        patch.set_edgecolor(COLORS[label])
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Batch Action L1")
    ax.set_title("Batch-level Action L1 Distribution on splitD")
    ax.grid(True, axis="y")
    for idx, group in enumerate(values, start=1):
        ax.text(idx, max(group) * 1.02, f"mean={mean(group):.4f}\nn={len(group)}", ha="center", va="bottom", fontsize=9, color=COLORS["dark"])
    fig.tight_layout()
    fig.savefig(out_dir / "09_splitD_batch_action_l1_distribution.png", dpi=300)
    plt.close(fig)


def plot_metric_table(eval_rows: list[dict], train_rows: dict[str, list[dict]], out_dir: Path) -> None:
    a = next(r for r in eval_rows if r["model"] == "A-only")
    b = next(r for r in eval_rows if r["model"] == "ABC-joint")
    rows = [
        ["Train final loss", f"{train_rows['A-only'][-1]['loss']:.4f}", f"{train_rows['ABC-joint'][-1]['loss']:.4f}", "-"],
        ["splitD Action L1", f"{a['action_l1']:.6f}", f"{b['action_l1']:.6f}", f"{(a['action_l1'] - b['action_l1']) / a['action_l1'] * 100:.2f}% lower"],
        ["splitD Action MSE", f"{a['action_mse']:.6f}", f"{b['action_mse']:.6f}", f"{(a['action_mse'] - b['action_mse']) / a['action_mse'] * 100:.2f}% lower"],
        ["splitD Action L2", f"{a['action_l2']:.6f}", f"{b['action_l2']:.6f}", f"{(a['action_l2'] - b['action_l2']) / a['action_l2'] * 100:.2f}% lower"],
        ["Evaluation coverage", f"{a['num_chunks']} chunks", f"{b['num_chunks']} chunks", "matched"],
    ]
    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "A-only", "ABC-joint", "ABC effect"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_facecolor("#F1F5F9")
            cell.set_text_props(weight="bold", color=COLORS["dark"])
        elif col == 3:
            cell.set_facecolor("#DCFCE7")
            cell.set_text_props(color="#166534", weight="bold")
        else:
            cell.set_facecolor("white")
    ax.set_title("Core Results Table", fontsize=15, fontweight="bold", color=COLORS["dark"], pad=14)
    fig.tight_layout()
    fig.savefig(out_dir / "10_core_results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(train_rows: dict[str, list[dict]], eval_rows: list[dict], chunks: list[dict], out_dir: Path) -> None:
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], hspace=0.32, wspace=0.24)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_chunks = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    for label, rows in train_rows.items():
        if rows:
            ax_loss.plot([r["step"] for r in rows], moving_average([r["loss"] for r in rows], 25), color=COLORS[label], linewidth=2.2, label=label)
    ax_loss.set_title("Training convergence")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax_loss.grid(True, axis="y")
    ax_loss.legend()

    labels = [r["model"] for r in eval_rows]
    values = [r["action_l1"] for r in eval_rows]
    ax_bars.bar(labels, values, color=[COLORS[label] for label in labels], width=0.55)
    ax_bars.set_title("splitD Action L1")
    ax_bars.set_ylim(0, max(values) * 1.25)
    ax_bars.grid(True, axis="y")
    for idx, value in enumerate(values):
        ax_bars.text(idx, value + max(values) * 0.04, f"{value:.4f}", ha="center", fontweight="bold")

    for label in MODELS:
        rows = sorted([r for r in chunks if r["model"] == label], key=lambda r: r["episode_start"])
        ax_chunks.plot([r["episode_start"] for r in rows], [r["action_l1"] for r in rows], marker="o", color=COLORS[label], linewidth=2.0, label=label)
    ax_chunks.set_title("Chunk-wise splitD Action L1")
    ax_chunks.set_xlabel("episode start")
    ax_chunks.set_ylabel("Action L1")
    ax_chunks.grid(True, axis="y")
    ax_chunks.legend()

    a = next(r for r in eval_rows if r["model"] == "A-only")
    b = next(r for r in eval_rows if r["model"] == "ABC-joint")
    l1_gain = (a["action_l1"] - b["action_l1"]) / a["action_l1"] * 100
    mse_gain = (a["action_mse"] - b["action_mse"]) / a["action_mse"] * 100
    l2_gain = (a["action_l2"] - b["action_l2"]) / a["action_l2"] * 100
    ax_text.axis("off")
    summary = (
        "Experiment summary\n\n"
        "Train: ACT, same architecture and hyperparameters\n"
        "Eval: zero-shot offline action error on splitD\n"
        f"Coverage: {a['num_chunks']} chunks, {a['num_eval_batches']} batches, {a['num_frames']} frames\n\n"
        f"Action L1: {a['action_l1']:.4f} -> {b['action_l1']:.4f}  ({l1_gain:.1f}% lower)\n"
        f"Action MSE: {a['action_mse']:.4f} -> {b['action_mse']:.4f}  ({mse_gain:.1f}% lower)\n"
        f"Action L2: {a['action_l2']:.4f} -> {b['action_l2']:.4f}  ({l2_gain:.1f}% lower)\n"
    )
    ax_text.text(0.02, 0.95, summary, va="top", ha="left", fontsize=12, color=COLORS["dark"], linespacing=1.45)

    fig.suptitle("ACT Cross-environment Generalization: A-only vs ABC-joint", fontsize=16, fontweight="bold", color=COLORS["dark"])
    fig.savefig(out_dir / "07_experiment_summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(path: Path, integrity_rows: list[dict], notes: list[str], eval_rows: list[dict], train_rows: dict[str, list[dict]]) -> None:
    a = next((r for r in eval_rows if r["model"] == "A-only"), None)
    b = next((r for r in eval_rows if r["model"] == "ABC-joint"), None)
    lines = ["# Experiment Integrity Report", ""]
    lines.append("## Status Checks")
    lines.append("")
    lines.append("| Item | Status | Detail |")
    lines.append("|---|---:|---|")
    for row in integrity_rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['detail']} |")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## Training Summary")
    lines.append("")
    lines.append("| Model | Parsed rows | Final step | Final loss | Min loss |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, rows in train_rows.items():
        final = rows[-1] if rows else {}
        min_loss = min((r["loss"] for r in rows), default=math.nan)
        lines.append(f"| {label} | {len(rows)} | {final.get('step', '')} | {final.get('loss', math.nan):.4f} | {min_loss:.4f} |")
    lines.append("")
    if a and b:
        lines.append("## Zero-shot splitD Offline Action Error")
        lines.append("")
        lines.append("| Metric | A-only | ABC-joint | Relative reduction |")
        lines.append("|---|---:|---:|---:|")
        for key in ["action_l1", "action_mse", "action_l2"]:
            gain = (a[key] - b[key]) / a[key] * 100
            lines.append(f"| {key} | {a[key]:.6f} | {b[key]:.6f} | {gain:.2f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_visuals"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    train_rows = {label: parse_train_log(args.output_root, label, exp) for label, exp in MODELS.items()}
    all_train_rows = [row for rows in train_rows.values() for row in rows]
    write_csv(args.output_dir / "training_log_scalars_clean.csv", all_train_rows)

    evals = {label: load_eval(args.output_root, label, exp) for label, exp in MODELS.items()}
    eval_rows = [data for data in evals.values() if data is not None]
    eval_summary_rows = [
        {
            "model": row["model"],
            "action_l1": row["action_l1"],
            "action_mse": row["action_mse"],
            "action_l2": row["action_l2"],
            "num_chunks": row["num_chunks"],
            "num_eval_batches": row["num_eval_batches"],
            "num_frames": row["num_frames"],
        }
        for row in eval_rows
    ]
    if len(eval_summary_rows) == 2:
        base, joint = eval_summary_rows
        eval_summary_rows.append(
            {
                "model": "ABC relative reduction",
                "action_l1": (base["action_l1"] - joint["action_l1"]) / base["action_l1"] * 100,
                "action_mse": (base["action_mse"] - joint["action_mse"]) / base["action_mse"] * 100,
                "action_l2": (base["action_l2"] - joint["action_l2"]) / base["action_l2"] * 100,
                "num_chunks": "",
                "num_eval_batches": "",
                "num_frames": "",
            }
        )
    write_csv(args.output_dir / "splitD_action_error_summary.csv", eval_summary_rows)

    chunks = chunk_rows(evals)
    write_csv(args.output_dir / "splitD_chunk_metrics.csv", chunks)
    batch_rows = load_batch_rows(args.output_root)
    write_csv(args.output_dir / "splitD_batch_metrics.csv", batch_rows)

    integrity_rows, notes = build_integrity(args.output_root, train_rows, evals)
    write_csv(args.output_dir / "integrity_checks.csv", integrity_rows)

    plot_training_loss(train_rows, args.output_dir)
    if len(eval_rows) == 2:
        plot_eval_bars(eval_rows, args.output_dir)
        plot_chunk_lines(chunks, args.output_dir, "action_l1", "splitD Chunk-wise Action L1", "04_splitD_chunk_action_l1.png")
        plot_chunk_lines(chunks, args.output_dir, "action_mse", "splitD Chunk-wise Action MSE", "05_splitD_chunk_action_mse.png")
        plot_improvement(chunks, args.output_dir)
        plot_dashboard(train_rows, eval_rows, chunks, args.output_dir)
        plot_chunk_heatmap(chunks, args.output_dir)
        plot_batch_distribution(batch_rows, args.output_dir)
        plot_metric_table(eval_rows, train_rows, args.output_dir)

    write_markdown_report(args.output_dir / "experiment_integrity_report.md", integrity_rows, notes, eval_rows, train_rows)
    print(f"Report visuals and integrity checks written to: {args.output_dir}")


if __name__ == "__main__":
    main()
