#!/usr/bin/env python3
"""从多个 CALVIN split 目录创建一个本地 LeRobot 风格数据集。

本工具会优先用软链接引用大体积 parquet 文件，避免重复复制数据；
同时会合并 jsonl 元数据，并重新编号 episode。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def load_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_episode_fields(row: dict, new_index: int, new_path: str, frame_start: int | None = None, length: int | None = None) -> dict:
    row = dict(row)
    for key in ["episode_index", "episode_idx", "index"]:
        if key in row:
            row[key] = new_index
    if frame_start is not None:
        if "from" in row:
            row["from"] = frame_start
        if "start" in row:
            row["start"] = frame_start
    if frame_start is not None and length is not None:
        if "to" in row:
            row["to"] = frame_start + length
        if "end" in row:
            row["end"] = frame_start + length
    if "data_path" in row:
        row["data_path"] = new_path
    if "path" in row and str(row["path"]).endswith(".parquet"):
        row["path"] = new_path
    row["source_split"] = row.get("source_split")
    return row


def _set_or_add_column(table: pa.Table, name: str, values) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    if field_index >= 0:
        column_type = table.schema.field(field_index).type
        array = pa.array(values, type=column_type)
        return table.set_column(field_index, name, array)
    return table.append_column(name, pa.array(values))


def write_reindexed_episode(src: Path, dst: Path, episode_index: int, frame_start: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(src)
    length = table.num_rows
    table = _set_or_add_column(table, "episode_index", [episode_index] * length)
    if table.schema.get_field_index("index") >= 0:
        table = _set_or_add_column(table, "index", list(range(frame_start, frame_start + length)))
    pq.write_table(table, dst)
    return length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/calvin_splitABC")
    parser.add_argument("--episodes-per-chunk", type=int, default=1000)
    args = parser.parse_args()

    output = args.output_root
    if output.exists():
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)

    manifest: list[dict] = []
    merged_episodes: list[dict] = []
    merged_stats: list[dict] = []
    merged_tasks: list[dict] = []
    seen_tasks = set()
    first_meta: Path | None = None
    first_info: dict | None = None
    global_index = 0
    global_frame_index = 0
    print("正在重写 episode parquet 内部的 episode_index/index；这一步会比软链接更慢，但可保证合并后索引连续。")

    for split in args.splits:
        split_root = args.input_root / split
        meta_root = split_root / "meta"
        data_root = split_root / "data"
        if not data_root.exists() or not (meta_root / "info.json").exists():
            raise SystemExit(f"缺少 LeRobot split 目录: {split_root}")
        if first_meta is None:
            first_meta = meta_root
            first_info = load_info(meta_root / "info.json")

        episodes = read_jsonl(meta_root / "episodes.jsonl")
        stats_path = meta_root / "episodes_stats.jsonl"
        stats = read_jsonl(stats_path) if stats_path.exists() else []
        stats_by_old = {}
        for row in stats:
            idx = row.get("episode_index", row.get("episode_idx", row.get("index")))
            stats_by_old[idx] = row

        for task in read_jsonl(meta_root / "tasks.jsonl"):
            key = json.dumps(task, sort_keys=True, ensure_ascii=False)
            if key not in seen_tasks:
                seen_tasks.add(key)
                merged_tasks.append(task)

        parquet_files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
        by_name = {p.name: p for p in parquet_files}

        for row in episodes:
            old_idx = row.get("episode_index", row.get("episode_idx", row.get("index")))
            old_name = f"episode_{int(old_idx):06d}.parquet" if old_idx is not None else None
            src = by_name.get(old_name) if old_name else None
            if src is None and "data_path" in row:
                candidate = split_root / str(row["data_path"])
                if candidate.exists():
                    src = candidate
            if src is None:
                raise SystemExit(f"无法找到 {split} 中该 episode 对应的 parquet 文件: {row}")

            chunk = global_index // args.episodes_per_chunk
            new_rel = f"data/chunk-{chunk:03d}/episode_{global_index:06d}.parquet"
            episode_length = write_reindexed_episode(src, output / new_rel, global_index, global_frame_index)

            new_row = update_episode_fields(row, global_index, new_rel, global_frame_index, episode_length)
            new_row["source_split"] = split
            merged_episodes.append(new_row)

            stat = stats_by_old.get(old_idx)
            if stat is not None:
                new_stat = update_episode_fields(stat, global_index, new_rel, global_frame_index, episode_length)
                new_stat["source_split"] = split
                merged_stats.append(new_stat)

            manifest.append(
                {
                    "new_episode_index": global_index,
                    "source_split": split,
                    "source_file": str(src),
                    "linked_file": new_rel,
                    "num_frames": episode_length,
                }
            )
            global_frame_index += episode_length
            global_index += 1

    if first_meta is None or first_info is None:
        raise SystemExit("没有找到任何输入 split")

    for name in ["conversion.json", "modality.json"]:
        src = first_meta / name
        if src.exists():
            shutil.copy2(src, output / "meta" / name)

    info = dict(first_info)
    info["repo_id"] = args.repo_id
    info["total_episodes"] = len(merged_episodes)
    info["total_frames"] = global_frame_index
    info["splits"] = {"train": f"0:{len(merged_episodes)}"}
    with (output / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    write_jsonl(output / "meta" / "episodes.jsonl", merged_episodes)
    if merged_stats:
        write_jsonl(output / "meta" / "episodes_stats.jsonl", merged_stats)
    write_jsonl(output / "meta" / "tasks.jsonl", merged_tasks)

    with (output / "merge_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_id": args.repo_id,
                "source_splits": args.splits,
                "num_episodes": len(merged_episodes),
                "episodes": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"已创建联合数据集: {output}，episode 数量: {len(merged_episodes)}")


if __name__ == "__main__":
    main()
