#!/usr/bin/env python3
"""从 rollout 视频中抽取代表性帧。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def sample_video(path: Path, output_dir: Path, frames_per_video: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return
    indices = sorted(set([0, total // 2, max(0, total - 1)]))[:frames_per_video]
    stem_dir = output_dir / path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(stem_dir / f"frame_{idx:06d}.png"), frame)
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.video_dir.exists():
        return
    videos = []
    for ext in ("*.mp4", "*.webm", "*.avi", "*.mov"):
        videos.extend(args.video_dir.rglob(ext))
    for video in sorted(videos)[: args.max_videos]:
        sample_video(video, args.output_dir, args.frames_per_video)
    print(f"已从 {min(len(videos), args.max_videos)} 个视频中抽取帧")


if __name__ == "__main__":
    main()
