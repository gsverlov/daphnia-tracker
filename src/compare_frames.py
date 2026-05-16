from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

from src import config as cfg
from src.visualize_tracks import _to_bgr, annotate_single_frame

_FRAMES = (100, 200, 300)


def compare_frames(
    video_path: Path,
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Produce original + annotated PNG pairs for a fixed set of reference frames.

    For each frame in _FRAMES: saves the raw video frame and the same frame
    with ellipse overlays drawn by annotate_single_frame, so the output is
    directly comparable to the full annotated video.

    Args:
        video_path: Source video.
        tracks_df: Tracks DataFrame (frame, track_id, x, y, …).
        detections_df: Detections DataFrame with ellipse columns.
        output_dir: Directory to write PNGs into.

    Returns:
        List of paths written (originals first, then annotated, per frame).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all originals in a single pass through the video.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    originals: dict[int, Path] = {}
    for frame_idx in _FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"Warning: cannot read frame {frame_idx} — skipping.")
            continue
        out = output_dir / f"compare_frame_{frame_idx}_original.png"
        cv2.imwrite(str(out), _to_bgr(frame))
        originals[frame_idx] = out
    cap.release()

    written: list[Path] = []
    for frame_idx, orig_path in originals.items():
        written.append(orig_path)
        ann_path = output_dir / f"compare_frame_{frame_idx}_annotated.png"
        annotate_single_frame(video_path, tracks_df, detections_df, frame_idx, ann_path)
        written.append(ann_path)

    return written


if __name__ == "__main__":
    video_path = cfg.VIDEO_PATH
    stem = video_path.stem
    tracks_path = cfg.OUTPUT_DIR / f"{stem}_tracks.csv"
    dets_path = cfg.OUTPUT_DIR / f"{stem}_detections.csv"

    if not tracks_path.exists():
        print(f"Tracks file not found: {tracks_path}", file=sys.stderr)
        sys.exit(1)
    if not dets_path.exists():
        print(f"Detections file not found: {dets_path}", file=sys.stderr)
        sys.exit(1)

    tracks_df = pd.read_csv(tracks_path)
    dets_df = pd.read_csv(dets_path)

    written = compare_frames(video_path, tracks_df, dets_df, cfg.OUTPUT_DIR)
    for p in written:
        print(p)
