from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as cfg

# ---------------------------------------------------------------------------
# Color table: tab20 pre-converted to BGR uint8 for OpenCV.
# track_id % _N_COLORS indexes this list so each ID gets a stable color.
# ---------------------------------------------------------------------------
_cmap = plt.get_cmap("tab20")
_COLORS_BGR: list[tuple[int, int, int]] = [
    (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
    for rgba in _cmap.colors
]
_N_COLORS = len(_COLORS_BGR)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a given track_id."""
    return _COLORS_BGR[int(track_id) % _N_COLORS]


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Return a writable BGR copy of a frame regardless of its original format."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def _draw_trail(
    canvas: np.ndarray,
    trail_pts: list[tuple[float, float]],
    color: tuple[int, int, int],
) -> None:
    """Draw a fading trail from oldest to newest position.

    Each segment is drawn at brightness scaled from 30% (oldest) to 100%
    (most recent) so the fish's recent path reads more clearly than its
    distant history.
    """
    n = len(trail_pts)
    for i in range(1, n):
        alpha = i / n
        faded = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
        p1 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
        p2 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
        cv2.line(canvas, p1, p2, faded, 1, cv2.LINE_AA)


def _draw_track(
    canvas: np.ndarray,
    track_id: int,
    x: float,
    y: float,
    trail_pts: list[tuple[float, float]] | None = None,
) -> None:
    """Draw a single track's circle, label, and optional trail onto canvas."""
    color = _track_color(track_id)
    if trail_pts and len(trail_pts) >= 2:
        _draw_trail(canvas, trail_pts, color)
    cv2.circle(canvas, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        str(track_id),
        (int(x) + 7, int(y) + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _open_writer(
    output_path: Path, fps: float, width: int, height: int
) -> tuple[cv2.VideoWriter, Path]:
    """Open a VideoWriter with mp4v codec; fall back to XVID + .avi on failure."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, output_path

    print(
        f"Warning: mp4v codec failed for {output_path}. "
        "Falling back to XVID + .avi"
    )
    output_path = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return writer, output_path


def annotate_video(
    video_path: Path,
    tracks_df: pd.DataFrame,
    output_path: Path,
    trail_length: int = 15,
) -> Path:
    """Render an annotated video with track circles, labels, and motion trails.

    For each frame, draws every active track from tracks_df as a filled
    circle colored by track_id, with the track ID printed alongside it,
    and a fading trail of the last trail_length positions.

    Args:
        video_path: Path to the source video.
        tracks_df: DataFrame with columns frame, track_id, x, y (at minimum).
        output_path: Desired output path (.mp4). May become .avi on Windows
            if the mp4v codec is unavailable.
        trail_length: Number of past frames to include in the trail.

    Returns:
        Actual output path (may differ from requested if codec fell back).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer, output_path = _open_writer(output_path, fps, width, height)

    # Pre-compute lookup structures for the inner loop.
    # frame_to_tracks: frame_idx -> list of {track_id, x, y}
    frame_to_tracks: dict[int, list[dict]] = {
        int(f): g[["track_id", "x", "y"]].to_dict("records")
        for f, g in tracks_df.groupby("frame")
    }
    # track_pos: track_id -> {frame_idx: (x, y)} for trail lookups
    track_pos: dict[int, dict[int, tuple[float, float]]] = {}
    for tid, group in tracks_df.groupby("track_id"):
        track_pos[int(tid)] = dict(
            zip(group["frame"].astype(int), zip(group["x"], group["y"]))
        )

    for frame_idx in tqdm(range(n_frames), desc="Annotating", unit="frame"):
        ok, frame = cap.read()
        if not ok:
            break

        canvas = _to_bgr(frame)

        for det in frame_to_tracks.get(frame_idx, []):
            track_id = int(det["track_id"])
            history = track_pos.get(track_id, {})
            trail_start = max(0, frame_idx - trail_length)
            trail_pts = [
                history[f]
                for f in range(trail_start, frame_idx + 1)
                if f in history
            ]
            _draw_track(canvas, track_id, det["x"], det["y"], trail_pts)

        writer.write(canvas)

    cap.release()
    writer.release()
    return output_path


def annotate_single_frame(
    video_path: Path,
    tracks_df: pd.DataFrame,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Save a single annotated frame as a PNG for quick visual inspection.

    Draws circles and labels for every track active at frame_idx. No trail
    is drawn (a single frame doesn't have history to show).

    Args:
        video_path: Path to the source video.
        tracks_df: DataFrame with columns frame, track_id, x, y.
        frame_idx: 0-indexed frame number to extract and annotate.
        output_path: PNG output path.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

    canvas = _to_bgr(frame)

    active = tracks_df[tracks_df["frame"] == frame_idx]
    for _, row in active.iterrows():
        _draw_track(canvas, int(row["track_id"]), float(row["x"]), float(row["y"]))

    cv2.imwrite(str(output_path), canvas)
    print(f"Saved {len(active)} tracks at frame {frame_idx} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize fish tracks overlaid on video."
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        metavar="N",
        help="Annotate a single frame N and save as PNG. "
             "Omit to annotate the whole video.",
    )
    args = parser.parse_args()

    video_path = cfg.VIDEO_PATH
    tracks_path = cfg.OUTPUT_DIR / f"{video_path.stem}_tracks.csv"

    if not tracks_path.exists():
        print(f"Tracks file not found: {tracks_path}", file=sys.stderr)
        sys.exit(1)

    tracks_df = pd.read_csv(tracks_path)

    if args.frame is not None:
        out_path = cfg.OUTPUT_DIR / f"{video_path.stem}_frame_{args.frame}.png"
        annotate_single_frame(video_path, tracks_df, args.frame, out_path)
    else:
        out_path = cfg.OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"
        actual_out = annotate_video(video_path, tracks_df, out_path)
        print(f"Annotated video → {actual_out}")
