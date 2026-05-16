from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as cfg

# ---------------------------------------------------------------------------
# Size-based color scheme — three bins split by p25 and p75 of per-tracklet
# major-axis medians.
# BGR: small=red (<p25), medium=green (p25..p75), large=blue (>=p75),
# unknown=gray.
# ---------------------------------------------------------------------------
_COLOR_SMALL: tuple[int, int, int] = (0, 0, 255)     # red
_COLOR_MEDIUM: tuple[int, int, int] = (0, 255, 0)    # green
_COLOR_LARGE: tuple[int, int, int] = (255, 0, 0)     # blue
_COLOR_UNKNOWN: tuple[int, int, int] = (128, 128, 128)

_DEFAULT_SEMI_MAJOR = 4   # fallback ellipse for tracks with no valid fit (8 px full axis)
_DEFAULT_SEMI_MINOR = 2

# Max distance (px) from a track's Kalman-filtered position to the nearest
# detection in the same frame for the row to count as a "real" sighting.
# Tracks remain alive in tracks.csv for up to TRACK_MAX_AGE frames after they
# stop matching detections (see src/tracking.py); during those frames the
# stored (x, y) is purely predicted and tends to drift. We use this threshold
# to skip drawing those ghost rows so they don't appear as off-screen-bound
# ellipses in the annotated video.
_MATCH_DISTANCE_PX: float = 20

# Max pixel distance between consecutive trail points. Segments longer than
# this are skipped (drawn as a visual gap) because they represent missing or
# unreliable data — typically a frame where the track had no real detection
# match, leaving a long jump between surviving points.
MAX_TRAIL_SEGMENT_PX: float = 30.0


def _size_color(
    major_px: float,
    p25: float,
    p75: float,
) -> tuple[int, int, int]:
    if major_px < p25:
        return _COLOR_SMALL
    if major_px >= p75:
        return _COLOR_LARGE
    return _COLOR_MEDIUM


def _load_size_thresholds(
    video_path: Path,
    detections_df: pd.DataFrame,
) -> tuple[float, float] | tuple[None, None]:
    """Return (p25, p75) of mean_major_axis_px — tracklet summary preferred, detections fallback."""
    stem = video_path.stem
    summary_csv = cfg.OUTPUT_DIR / f"{stem}_tracklet_summary.csv"
    if summary_csv.exists():
        ts = pd.read_csv(summary_csv)
        col = "mean_major_axis_px"
        if col in ts.columns:
            vals = ts[col].dropna()
            if len(vals) >= 2:
                return float(vals.quantile(0.25)), float(vals.quantile(0.75))
    if "major_axis_px" in detections_df.columns:
        vals = detections_df["major_axis_px"].dropna()
        if len(vals) >= 2:
            return float(vals.quantile(0.25)), float(vals.quantile(0.75))
    return None, None


def _build_ellipse_lookup(
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
) -> tuple[dict[int, tuple[float, float]], dict[int, dict[int, float]]]:
    """Nearest-centroid join per frame, with a distance threshold to filter
    Kalman-predicted-only track rows.

    Tracks have Kalman-filtered positions; detections have raw contour
    positions. For each track in each frame we find the nearest detection
    in that frame; the pair is accepted only when the distance is
    <= _MATCH_DISTANCE_PX. Predicted-only frames (track stayed alive past
    its last real match) are excluded from both return values, so:

      * track_sizes (median major/minor) is computed only over real sightings
      * frame_orient membership doubles as the "this row was matched to a
        real detection" signal used by the renderers to skip ghost rows.

    Returns:
        track_sizes: track_id → (median_major_px, median_minor_px), nanmedian
            over matched frames only.
        frame_orient: frame_idx → {track_id: orientation_deg} for matched
            (frame, track_id) pairs only. Frames with no matched tracks are
            absent from the dict.
    """
    has_ellipse = (
        "major_axis_px" in detections_df.columns
        and "minor_axis_px" in detections_df.columns
        and "orientation_deg" in detections_df.columns
    )
    if not has_ellipse or detections_df.empty:
        return {}, {}

    track_major: dict[int, list[float]] = {}
    track_minor: dict[int, list[float]] = {}
    frame_orient: dict[int, dict[int, float]] = {}

    det_by_frame: dict[int, pd.DataFrame] = {
        int(f): g.reset_index(drop=True)
        for f, g in detections_df.groupby("frame")
    }

    for frame_idx, track_group in tracks_df.groupby("frame"):
        frame_idx = int(frame_idx)
        det_group = det_by_frame.get(frame_idx)
        if det_group is None or det_group.empty:
            continue

        track_xy = track_group[["x", "y"]].to_numpy()
        det_xy = det_group[["x", "y"]].to_numpy()

        # Vectorised nearest-centroid: (n_tracks, n_dets)
        diffs = track_xy[:, np.newaxis, :] - det_xy[np.newaxis, :, :]
        dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
        nearest = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(track_group)), nearest]

        per_frame: dict[int, float] = {}
        for i, (_, track_row) in enumerate(track_group.iterrows()):
            if nearest_dist[i] > _MATCH_DISTANCE_PX:
                continue
            tid = int(track_row["track_id"])
            nd = det_group.iloc[nearest[i]]
            track_major.setdefault(tid, []).append(float(nd["major_axis_px"]))
            track_minor.setdefault(tid, []).append(float(nd["minor_axis_px"]))
            per_frame[tid] = float(nd["orientation_deg"])

        if per_frame:
            frame_orient[frame_idx] = per_frame

    track_sizes: dict[int, tuple[float, float]] = {
        tid: (float(np.nanmedian(track_major[tid])), float(np.nanmedian(track_minor[tid])))
        for tid in track_major
    }
    return track_sizes, frame_orient


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

    Brightness scales from 30% (oldest) to 100% (most recent).
    """
    n = len(trail_pts)
    for i in range(1, n):
        alpha = i / n
        faded = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
        p1 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
        p2 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
        if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) > MAX_TRAIL_SEGMENT_PX:
            continue
        cv2.line(canvas, p1, p2, faded, 1, cv2.LINE_AA)


def _draw_track(
    canvas: np.ndarray,
    x: float,
    y: float,
    semi_major: int,
    semi_minor: int,
    orientation: float,
    color: tuple[int, int, int],
    trail_pts: list[tuple[float, float]] | None = None,
) -> None:
    """Draw a filled, size-colored ellipse and optional fading trail onto canvas."""
    if trail_pts and len(trail_pts) >= 2:
        _draw_trail(canvas, trail_pts, color)
    cv2.ellipse(
        canvas,
        (int(x), int(y)),
        (semi_major, semi_minor),
        orientation,
        0, 360,
        color,
        -1,
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
    detections_df: pd.DataFrame,
    output_path: Path,
    trail_length: int = 15,
) -> Path:
    """Render an annotated video with size-colored ellipses and motion trails.

    Each track is drawn as a filled ellipse at its Kalman-filtered centroid.
    Color encodes body size: red=small (<p25), green=medium (p25..p75), blue=large (>=p75).
    Per-track size is the nanmedian major/minor axes from a nearest-centroid
    join between tracks_df and detections_df. Per-frame orientation comes from
    the nearest detection in that frame so it reflects current swimming direction.

    Tracks with no valid ellipse data fall back to a small gray 8×4 px ellipse.
    Trails fade from 30% brightness (oldest) to 100% (current position).

    Args:
        video_path: Path to the source video.
        tracks_df: DataFrame with columns frame, track_id, x, y (at minimum).
        detections_df: DataFrame with columns frame, x, y, major_axis_px,
            minor_axis_px, orientation_deg (from run_detection).
        output_path: Desired output path (.mp4). May become .avi on Windows
            if the mp4v codec is unavailable.
        trail_length: Number of past frames to include in the trail.

    Returns:
        Actual output path (may differ from requested if codec fell back).
    """
    print("Building ellipse lookup …")
    track_sizes, frame_orient = _build_ellipse_lookup(tracks_df, detections_df)
    p25, p75 = _load_size_thresholds(video_path, detections_df)
    thresholds_valid = p25 is not None and p75 is not None

    # Pre-compute per-track color and semi-axes (constant across all frames).
    track_color: dict[int, tuple[int, int, int]] = {}
    track_semi: dict[int, tuple[int, int]] = {}
    for tid in tracks_df["track_id"].unique():
        tid = int(tid)
        sizes = track_sizes.get(tid)
        if sizes is not None and np.isfinite(sizes[0]) and np.isfinite(sizes[1]):
            med_major, med_minor = sizes
            color = _size_color(med_major, p25, p75) if thresholds_valid else _COLOR_UNKNOWN
            semi_maj = max(1, int(round(med_major / 2)))
            semi_min = max(1, int(round(med_minor / 2)))
        else:
            color = _COLOR_UNKNOWN
            semi_maj = _DEFAULT_SEMI_MAJOR
            semi_min = _DEFAULT_SEMI_MINOR
        track_color[tid] = color
        track_semi[tid] = (semi_maj, semi_min)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer, output_path = _open_writer(output_path, fps, width, height)

    frame_to_tracks: dict[int, list[dict]] = {
        int(f): g[["track_id", "x", "y"]].to_dict("records")
        for f, g in tracks_df.groupby("frame")
    }
    # Build track position history only over matched (frame, track_id) pairs
    # so trails don't carry Kalman-predicted ghosts. frame_orient's keys
    # encode the matched set (see _build_ellipse_lookup).
    matched_pairs: set[tuple[int, int]] = {
        (f, tid) for f, ids in frame_orient.items() for tid in ids
    }
    track_pos: dict[int, dict[int, tuple[float, float]]] = {}
    for tid, group in tracks_df.groupby("track_id"):
        tid = int(tid)
        track_pos[tid] = {
            int(f): (float(x), float(y))
            for f, x, y in zip(group["frame"], group["x"], group["y"])
            if (int(f), tid) in matched_pairs
        }

    drawn = 0
    for frame_idx in tqdm(range(n_frames), desc="Annotating", unit="frame"):
        ok, frame = cap.read()
        if not ok:
            break

        canvas = _to_bgr(frame)
        orient_at_frame = frame_orient.get(frame_idx, {})

        for det in frame_to_tracks.get(frame_idx, []):
            tid = int(det["track_id"])
            if tid not in orient_at_frame:
                # Predicted-only row — no real detection within _MATCH_DISTANCE_PX.
                continue
            color = track_color.get(tid, _COLOR_UNKNOWN)
            semi_maj, semi_min = track_semi.get(tid, (_DEFAULT_SEMI_MAJOR, _DEFAULT_SEMI_MINOR))
            orient = orient_at_frame[tid]
            if not np.isfinite(orient):
                orient = 0.0

            history = track_pos.get(tid, {})
            trail_start = max(0, frame_idx - trail_length)
            trail_pts = [
                history[f]
                for f in range(trail_start, frame_idx + 1)
                if f in history
            ]
            _draw_track(canvas, det["x"], det["y"], semi_maj, semi_min, orient, color, trail_pts)
            drawn += 1

        writer.write(canvas)

    cap.release()
    writer.release()

    total = len(tracks_df)
    print(
        f"Tracks rows total       : {total:,}\n"
        f"  drawn                 : {drawn:,}\n"
        f"  skipped (predicted)   : {total - drawn:,}"
    )
    return output_path


def annotate_single_frame(
    video_path: Path,
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Save a single annotated frame as a PNG for quick visual inspection.

    Per-track size is computed from the full track history (nanmedian across
    all frames) so size coloring matches the full video. Per-frame orientation
    comes from the nearest detection at frame_idx.

    Args:
        video_path: Path to the source video.
        tracks_df: DataFrame with columns frame, track_id, x, y.
        detections_df: DataFrame with columns frame, x, y, major_axis_px,
            minor_axis_px, orientation_deg (from run_detection).
        frame_idx: 0-indexed frame number to extract and annotate.
        output_path: PNG output path.
    """
    track_sizes, frame_orient = _build_ellipse_lookup(tracks_df, detections_df)
    p25, p75 = _load_size_thresholds(video_path, detections_df)
    thresholds_valid = p25 is not None and p75 is not None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

    canvas = _to_bgr(frame)
    active = tracks_df[tracks_df["frame"] == frame_idx]
    orient_at_frame = frame_orient.get(frame_idx, {})

    drawn = 0
    for _, row in active.iterrows():
        tid = int(row["track_id"])
        if tid not in orient_at_frame:
            # Predicted-only row — no real detection within _MATCH_DISTANCE_PX.
            continue
        sizes = track_sizes.get(tid)
        if sizes is not None and np.isfinite(sizes[0]) and np.isfinite(sizes[1]):
            med_major, med_minor = sizes
            color = _size_color(med_major, p25, p75) if thresholds_valid else _COLOR_UNKNOWN
            semi_maj = max(1, int(round(med_major / 2)))
            semi_min = max(1, int(round(med_minor / 2)))
        else:
            color = _COLOR_UNKNOWN
            semi_maj = _DEFAULT_SEMI_MAJOR
            semi_min = _DEFAULT_SEMI_MINOR

        orient = orient_at_frame[tid]
        if not np.isfinite(orient):
            orient = 0.0

        _draw_track(canvas, float(row["x"]), float(row["y"]), semi_maj, semi_min, orient, color)
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved {drawn} of {len(active)} tracks at frame {frame_idx} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Daphnia tracks overlaid on video."
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

    if args.frame is not None:
        out_path = cfg.OUTPUT_DIR / f"{stem}_frame_{args.frame}.png"
        annotate_single_frame(video_path, tracks_df, dets_df, args.frame, out_path)
    else:
        out_path = cfg.OUTPUT_DIR / f"{stem}_annotated.mp4"
        actual_out = annotate_video(video_path, tracks_df, dets_df, out_path)
        print(f"Annotated video → {actual_out}")
