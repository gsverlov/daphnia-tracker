from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config as cfg

# BGR colors: small=red, medium=green, large=blue
_COLOR_SMALL: tuple[int, int, int] = (0, 0, 255)
_COLOR_MEDIUM: tuple[int, int, int] = (0, 255, 0)
_COLOR_LARGE: tuple[int, int, int] = (255, 0, 0)


def _load_frame_bgr(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def _load_thresholds(tracklet_summary_csv: Path | None) -> tuple[float, float] | tuple[None, None]:
    """Return (p25, p75) of mean_major_axis_px from the tracklet summary, or (None, None)."""
    if tracklet_summary_csv is None or not tracklet_summary_csv.exists():
        return None, None
    ts = pd.read_csv(tracklet_summary_csv)
    col = "mean_major_axis_px"
    if col not in ts.columns:
        return None, None
    vals = ts[col].dropna()
    if len(vals) < 2:
        return None, None
    return float(vals.quantile(0.25)), float(vals.quantile(0.75))


def _size_color(
    major_px: float,
    p25: float,
    p75: float,
) -> tuple[int, int, int]:
    if major_px < p25:
        return _COLOR_SMALL
    if major_px > p75:
        return _COLOR_LARGE
    return _COLOR_MEDIUM


def visualize_size(
    video_path: Path,
    detections_csv: Path,
    tracklet_summary_csv: Path | None,
    frame_idx: int,
    output_original: Path,
    output_annotated: Path,
) -> None:
    """Annotate one frame with fitted ellipse outlines, colored by body size.

    Ellipses are drawn as outlines only (thickness=1) so the underlying
    Daphnia body remains visible. Each ellipse is labeled with its
    major_axis_px to one decimal place. Color bins use the p25/p75 of
    mean_major_axis_px from the tracklet summary; falls back to population
    quantiles from all detections if the summary is unavailable.

    Detections with NaN major_axis_px (contours with fewer than 5 points,
    which cv2.fitEllipse cannot fit) are skipped and counted.

    Args:
        video_path: Source video file.
        detections_csv: Per-frame detections CSV from run_detection.
        tracklet_summary_csv: Tracklet summary CSV for size thresholds;
            None or missing falls back to detections-wide quantiles.
        frame_idx: 0-indexed frame number to annotate.
        output_original: Destination PNG for the raw, unannotated frame.
        output_annotated: Destination PNG for the frame with ellipse overlays.
    """
    dets_all = pd.read_csv(detections_csv)
    dets = dets_all[dets_all["frame"] == frame_idx].copy()

    has_ellipse = dets["major_axis_px"].notna()
    valid_dets = dets[has_ellipse]
    n_skipped = int((~has_ellipse).sum())

    # Prefer tracklet-summary thresholds; fall back to full-detections quantiles.
    p25, p75 = _load_thresholds(tracklet_summary_csv)
    if p25 is None:
        fallback = dets_all["major_axis_px"].dropna()
        if len(fallback) >= 2:
            p25 = float(fallback.quantile(0.25))
            p75 = float(fallback.quantile(0.75))

    canvas = _load_frame_bgr(video_path, frame_idx)

    output_original.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_original), canvas)

    thresholds_valid = p25 is not None and p75 is not None

    for _, row in valid_dets.iterrows():
        cx = int(round(float(row["x"])))
        cy = int(round(float(row["y"])))
        major = float(row["major_axis_px"])
        minor = float(row["minor_axis_px"])
        angle = float(row["orientation_deg"])

        # cv2.fitEllipse returns full axis lengths; cv2.ellipse wants semi-axes.
        semi_major = max(1, int(round(major / 2)))
        semi_minor = max(1, int(round(minor / 2)))

        color = _size_color(major, p25, p75) if thresholds_valid else _COLOR_MEDIUM

        cv2.ellipse(
            canvas,
            (cx, cy),
            (semi_major, semi_minor),
            angle,
            0, 360,
            color,
            1,
            cv2.LINE_AA,
        )

        label = f"{major:.1f}"
        cv2.putText(
            canvas,
            label,
            (cx + semi_major + 2, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    output_annotated.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_annotated), canvas)

    print(f"Frame            : {frame_idx}")
    print(f"Detections drawn : {len(valid_dets)}")
    print(f"Skipped (NaN)    : {n_skipped}")
    if thresholds_valid:
        src_label = "tracklet summary" if (tracklet_summary_csv and tracklet_summary_csv.exists()) else "all detections"
        print(f"Thresholds from  : {src_label}")
        print(f"  small  (red)   : major_axis < {p25:.1f} px")
        print(f"  medium (green) : {p25:.1f} – {p75:.1f} px")
        print(f"  large  (blue)  : major_axis > {p75:.1f} px")
    else:
        print("Size thresholds  : insufficient data — all ellipses green")
    print(f"Original         : {output_original}")
    print(f"Annotated        : {output_annotated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ellipse body-size fits on a single video frame."
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=200,
        metavar="N",
        help="0-indexed frame number to annotate (default: 200).",
    )
    parser.add_argument(
        "--detections-csv",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to detections CSV (default: output/{stem}_detections.csv).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to source video (default: cfg.VIDEO_PATH).",
    )
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else cfg.VIDEO_PATH
    stem = video_path.stem
    det_csv = (
        Path(args.detections_csv) if args.detections_csv
        else cfg.OUTPUT_DIR / f"{stem}_detections.csv"
    )
    summary_csv = cfg.OUTPUT_DIR / f"{stem}_tracklet_summary.csv"
    out_original = cfg.OUTPUT_DIR / f"size_check_frame_{args.frame}_original.png"
    out_annotated = cfg.OUTPUT_DIR / f"size_check_frame_{args.frame}_annotated.png"

    if not det_csv.exists():
        print(f"Detections file not found: {det_csv}", file=sys.stderr)
        sys.exit(1)

    visualize_size(video_path, det_csv, summary_csv, args.frame, out_original, out_annotated)
