from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.cm as mcm
import numpy as np
import pandas as pd

from src import config as cfg

_CMAP = mcm.coolwarm   # low intensity → blue, high intensity → red
_CIRCLE_RADIUS = 4


def _intensity_to_bgr(
    intensity: float,
    vmin: float,
    vmax: float,
) -> tuple[int, int, int]:
    """Map a scalar intensity to a BGR color via the coolwarm colormap."""
    denom = vmax - vmin
    t = float(np.clip((intensity - vmin) / denom if denom > 0 else 0.5, 0.0, 1.0))
    r, g, b, _ = _CMAP(t)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw_colorbar(
    canvas: np.ndarray,
    vmin: float,
    vmax: float,
    x0: int,
    y0: int,
    bar_width: int = 16,
    bar_height: int = 130,
) -> None:
    """Draw a vertical coolwarm colorbar with tick labels onto canvas in-place.

    top of bar = vmax (red), bottom = vmin (blue), matching the standard
    convention that higher values appear warmer.
    """
    # White backing so labels are legible over any frame content.
    pad = 4
    cv2.rectangle(
        canvas,
        (x0 - pad, y0 - 16),
        (x0 + bar_width + 58, y0 + bar_height + pad),
        (255, 255, 255),
        -1,
    )

    for i in range(bar_height):
        t = 1.0 - i / max(bar_height - 1, 1)   # top → 1.0 (hot/red), bottom → 0.0 (cool/blue)
        r, g, b, _ = _CMAP(t)
        bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.line(canvas, (x0, y0 + i), (x0 + bar_width, y0 + i), bgr, 1)

    cv2.rectangle(canvas, (x0, y0), (x0 + bar_width, y0 + bar_height), (80, 80, 80), 1)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ty = y0 + int((1.0 - frac) * (bar_height - 1))
        val = vmin + frac * (vmax - vmin)
        cv2.line(canvas, (x0 + bar_width, ty), (x0 + bar_width + 4, ty), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            f"{val:.0f}",
            (x0 + bar_width + 7, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "intensity",
        (x0, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )


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


def visualize_intensity_spatial(
    video_path: Path,
    detections_csv: Path,
    frame_idx: int,
    output_original: Path,
    output_annotated: Path,
) -> None:
    """Render detections as filled circles color-graded by mean_intensity.

    Color scale: coolwarm — blue for low intensity (dark Daphnia bodies),
    red for high intensity (bright artefacts, shadow edges, debris).
    vmin/vmax are set from the frame's own detections so the full colormap
    range is always used, making spatial clustering of bright or dark
    detections easy to spot.

    The colorbar is embedded in the bottom-right corner on a white backing
    so it is legible regardless of the frame background.

    Args:
        video_path: Source video.
        detections_csv: Per-frame detections CSV from run_detection.
        frame_idx: 0-indexed frame number to annotate.
        output_original: Destination PNG for the raw, unannotated frame.
        output_annotated: Destination PNG for the annotated frame.
    """
    dets_all = pd.read_csv(detections_csv)
    dets = dets_all[dets_all["frame"] == frame_idx].copy()

    has_intensity = dets["mean_intensity"].notna()
    valid_dets = dets[has_intensity]
    n_skipped = int((~has_intensity).sum())

    canvas = _load_frame_bgr(video_path, frame_idx)

    output_original.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_original), canvas)

    canvas = canvas.copy()

    if not valid_dets.empty:
        vmin = float(valid_dets["mean_intensity"].min())
        vmax = float(valid_dets["mean_intensity"].max())

        for _, row in valid_dets.iterrows():
            cx = int(round(float(row["x"])))
            cy = int(round(float(row["y"])))
            color = _intensity_to_bgr(float(row["mean_intensity"]), vmin, vmax)
            cv2.circle(canvas, (cx, cy), _CIRCLE_RADIUS, color, -1, cv2.LINE_AA)

        h, w = canvas.shape[:2]
        bar_w, bar_h = 16, 130
        right_pad = 70   # space for tick labels
        bottom_pad = 10
        cb_x = w - bar_w - right_pad - bottom_pad
        cb_y = h - bar_h - bottom_pad
        _draw_colorbar(canvas, vmin, vmax, cb_x, cb_y, bar_width=bar_w, bar_height=bar_h)
    else:
        vmin = vmax = float("nan")

    output_annotated.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_annotated), canvas)

    print(f"Frame            : {frame_idx}")
    print(f"Detections drawn : {len(valid_dets)}")
    print(f"Skipped (NaN)    : {n_skipped}")
    if not valid_dets.empty:
        print(f"Intensity range  : {vmin:.1f} – {vmax:.1f}")
    print(f"Original         : {output_original}")
    print(f"Annotated        : {output_annotated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial intensity overlay for a single video frame."
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=200,
        metavar="N",
        help="0-indexed frame to annotate (default: 200).",
    )
    parser.add_argument(
        "--detections-csv",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to detections CSV (default: output/{stem}_detections.csv).",
    )
    args = parser.parse_args()

    video_path = cfg.VIDEO_PATH
    stem = video_path.stem
    det_csv = (
        Path(args.detections_csv) if args.detections_csv
        else cfg.OUTPUT_DIR / f"{stem}_detections.csv"
    )
    out_original = cfg.OUTPUT_DIR / f"intensity_check_frame_{args.frame}_original.png"
    out_annotated = cfg.OUTPUT_DIR / f"intensity_check_frame_{args.frame}_annotated.png"

    if not det_csv.exists():
        print(f"Detections file not found: {det_csv}", file=sys.stderr)
        sys.exit(1)

    visualize_intensity_spatial(video_path, det_csv, args.frame, out_original, out_annotated)
