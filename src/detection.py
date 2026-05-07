from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as cfg


def build_background_subtractor() -> cv2.BackgroundSubtractorMOG2:
    """Create a MOG2 background subtractor configured from src/config.py.

    Does not perform warmup — that happens inside run_detection so the
    model is built while reading the video sequentially.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=cfg.MOG2_HISTORY,
        varThreshold=cfg.MOG2_VAR_THRESHOLD,
        detectShadows=cfg.MOG2_DETECT_SHADOWS,
    )


def build_roi_mask(height: int, width: int) -> np.ndarray:
    """Return a uint8 mask of shape (height, width) encoding the active detection region.

    255 inside the ROI, 0 in the excluded margins. Margin sizes are read
    from cfg.ROI_TOP/BOTTOM/LEFT/RIGHT. A margin of 0 means no exclusion
    on that edge.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    y0 = cfg.ROI_TOP
    y1 = height - cfg.ROI_BOTTOM if cfg.ROI_BOTTOM > 0 else height
    x0 = cfg.ROI_LEFT
    x1 = width - cfg.ROI_RIGHT if cfg.ROI_RIGHT > 0 else width
    mask[y0:y1, x0:x1] = 255
    return mask


def detect_frame(
    frame_gray: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractorMOG2,
    roi_mask: np.ndarray,
) -> list[dict]:
    """Run one grayscale frame through the full detection pipeline.

    Pipeline: MOG2 foreground mask → morphological opening → ROI mask →
    contour filtering by area → centroid and bounding-box extraction.

    The background model is NOT updated by this call (learningRate=0).
    All area and morphology parameters come from src/config.py.

    Args:
        frame_gray: Grayscale uint8 array, shape (H, W). Caller is
            responsible for converting colour frames before passing here.
        bg_subtractor: Warmed-up MOG2 subtractor.
        roi_mask: uint8 mask from build_roi_mask(); 255 = active region.

    Returns:
        List of dicts, one per detected fish, with keys:
            x (float): moment-centroid column, pixels from left.
            y (float): moment-centroid row, pixels from top.
            w (int):   bounding-box width in pixels.
            h (int):   bounding-box height in pixels.
            area (float): contour area in pixels².
        Empty list when no detections pass the area filter.
    """
    raw_mask = bg_subtractor.apply(frame_gray, learningRate=0)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE)
    )
    clean_mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_OPEN, kernel, iterations=cfg.MORPH_OPEN_ITERATIONS
    )
    clean_mask = cv2.bitwise_and(clean_mask, roi_mask)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (cfg.MIN_AREA <= area <= cfg.MAX_AREA):
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            # Degenerate contour with no area — skip rather than divide by zero.
            continue
        x = M["m10"] / M["m00"]
        y = M["m01"] / M["m00"]
        _, _, bw, bh = cv2.boundingRect(contour)
        detections.append({"x": x, "y": y, "w": bw, "h": bh, "area": area})

    return detections


def run_detection(video_path: Path, progress: bool = True) -> pd.DataFrame:
    """Run the full detection pipeline on a video file.

    Reads the video sequentially. The first cfg.MOG2_WARMUP_FRAMES frames
    are fed to the background subtractor at the default (auto-adaptive)
    learning rate to build the background model; no detections are collected
    during this period. From frame cfg.MOG2_WARMUP_FRAMES onward the model
    is frozen (learningRate=0) and detections are accumulated.

    Frames that yield no detections after filtering produce no rows in the
    output — the frame index simply does not appear. This is intentional;
    a missing frame is not an error.

    If a frame cannot be read (e.g. corrupted data or a short video), the
    loop breaks and any detections collected up to that point are returned.

    Args:
        video_path: Path to the input video file.
        progress: Show a tqdm progress bar while processing.

    Returns:
        DataFrame with columns:
            frame (int64): 0-indexed original video frame number.
            x (float64):   moment-centroid column in pixels.
            y (float64):   moment-centroid row in pixels.
            w (int64):     bounding-box width in pixels.
            h (int64):     bounding-box height in pixels.
            area (float64): contour area in pixels².
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_subtractor = build_background_subtractor()
    roi_mask = build_roi_mask(height, width)

    rows: list[dict] = []
    frame_iter: range | tqdm = range(n_frames)
    if progress:
        frame_iter = tqdm(frame_iter, desc="Detecting", unit="frame")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        if frame_idx < cfg.MOG2_WARMUP_FRAMES:
            bg_subtractor.apply(gray)
            continue

        for det in detect_frame(gray, bg_subtractor, roi_mask):
            rows.append({"frame": frame_idx, **det})

    cap.release()

    return pd.DataFrame(
        rows,
        columns=["frame", "x", "y", "w", "h", "area"],
    )


if __name__ == "__main__":
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.VIDEO_PATH

    t0 = time.perf_counter()
    df = run_detection(video_path)
    elapsed = time.perf_counter() - t0

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.OUTPUT_DIR / f"{video_path.stem}_detections.csv"
    df.to_csv(out_path, index=False)

    # Post-warmup frame count for zero-detection summary
    _cap = cv2.VideoCapture(str(video_path))
    _n_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    post_warmup = max(0, _n_frames - cfg.MOG2_WARMUP_FRAMES)
    n_det_frames = df["frame"].nunique() if not df.empty else 0
    zero_det_frames = post_warmup - n_det_frames
    mean_per_frame = len(df) / n_det_frames if n_det_frames > 0 else 0.0

    print(f"Video                 : {video_path}")
    print(f"Output                : {out_path}")
    print(f"Total detections      : {len(df):,}")
    print(f"Mean / frame          : {mean_per_frame:.1f}")
    print(f"Zero-detection frames : {zero_det_frames} / {post_warmup}")
    print(f"Runtime               : {elapsed:.1f}s")
