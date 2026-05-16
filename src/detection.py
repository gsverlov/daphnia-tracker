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


def _split_by_watershed(
    contour: np.ndarray,
    frame_gray: np.ndarray,
) -> list[np.ndarray]:
    """Split a large contour into sub-blob contours via distance transform + watershed.

    Returns the per-sub-region external contours (largest one per region).
    Returns an empty list when the contour has only one peak — the caller
    should then fall back to using the original contour as a single detection.
    """
    h, w = frame_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, thickness=-1)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak_max = float(dist.max())
    if peak_max <= 0:
        return []

    # Seeds = high-distance peaks; 0.5 × max isolates the cores of touching
    # blobs while leaving the saddle regions between them as "unknown".
    _, sure_fg = cv2.threshold(dist, 0.5 * peak_max, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(sure_fg)
    # n_labels includes background (label 0). Need >=2 fg components to split.
    if n_labels < 3:
        return []

    # Markers: bg=1, fg seeds=2..N, unknown=0 (per cv2.watershed convention).
    markers = labels.astype(np.int32) + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0

    img3 = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)

    sub_contours: list[np.ndarray] = []
    for label_id in range(2, n_labels + 1):
        sub_mask = (markers == label_id).astype(np.uint8) * 255
        cs, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            sub_contours.append(max(cs, key=cv2.contourArea))
    return sub_contours


def detect_frame(
    frame_gray: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractorMOG2,
    roi_mask: np.ndarray,
) -> tuple[list[dict], int, int, int, int, int, int, int]:
    """Run one grayscale frame through the full detection pipeline.

    Pipeline: MOG2 foreground mask → morphological opening → ROI mask →
    contour filtering by area → centroid and bounding-box extraction →
    optional shadow-contrast filter in the left zone.

    The background model is NOT updated by this call (learningRate=0).
    All area, morphology, and shadow-filter parameters come from src/config.py.

    Args:
        frame_gray: Grayscale uint8 array, shape (H, W). Caller is
            responsible for converting colour frames before passing here.
        bg_subtractor: Warmed-up MOG2 subtractor.
        roi_mask: uint8 mask from build_roi_mask(); 255 = active region.

    Returns:
        Tuple of (detections, n_shadow_rejected_left, n_shadow_rejected_right):
            detections: list of dicts, one per kept detection, with keys:
                x (float): moment-centroid column, pixels from left.
                y (float): moment-centroid row, pixels from top.
                w (int):   bounding-box width in pixels.
                h (int):   bounding-box height in pixels.
                area (float): contour area in pixels².
                major_axis_px (float): major ellipse axis in pixels; NaN when the
                    contour has fewer than 5 points (cv2.fitEllipse minimum).
                minor_axis_px (float): minor ellipse axis in pixels; NaN as above.
                orientation_deg (float): ellipse tilt in degrees [0, 180) per
                    OpenCV fitEllipse convention; NaN as above.
                mean_intensity (float): mean grayscale pixel value inside the
                    contour boundary, sampled from frame_gray.
                contrast (float): contour mean minus mean intensity of a thin
                    ring offset outward from the contour edge. Negative values
                    indicate the contour is darker than its surroundings (real
                    Daphnia); near-zero or positive values indicate shadows.
                    Computed for every kept detection regardless of zone.
            n_shadow_rejected_left: count of left-zone detections rejected by
                the shadow-contrast filter (0 when SHADOW_FILTER_ENABLED is False).
            n_shadow_rejected_right: same, for the right zone.
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

    # Ring kernels for shadow-contrast filter. Inner-edge dilation creates the
    # gap between the contour and the ring; outer-edge dilation defines the
    # ring's outer boundary. Subtracting the two yields the ring band.
    inner_radius = cfg.SHADOW_FILTER_RING_GAP_PX
    outer_radius = cfg.SHADOW_FILTER_RING_GAP_PX + cfg.SHADOW_FILTER_RING_THICKNESS_PX
    inner_diameter = 2 * inner_radius + 1
    outer_diameter = 2 * outer_radius + 1
    kernel_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inner_diameter, inner_diameter)
    )
    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (outer_diameter, outer_diameter)
    )

    detections: list[dict] = []
    n_shadow_rejected_left = 0
    n_shadow_rejected_right = 0
    n_ellipse_rejected = 0
    n_singletons = 0
    n_split = 0
    n_split_subs = 0
    n_split_fallback = 0
    frame_width = frame_gray.shape[1]
    right_zone_start = frame_width - cfg.SHADOW_FILTER_ZONE_WIDTH_RIGHT
    contour_mask = np.zeros(frame_gray.shape, dtype=np.uint8)

    def _process(sub_contour: np.ndarray) -> str:
        """Run the existing per-contour pipeline on one (sub-)contour.

        Returns one of "ok", "shadow_left", "shadow_right", "ellipse", "skip"
        and appends to outer-scope `detections` / counters as appropriate.
        Logic mirrors the original detect_frame loop body unchanged — only
        wrapped so it can be applied to watershed sub-regions too.
        """
        nonlocal n_shadow_rejected_left, n_shadow_rejected_right, n_ellipse_rejected
        sub_area = cv2.contourArea(sub_contour)
        M = cv2.moments(sub_contour)
        if M["m00"] == 0:
            return "skip"
        x = M["m10"] / M["m00"]
        y = M["m01"] / M["m00"]
        _, _, bw, bh = cv2.boundingRect(sub_contour)
        if len(sub_contour) >= 5:
            (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(sub_contour)
            major_axis = float(max(axis_a, axis_b))
            minor_axis = float(min(axis_a, axis_b))
            orientation = float(angle)
            if not (np.isfinite(major_axis) and np.isfinite(minor_axis)):
                major_axis = float("nan")
                minor_axis = float("nan")
                orientation = float("nan")
        else:
            major_axis = float("nan")
            minor_axis = float("nan")
            orientation = float("nan")

        if np.isfinite(major_axis) and np.isfinite(minor_axis):
            aspect_ratio = major_axis / max(minor_axis, 0.001)
            if major_axis > cfg.MAX_MAJOR_AXIS_PX or aspect_ratio > cfg.MAX_ASPECT_RATIO:
                n_ellipse_rejected += 1
                return "ellipse"

        contour_mask[:] = 0
        cv2.drawContours(contour_mask, [sub_contour], 0, 255, thickness=-1)
        mean_intensity = float(cv2.mean(frame_gray, mask=contour_mask)[0])

        inner_mask = cv2.dilate(contour_mask, kernel_inner, iterations=1)
        outer_mask = cv2.dilate(contour_mask, kernel_outer, iterations=1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        ring_intensity = float(cv2.mean(frame_gray, mask=ring_mask)[0])
        contrast = mean_intensity - ring_intensity

        if cfg.SHADOW_FILTER_ENABLED and contrast > cfg.SHADOW_FILTER_MARGIN:
            if x < cfg.SHADOW_FILTER_ZONE_WIDTH:
                n_shadow_rejected_left += 1
                return "shadow_left"
            if x > right_zone_start:
                n_shadow_rejected_right += 1
                return "shadow_right"

        detections.append({
            "x": x, "y": y, "w": bw, "h": bh, "area": sub_area,
            "major_axis_px": major_axis,
            "minor_axis_px": minor_axis,
            "orientation_deg": orientation,
            "mean_intensity": mean_intensity,
            "contrast": contrast,
        })
        return "ok"

    for contour in contours:
        area = cv2.contourArea(contour)
        if not (cfg.MIN_AREA <= area <= cfg.MAX_AREA):
            continue

        if area < cfg.WATERSHED_SPLIT_THRESHOLD_PX2:
            n_singletons += 1
            _process(contour)
            continue

        # Large contour — attempt watershed split.
        sub_contours = _split_by_watershed(contour, frame_gray)
        valid_subs = [s for s in sub_contours if cv2.contourArea(s) >= cfg.MIN_AREA]
        if not valid_subs:
            n_split_fallback += 1
            _process(contour)
            continue

        n_split += 1
        for sub in valid_subs:
            if _process(sub) == "ok":
                n_split_subs += 1

    return (
        detections,
        n_shadow_rejected_left,
        n_shadow_rejected_right,
        n_ellipse_rejected,
        n_singletons,
        n_split,
        n_split_subs,
        n_split_fallback,
    )


def run_detection(
    video_path: Path, progress: bool = True
) -> tuple[pd.DataFrame, int, int, int, int, int, int, int]:
    """Run the full detection pipeline on a video file.

    The first cfg.MOG2_WARMUP_FRAMES frames are fed to the background
    subtractor at the default (auto-adaptive) learning rate to build the
    background model; no detections are collected during this pass. The
    video is then rewound to frame 0 and every frame — including the warmup
    region — is processed against the now-frozen model (learningRate=0), so
    detections are emitted for all frames 0..n_frames-1.

    Frames that yield no detections after filtering produce no rows in the
    output — the frame index simply does not appear. This is intentional;
    a missing frame is not an error.

    If a frame cannot be read (e.g. corrupted data or a short video), the
    loop breaks and any detections collected up to that point are returned.

    Args:
        video_path: Path to the input video file.
        progress: Show a tqdm progress bar while processing.

    Returns:
        Tuple of (df, n_shadow_rejected_left, n_shadow_rejected_right,
        n_ellipse_rejected, n_singletons, n_split, n_split_subs, n_split_fallback):
            df: DataFrame with columns:
                frame (int64):         0-indexed original video frame number.
                x (float64):           moment-centroid column in pixels.
                y (float64):           moment-centroid row in pixels.
                w (int64):             bounding-box width in pixels.
                h (int64):             bounding-box height in pixels.
                area (float64):        contour area in pixels².
                major_axis_px (float64): major ellipse axis; NaN when <5 contour pts.
                minor_axis_px (float64): minor ellipse axis; NaN as above.
                orientation_deg (float64): ellipse tilt [0, 180°); NaN as above.
                mean_intensity (float64): mean grayscale value inside the contour.
                contrast (float64):    contour mean minus surrounding-ring mean
                    intensity. Negative = darker than surroundings (real Daphnia);
                    near-zero or positive = shadow-like.
            n_shadow_rejected_left: total left-zone detections rejected by the
                shadow-contrast filter across all frames (0 when disabled).
            n_shadow_rejected_right: same, for the right zone.
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
    n_shadow_rejected_left = 0
    n_shadow_rejected_right = 0
    n_ellipse_rejected = 0
    n_singletons = 0
    n_split = 0
    n_split_subs = 0
    n_split_fallback = 0

    # Phase 1: warmup — feed the first frames at the default (auto-adaptive)
    # learning rate to build the background model. No detections collected.
    for _ in range(min(cfg.MOG2_WARMUP_FRAMES, n_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        bg_subtractor.apply(gray)

    # Phase 2: rewind to frame 0 and process every frame against the now-frozen
    # background model (detect_frame uses learningRate=0), so the warmup region
    # is evaluated, not retrained on.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_iter: range | tqdm = range(n_frames)
    if progress:
        frame_iter = tqdm(frame_iter, desc="Detecting", unit="frame")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        (
            dets,
            n_rej_left,
            n_rej_right,
            n_rej_ellipse,
            n_s,
            n_sp,
            n_sps,
            n_spf,
        ) = detect_frame(gray, bg_subtractor, roi_mask)
        n_shadow_rejected_left += n_rej_left
        n_shadow_rejected_right += n_rej_right
        n_ellipse_rejected += n_rej_ellipse
        n_singletons += n_s
        n_split += n_sp
        n_split_subs += n_sps
        n_split_fallback += n_spf
        for det in dets:
            rows.append({"frame": frame_idx, **det})

    cap.release()

    df = pd.DataFrame(
        rows,
        columns=["frame", "x", "y", "w", "h", "area",
                 "major_axis_px", "minor_axis_px", "orientation_deg",
                 "mean_intensity", "contrast"],
    )
    return (
        df,
        n_shadow_rejected_left,
        n_shadow_rejected_right,
        n_ellipse_rejected,
        n_singletons,
        n_split,
        n_split_subs,
        n_split_fallback,
    )


if __name__ == "__main__":
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.VIDEO_PATH

    t0 = time.perf_counter()
    (
        df,
        n_shadow_rejected_left,
        n_shadow_rejected_right,
        n_ellipse_rejected,
        n_singletons,
        n_split,
        n_split_subs,
        n_split_fallback,
    ) = run_detection(video_path)
    elapsed = time.perf_counter() - t0

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.OUTPUT_DIR / f"{video_path.stem}_detections.csv"
    df.to_csv(out_path, index=False)

    # Total frame count for zero-detection summary (all frames are processed)
    _cap = cv2.VideoCapture(str(video_path))
    _n_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    total_frames = max(0, _n_frames)
    n_det_frames = df["frame"].nunique() if not df.empty else 0
    zero_det_frames = total_frames - n_det_frames
    mean_per_frame = len(df) / n_det_frames if n_det_frames > 0 else 0.0

    print(f"Video                 : {video_path}")
    print(f"Output                : {out_path}")
    print(f"Total detections      : {len(df):,}")
    print(f"Mean / frame          : {mean_per_frame:.1f}")
    print(f"Zero-detection frames : {zero_det_frames} / {total_frames}")
    if cfg.SHADOW_FILTER_ENABLED:
        print(
            f"Shadow filter         : enabled "
            f"(margin={cfg.SHADOW_FILTER_MARGIN}, "
            f"left<{cfg.SHADOW_FILTER_ZONE_WIDTH}px, "
            f"right>frame-{cfg.SHADOW_FILTER_ZONE_WIDTH_RIGHT}px)"
        )
    else:
        print("Shadow filter         : DISABLED")
    n_shadow_rejected = n_shadow_rejected_left + n_shadow_rejected_right
    print(
        f"Shadow filter rejects : {n_shadow_rejected} total "
        f"(left={n_shadow_rejected_left}, right={n_shadow_rejected_right})"
    )
    print(
        f"Ellipse filter rejects: {n_ellipse_rejected} "
        f"(major>{cfg.MAX_MAJOR_AXIS_PX}px or aspect>{cfg.MAX_ASPECT_RATIO})"
    )
    print(
        f"Watershed splitting   : threshold={cfg.WATERSHED_SPLIT_THRESHOLD_PX2:.0f}px²"
    )
    print(f"  singleton contours  : {n_singletons:,}")
    print(f"  split contours      : {n_split:,}  → {n_split_subs:,} sub-detections")
    print(f"  fallback (no peaks) : {n_split_fallback:,}")
    if not df.empty and "major_axis_px" in df.columns:
        n_valid = int(df["major_axis_px"].notna().sum())
        n_nan = len(df) - n_valid
        print(f"Ellipse fits (valid)  : {n_valid} / {len(df)} ({100 * n_valid / len(df):.1f}%)")
        print(f"Inf/NaN ellipse fits  : {n_nan} (will not contribute to size stats)")
        print(f"Mean major axis       : {df['major_axis_px'].mean():.1f} px")
    if not df.empty and "mean_intensity" in df.columns:
        iv = df["mean_intensity"]
        p5, p25, p50, p75, p95 = np.percentile(iv, [5, 25, 50, 75, 95])
        print(f"Intensity min         : {iv.min():.1f}")
        print(f"Intensity p5/p25/p50  : {p5:.1f} / {p25:.1f} / {p50:.1f}")
        print(f"Intensity p75/p95/max : {p75:.1f} / {p95:.1f} / {iv.max():.1f}")
    print(f"Runtime               : {elapsed:.1f}s")
