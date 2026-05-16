from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import config as cfg

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_mm_columns(
    tracklet_summary: pd.DataFrame,
    speeds_df: pd.DataFrame,
    pixels_per_mm: float,
) -> None:
    """Append mm-unit columns to tracklet_summary and speeds_df in place."""
    mm_per_px = 1.0 / pixels_per_mm
    tracklet_summary["mean_speed_mm_s"] = tracklet_summary["mean_speed_px_s"] * mm_per_px
    tracklet_summary["max_speed_mm_s"] = tracklet_summary["max_speed_px_s"] * mm_per_px
    tracklet_summary["total_distance_mm"] = tracklet_summary["total_distance_px"] * mm_per_px
    speeds_df["speed_mm_s"] = speeds_df["speed_px_s"] * mm_per_px


def _mean_angular_velocity_deg_s(
    xy: np.ndarray,
    frames: np.ndarray,
    fps: float,
    min_disp: float,
) -> float:
    """Mean absolute angular velocity (deg/s) for one tracklet.

    Steps with displacement < min_disp are skipped to suppress heading
    estimates dominated by position noise. Requires at least 3 valid steps
    to produce two headings and one angular change; returns NaN otherwise.

    Angular differences are wrapped to [-180, 180] before taking absolute
    value, so a 350° clockwise change is treated as 10° counter-clockwise.

    Per-step angular velocity = |Δheading| / Δtime is averaged across all
    valid consecutive heading pairs. The time denominator uses the actual
    frame gap between the ends of consecutive valid steps.

    Args:
        xy: Shape (N, 2) array of [x, y] in chronological order.
        frames: Integer frame indices, same length as xy.
        fps: Frames per second.
        min_disp: Minimum Euclidean displacement (px) for a step to count.

    Returns:
        Mean absolute angular velocity in deg/s, or NaN if fewer than 3
        valid displacement steps exist.
    """
    if len(xy) < 3:
        return float("nan")

    headings: list[float] = []
    head_frames: list[int] = []

    for i in range(len(xy) - 1):
        dx = float(xy[i + 1, 0] - xy[i, 0])
        dy = float(xy[i + 1, 1] - xy[i, 1])
        if np.hypot(dx, dy) >= min_disp:
            headings.append(np.degrees(np.arctan2(dy, dx)))
            head_frames.append(int(frames[i + 1]))

    if len(headings) < 3:
        return float("nan")

    h = np.asarray(headings)
    f = np.asarray(head_frames, dtype=float)

    delta_deg = np.diff(h)
    delta_deg = (delta_deg + 180.0) % 360.0 - 180.0   # wrap to [-180, 180]
    delta_t = np.diff(f) / fps

    valid = delta_t > 0
    if not valid.any():
        return float("nan")

    return float((np.abs(delta_deg[valid]) / delta_t[valid]).mean())


def _compute_yfrac(
    tracks_df: pd.DataFrame,
    frame_height: int,
) -> pd.DataFrame:
    """Per-tracklet fraction of frames in each vertical third of the ROI.

    The vertical ROI is y in (cfg.ROI_TOP, frame_height - cfg.ROI_BOTTOM).
    The range is split into three equal-height bands — top, center, bot —
    and each frame of each tracklet is classified into one. Frames with y
    outside the ROI bounds are clipped to the nearest band (rare in
    practice: detection zeroes those rows of the foreground mask).

    Args:
        tracks_df: DataFrame with columns track_id, y.
        frame_height: Source video frame height in pixels.

    Returns:
        DataFrame indexed by track_id with columns
        yfrac_top, yfrac_center, yfrac_bot. Rows sum to 1.0.
    """
    if tracks_df.empty:
        return pd.DataFrame(columns=["yfrac_top", "yfrac_center", "yfrac_bot"])

    roi_top = float(cfg.ROI_TOP)
    roi_bot = float(frame_height - cfg.ROI_BOTTOM)
    if roi_bot <= roi_top:
        raise ValueError(
            f"Invalid vertical ROI: ROI_TOP={cfg.ROI_TOP}, "
            f"ROI_BOTTOM={cfg.ROI_BOTTOM}, frame_height={frame_height}"
        )
    third = (roi_bot - roi_top) / 3.0
    edge_top_center = roi_top + third
    edge_center_bot = roi_top + 2.0 * third

    y = tracks_df["y"].to_numpy(dtype=float)
    band = np.where(
        y < edge_top_center, "top",
        np.where(y < edge_center_bot, "center", "bot"),
    )

    df = pd.DataFrame({
        "track_id": tracks_df["track_id"].to_numpy(),
        "band": band,
    })
    counts = df.groupby(["track_id", "band"]).size().unstack(fill_value=0)
    for col in ("top", "center", "bot"):
        if col not in counts.columns:
            counts[col] = 0
    totals = counts[["top", "center", "bot"]].sum(axis=1).to_numpy(dtype=float)
    fracs = counts[["top", "center", "bot"]].to_numpy(dtype=float) / totals[:, None]
    out = pd.DataFrame(
        fracs,
        index=counts.index,
        columns=["yfrac_top", "yfrac_center", "yfrac_bot"],
    )
    out.index.name = "track_id"
    return out


def _join_body_size_to_tracks(
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ellipse columns from detections_df to each row in tracks_df.

    tracks_df has Kalman-filtered positions; detections_df has raw contour
    positions. They share a frame index but have no direct track-id link.
    For each track position, the nearest detection in the same frame by
    Euclidean distance is found and its ellipse measurements are copied.

    If detections_df lacks the ellipse columns (pre-Phase-D.5 data),
    all three columns are filled with NaN.

    Args:
        tracks_df: Full tracks DataFrame with columns frame, x, y.
        detections_df: Detection DataFrame from run_detection, expected to
            contain major_axis_px, minor_axis_px, orientation_deg.

    Returns:
        Copy of tracks_df with major_axis_px, minor_axis_px,
        orientation_deg appended. NaN where no same-frame detection exists.
    """
    ellipse_cols = ["major_axis_px", "minor_axis_px", "orientation_deg"]
    out = tracks_df.copy()
    for c in ellipse_cols:
        out[c] = float("nan")

    if not all(c in detections_df.columns for c in ellipse_cols):
        return out

    det_by_frame: dict[int, pd.DataFrame] = {
        int(f): g.reset_index(drop=True)
        for f, g in detections_df.groupby("frame")
    }

    for frame_idx, trk_group in tracks_df.groupby("frame"):
        dets = det_by_frame.get(int(frame_idx))
        if dets is None or dets.empty:
            continue
        tx = trk_group["x"].to_numpy(dtype=float)
        ty = trk_group["y"].to_numpy(dtype=float)
        det_x = dets["x"].to_numpy(dtype=float)
        det_y = dets["y"].to_numpy(dtype=float)
        dist2 = (tx[:, np.newaxis] - det_x) ** 2 + (ty[:, np.newaxis] - det_y) ** 2
        nearest = np.argmin(dist2, axis=1)   # shape (n_tracks_in_frame,)
        for col in ellipse_cols:
            out.loc[trk_group.index, col] = dets[col].to_numpy()[nearest]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_per_frame_speeds(
    tracks_df: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """Compute per-frame speed in px/s for every tracklet.

    Speed is derived from consecutive (x, y) positions rather than from the
    Kalman vx/vy estimates. The Kalman velocity is a filtered quantity that
    lags real motion; finite differences from positions reflect what the fish
    actually did each frame.

    When consecutive rows for the same track_id are separated by more than one
    frame (a track absent for one frame then reconfirmed), the distance is
    divided by the actual frame gap rather than assuming a 1-frame step.
    This gives a correct px/s value even when the tracker skips frames.

    The first frame of each tracklet produces no speed (no prior position).
    Those rows are dropped so every returned row has a valid speed.

    Args:
        tracks_df: DataFrame with columns frame, track_id, x, y.
        fps: Frames per second, used to convert distances to px/s.

    Returns:
        DataFrame with columns: frame, track_id, speed_px_s.
        One row per (track_id, frame) pair, first frame of each tracklet excluded.
    """
    if tracks_df.empty:
        return pd.DataFrame(columns=["frame", "track_id", "speed_px_s"])

    t = tracks_df.sort_values(["track_id", "frame"]).copy()
    gb = t.groupby("track_id", sort=False)

    dx = gb["x"].diff()
    dy = gb["y"].diff()
    delta_frames = gb["frame"].diff()           # actual frame gap (≥1)

    dist_px = np.sqrt(dx ** 2 + dy ** 2)
    t["speed_px_s"] = dist_px * fps / delta_frames

    result = t.dropna(subset=["speed_px_s"])[["frame", "track_id", "speed_px_s"]]
    return result.reset_index(drop=True)


def compute_tracklet_summary(
    tracks_df: pd.DataFrame,
    speeds_df: pd.DataFrame,
    fps: float,
    min_tracklet_frames: int,
    frame_height: int,
    detections_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate per-tracklet statistics including Cho et al. 2022 metrics.

    Five paper-aligned features are computed alongside the existing metrics:

    - mean_major_axis_px: median of per-frame major ellipse axis across the
      tracklet. Named "mean" following Cho et al. convention; median is used
      for robustness to single-frame fitting artifacts.
    - major_axis_iqr_px: IQR of major axis — within-tracklet variance useful
      for detecting partial detections or size estimation noise.
    - sd_speed_px_s: within-tracklet speed standard deviation.
    - fwdrun_fraction: fraction of speed measurements exceeding
      cfg.FWDRUN_THRESHOLD_MULTIPLIER × population mean speed. Two-pass:
      population mean is computed first from qualifying tracklets (n_frames >=
      min_tracklet_frames), then per-tracklet fractions are derived.
    - mean_angular_velocity_deg_s: mean absolute turning rate in deg/s.
      NaN when fewer than 3 valid-displacement steps exist.

    total_distance_px is computed directly from consecutive position pairs in
    tracks_df because recovering distance from speeds_df would require the
    frame-gap denominator that speeds_df does not carry.

    Body-size columns require detections_df from Phase D.5 detection output.
    When detections_df is None or missing ellipse columns, those columns are NaN.

    Args:
        tracks_df: Full tracks DataFrame (frame, track_id, x, y).
        speeds_df: Output of compute_per_frame_speeds.
        fps: Frames per second.
        min_tracklet_frames: Minimum n_frames for a tracklet to contribute
            to the population mean speed used in FwdRun thresholding.
        frame_height: Source video frame height in pixels, used to define
            the vertical ROI for the YFraction bands.
        detections_df: Optional raw detections DataFrame with ellipse columns.

    Returns:
        DataFrame with one row per track_id. Columns:
        track_id, n_frames, duration_s,
        mean_speed_px_s, sd_speed_px_s, max_speed_px_s,
        total_distance_px,
        fwdrun_fraction, mean_angular_velocity_deg_s,
        mean_major_axis_px, major_axis_iqr_px,
        yfrac_top, yfrac_center, yfrac_bot.
    """
    if tracks_df.empty:
        return pd.DataFrame(columns=[
            "track_id", "n_frames", "duration_s",
            "mean_speed_px_s", "sd_speed_px_s", "max_speed_px_s",
            "total_distance_px",
            "fwdrun_fraction", "mean_angular_velocity_deg_s",
            "mean_major_axis_px", "major_axis_iqr_px",
            "yfrac_top", "yfrac_center", "yfrac_bot",
        ])

    t = tracks_df.sort_values(["track_id", "frame"])

    # n_frames: all positions including first (which has no speed)
    n_frames = t.groupby("track_id")["frame"].count().rename("n_frames")

    # total_distance_px from consecutive position diffs
    gb = t.groupby("track_id", sort=False)
    dist_px = np.sqrt(gb["x"].diff() ** 2 + gb["y"].diff() ** 2)
    total_dist = dist_px.groupby(t["track_id"]).sum().rename("total_distance_px")

    # Speed aggregates from speeds_df
    if speeds_df.empty:
        speed_agg = pd.DataFrame(
            index=n_frames.index,
            columns=["mean_speed_px_s", "sd_speed_px_s", "max_speed_px_s"],
            dtype=float,
        )
    else:
        speed_agg = speeds_df.groupby("track_id")["speed_px_s"].agg(
            mean_speed_px_s="mean",
            sd_speed_px_s="std",
            max_speed_px_s="max",
        )

    # Angular velocity — explicit loop avoids DataFrameGroupBy.apply version concerns
    ang_vel_dict: dict[int, float] = {}
    for tid, group in t.groupby("track_id"):
        grp = group.sort_values("frame")
        ang_vel_dict[int(tid)] = _mean_angular_velocity_deg_s(
            grp[["x", "y"]].to_numpy(dtype=float),
            grp["frame"].to_numpy(),
            fps,
            cfg.MIN_DISPLACEMENT_FOR_HEADING_PX,
        )
    ang_vel = pd.Series(ang_vel_dict, name="mean_angular_velocity_deg_s")
    ang_vel.index.name = "track_id"

    summary = pd.concat([n_frames, speed_agg, total_dist, ang_vel], axis=1)
    summary = summary.reset_index()

    summary["mean_speed_px_s"] = summary["mean_speed_px_s"].fillna(0.0)
    summary["sd_speed_px_s"] = summary["sd_speed_px_s"].fillna(0.0)
    summary["max_speed_px_s"] = summary["max_speed_px_s"].fillna(0.0)
    summary["total_distance_px"] = summary["total_distance_px"].fillna(0.0)
    # mean_angular_velocity_deg_s: preserve NaN for short/stationary tracklets

    summary["duration_s"] = summary["n_frames"] / fps

    # FwdRun — two-pass: population mean from qualifying tracklets first,
    # then per-tracklet fraction derived against that threshold.
    qualifying = summary[summary["n_frames"] >= min_tracklet_frames]
    if len(qualifying) > 0:
        w = qualifying["n_frames"].to_numpy(dtype=float)
        pop_mean_speed = float(
            np.average(qualifying["mean_speed_px_s"].to_numpy(), weights=w)
        )
    else:
        pop_mean_speed = 0.0

    fwdrun_threshold = pop_mean_speed * cfg.FWDRUN_THRESHOLD_MULTIPLIER

    if speeds_df.empty or fwdrun_threshold <= 0:
        summary["fwdrun_fraction"] = 0.0
    else:
        fwdrun = (
            speeds_df.groupby("track_id")["speed_px_s"]
            .apply(lambda s: float((s > fwdrun_threshold).mean()))
            .rename("fwdrun_fraction")
        )
        summary = summary.merge(fwdrun.reset_index(), on="track_id", how="left")
        summary["fwdrun_fraction"] = summary["fwdrun_fraction"].fillna(0.0)

    # Body size — nearest-centroid join with raw detections per frame
    if detections_df is not None and not detections_df.empty:
        joined = _join_body_size_to_tracks(t, detections_df)
        major = joined.groupby("track_id")["major_axis_px"]
        body_agg = pd.DataFrame({
            "mean_major_axis_px": major.median(),
            "major_axis_iqr_px": major.quantile(0.75) - major.quantile(0.25),
        })
    else:
        body_agg = pd.DataFrame(
            {"mean_major_axis_px": np.nan, "major_axis_iqr_px": np.nan},
            index=n_frames.index,
        )

    summary = summary.merge(body_agg.reset_index(), on="track_id", how="left")

    # Vertical occupancy across three equal bands of the detection ROI.
    yfrac = _compute_yfrac(t, frame_height)
    summary = summary.merge(yfrac.reset_index(), on="track_id", how="left")

    return summary[[
        "track_id", "n_frames", "duration_s",
        "mean_speed_px_s", "sd_speed_px_s", "max_speed_px_s",
        "total_distance_px",
        "fwdrun_fraction", "mean_angular_velocity_deg_s",
        "mean_major_axis_px", "major_axis_iqr_px",
        "yfrac_top", "yfrac_center", "yfrac_bot",
    ]]


def compute_population_summary(
    tracklet_summary: pd.DataFrame,
    speeds_df: pd.DataFrame,
    min_tracklet_frames: int,
) -> dict:
    """Compute the paper-aligned population-level summary.

    Tracklets shorter than min_tracklet_frames are excluded before computing
    any aggregate. Population means are weighted by tracklet length so
    longer (and therefore more reliable) tracklets contribute more. Body-size
    uses a NaN-safe weighted mean; tracklets without a valid major-axis
    estimate are excluded from that aggregate.

    The output is restricted to the Cho et al. 2022 paper metrics under the
    paper's naming. The three length/speed quantities are emitted in both
    pixel units (verbatim from the tracking output) and millimetre units
    (using cfg.PIXELS_PER_MM, derived from the ROI and known tank size).

    Args:
        tracklet_summary: Output of compute_tracklet_summary.
        speeds_df: Output of compute_per_frame_speeds.
        min_tracklet_frames: Minimum tracklet length to include.

    Returns:
        Dict with ten keys, in order: "Major axis (px)", "Speed (px/s)",
        "SD of Speed (px/s)", "Major axis (mm)", "Speed (mm/s)",
        "SD of Speed (mm/s)", "FwdRun", "YFractionTop", "YFractionCenter",
        "YFractionBottom".
    """
    valid = tracklet_summary[tracklet_summary["n_frames"] >= min_tracklet_frames].copy()
    n_qualifying = int(len(valid))

    inv_pxmm = 1.0 / cfg.PIXELS_PER_MM

    if n_qualifying == 0:
        return {
            "Major axis (px)": float("nan"),
            "Speed (px/s)": 0.0,
            "SD of Speed (px/s)": 0.0,
            "Major axis (mm)": float("nan"),
            "Speed (mm/s)": 0.0,
            "SD of Speed (mm/s)": 0.0,
            "FwdRun": 0.0,
            "YFractionTop": 0.0,
            "YFractionCenter": 0.0,
            "YFractionBottom": 0.0,
        }

    weights = valid["n_frames"].to_numpy(dtype=float)

    mean_speed = float(np.average(valid["mean_speed_px_s"].to_numpy(), weights=weights))
    mean_sd_speed = float(np.average(valid["sd_speed_px_s"].to_numpy(), weights=weights))
    mean_fwdrun = float(np.average(valid["fwdrun_fraction"].to_numpy(), weights=weights))

    # Body size: NaN-safe weighted mean across qualifying tracklets.
    if "mean_major_axis_px" in valid.columns:
        major_vals = valid["mean_major_axis_px"].to_numpy(dtype=float)
        major_mask = ~np.isnan(major_vals)
        if major_mask.any():
            mean_major = float(np.average(major_vals[major_mask], weights=weights[major_mask]))
        else:
            mean_major = float("nan")
    else:
        mean_major = float("nan")

    # YFraction: tracklet-length-weighted means across the three bands.
    # The per-tracklet rows already sum to 1.0, so the weighted means
    # naturally sum to 1.0 too — no further normalization needed.
    yfrac_top = float(np.average(valid["yfrac_top"].to_numpy(dtype=float), weights=weights))
    yfrac_center = float(np.average(valid["yfrac_center"].to_numpy(dtype=float), weights=weights))
    yfrac_bot = float(np.average(valid["yfrac_bot"].to_numpy(dtype=float), weights=weights))

    return {
        "Major axis (px)": mean_major,
        "Speed (px/s)": mean_speed,
        "SD of Speed (px/s)": mean_sd_speed,
        "Major axis (mm)": mean_major * inv_pxmm,
        "Speed (mm/s)": mean_speed * inv_pxmm,
        "SD of Speed (mm/s)": mean_sd_speed * inv_pxmm,
        "FwdRun": mean_fwdrun,
        "YFractionTop": yfrac_top,
        "YFractionCenter": yfrac_center,
        "YFractionBottom": yfrac_bot,
    }


def run_analysis(
    tracks_csv: Path,
    video_path: Path,
    output_dir: Path,
    pixels_per_mm: float | None = None,
    detections_csv: Path | None = None,
) -> dict:
    """Orchestrate the full analysis pipeline.

    Loads tracks CSV, reads video metadata, calls all analysis functions with
    thresholds from src/config.py, saves outputs to output_dir, and returns
    the population summary dict.

    Output files (where {stem} = tracks_csv.stem with '_tracks' stripped):
      {stem}_tracklet_summary.csv   — per-tracklet stats
      {stem}_summary.csv            — one-row population summary

    If pixels_per_mm is provided, additional *_mm and *_mm_s columns are
    appended to tracklet_summary and speeds before saving.

    If detections_csv is provided or auto-derived, raw detections are used
    for body-size join in compute_tracklet_summary. Auto-derivation looks for
    {stem}_detections.csv in output_dir.

    Args:
        tracks_csv: Path to the tracks CSV produced by run_tracking.
        video_path: Path to the source video (read for fps and frame dimensions).
        output_dir: Directory where output files are written.
        pixels_per_mm: Optional spatial calibration for unit conversion.
        detections_csv: Optional path to the detections CSV. When None, the
            function looks for {stem}_detections.csv in output_dir.

    Returns:
        Population summary dict (same as compute_population_summary output).
    """
    tracks_df = pd.read_csv(tracks_csv)
    stem = tracks_csv.stem.replace("_tracks", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Auto-derive detections CSV when not provided
    if detections_csv is None:
        candidate = output_dir / f"{stem}_detections.csv"
        if candidate.exists():
            detections_csv = candidate

    detections_df: pd.DataFrame | None = None
    if detections_csv is not None and Path(detections_csv).exists():
        detections_df = pd.read_csv(detections_csv)

    speeds_df = compute_per_frame_speeds(tracks_df, fps)

    tracklet_summary = compute_tracklet_summary(
        tracks_df, speeds_df,
        fps=fps,
        min_tracklet_frames=cfg.MIN_TRACKLET_FRAMES_FOR_SUMMARY,
        frame_height=frame_height,
        detections_df=detections_df,
    )

    if pixels_per_mm is not None:
        _add_mm_columns(tracklet_summary, speeds_df, pixels_per_mm)

    pop_summary = compute_population_summary(
        tracklet_summary, speeds_df,
        min_tracklet_frames=cfg.MIN_TRACKLET_FRAMES_FOR_SUMMARY,
    )

    tracklet_summary.to_csv(output_dir / f"{stem}_tracklet_summary.csv", index=False)
    pd.DataFrame([pop_summary]).to_csv(output_dir / f"{stem}_summary.csv", index=False)

    return pop_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Daphnia trajectory analysis.")
    parser.add_argument(
        "tracks_csv",
        nargs="?",
        default=str(cfg.OUTPUT_DIR / f"{cfg.VIDEO_PATH.stem}_tracks.csv"),
        help="Path to tracks CSV (default: output/image_3_tracks.csv)",
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(cfg.VIDEO_PATH),
        help="Path to source video (default: cfg.VIDEO_PATH)",
    )
    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        default=None,
        metavar="N",
        help="Spatial calibration: pixels per mm. Adds _mm/_mm_s columns when provided.",
    )
    parser.add_argument(
        "--detections-csv",
        type=str,
        default=None,
        metavar="CSV",
        help="Path to detections CSV for body-size join. "
             "Auto-derived from tracks stem in output/ if omitted.",
    )
    args = parser.parse_args()

    det_csv = Path(args.detections_csv) if args.detections_csv else None

    result = run_analysis(
        Path(args.tracks_csv),
        Path(args.video_path),
        cfg.OUTPUT_DIR,
        pixels_per_mm=args.pixels_per_mm,
        detections_csv=det_csv,
    )

    print("\n--- Population Summary ---")
    print(
        f"  Tank assumption: {cfg.TANK_WIDTH_MM:.0f} mm x {cfg.TANK_HEIGHT_MM:.0f} mm "
        f"swimming arena -> {cfg.PIXELS_PER_MM:.2f} px/mm"
    )
    for key, val in result.items():
        if isinstance(val, float):
            if np.isnan(val):
                print(f"  {key}: nan")
            else:
                print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
