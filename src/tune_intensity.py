from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.tracking import run_tracking
from src.visualize_tracks import annotate_video


def tune_intensity_zone(
    detections_csv: Path,
    video_path: Path,
    threshold: float,
    zone_width: int,
    output_path: Path,
) -> None:
    """Apply a left-zone intensity filter and render an annotated video.

    Only detections in the left zone (x < zone_width) are subject to the
    intensity filter. Detections at x >= zone_width are kept unconditionally,
    regardless of their intensity.

    No production files (detections CSV, tracks CSV, config) are modified.

    The "before" track count is read from the production tracks CSV when it
    exists to avoid re-running tracking twice.

    Args:
        detections_csv: Detections CSV with a mean_intensity column.
        video_path: Source video for frame extraction.
        threshold: In the left zone, drop detections with mean_intensity > threshold.
        zone_width: Pixel column defining the left zone; detections with
            x >= zone_width are unaffected by the filter.
        output_path: Desired output path (.mp4); may become .avi on codec failure.
    """
    dets_all = pd.read_csv(detections_csv)
    n_before = len(dets_all)

    if "mean_intensity" not in dets_all.columns:
        print(
            f"mean_intensity column not found in {detections_csv}.\n"
            "Re-run detection first:  python -m src.detection",
            file=sys.stderr,
        )
        sys.exit(1)

    # Reject where BOTH conditions hold: inside the left zone AND too bright.
    reject_mask = (dets_all["x"] < zone_width) & (dets_all["mean_intensity"] > threshold)
    dets_filtered = dets_all[~reject_mask].copy()
    n_after = len(dets_filtered)
    n_dropped = n_before - n_after
    pct_dropped = 100.0 * n_dropped / n_before if n_before > 0 else 0.0

    # "Before" track count: prefer the production tracks CSV to avoid a second
    # tracking run. Fall back to re-tracking only when the CSV is absent.
    stem = detections_csv.stem.replace("_detections", "")
    prod_tracks_csv = detections_csv.parent / f"{stem}_tracks.csv"
    if prod_tracks_csv.exists():
        _prod = pd.read_csv(prod_tracks_csv, usecols=["track_id"])
        n_tracks_before = int(_prod["track_id"].nunique())
        before_source = "existing tracks.csv"
    else:
        print("No production tracks.csv found — tracking unfiltered detections for comparison …")
        _t_before = run_tracking(dets_all, progress=False)
        n_tracks_before = int(_t_before["track_id"].nunique()) if not _t_before.empty else 0
        before_source = "re-computed"

    print(f"Running tracking on {n_after:,} filtered detections …")
    tracks_after = run_tracking(dets_filtered, progress=True)
    n_tracks_after = int(tracks_after["track_id"].nunique()) if not tracks_after.empty else 0

    print("Rendering annotated video …")
    actual_out = annotate_video(video_path, tracks_after, output_path)

    print()
    print(f"Threshold              : {threshold:g}")
    print(f"Zone width             : {zone_width} px  (filter applies where x < {zone_width})")
    print(f"Detections before      : {n_before:,}")
    print(f"Detections after filter: {n_after:,}")
    print(f"Detections rejected    : {n_dropped:,}  ({pct_dropped:.1f}%)")
    print(f"Unique tracks before   : {n_tracks_before}  [{before_source}]")
    print(f"Unique tracks after    : {n_tracks_after}")
    print(f"Output                 : {actual_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sandbox: left-zone intensity filter → rerun tracking → annotated video. "
            "Writes nothing to production outputs."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=140.0,
        metavar="N",
        help=(
            "In the left zone, drop detections with mean_intensity > N "
            "(default: 140)."
        ),
    )
    parser.add_argument(
        "--zone-width",
        type=int,
        default=150,
        metavar="PX",
        help=(
            "Left-zone width in pixels; detections with x >= zone-width "
            "are kept regardless of intensity (default: 150)."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        metavar="SUFFIX",
        help="Optional suffix appended to the output filename for run comparison.",
    )
    args = parser.parse_args()

    stem = cfg.VIDEO_PATH.stem
    det_csv = cfg.OUTPUT_DIR / f"{stem}_detections.csv"

    if not det_csv.exists():
        print(f"Detections file not found: {det_csv}", file=sys.stderr)
        sys.exit(1)

    name_part = f"_{args.output_name}" if args.output_name else ""
    out_path = (
        cfg.OUTPUT_DIR
        / f"intensity_tune_t{args.threshold:g}_z{args.zone_width}{name_part}.mp4"
    )

    tune_intensity_zone(det_csv, cfg.VIDEO_PATH, args.threshold, args.zone_width, out_path)
