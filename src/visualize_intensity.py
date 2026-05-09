from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg

_THRESHOLDS = (90, 100, 110, 120, 130)


def visualize_intensity(
    detections_csv: Path,
    output_path: Path,
    thresholds: tuple[float, ...] = _THRESHOLDS,
) -> None:
    """Plot a log-scale histogram of mean_intensity across all detections.

    Vertical dashed lines mark candidate filter thresholds so the user can
    visually read off how many detections each threshold would keep/drop.
    The exact counts are also printed to stdout in a table.

    Args:
        detections_csv: Detections CSV from run_detection.
        output_path: Destination PNG path.
        thresholds: Candidate intensity thresholds to annotate.
    """
    df = pd.read_csv(detections_csv)

    if "mean_intensity" not in df.columns:
        print(f"No mean_intensity column in {detections_csv}", file=sys.stderr)
        sys.exit(1)

    iv = df["mean_intensity"].dropna()
    n_nan = int(df["mean_intensity"].isna().sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(iv.to_numpy(), bins=50, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Mean grayscale intensity inside contour (0 = black, 255 = white)")
    ax.set_ylabel("Detection count (log scale)")
    ax.set_title(f"Detection intensity distribution  (n={len(iv):,}, NaN skipped: {n_nan})")

    for t in thresholds:
        ax.axvline(t, color="crimson", linestyle="--", linewidth=0.9, alpha=0.8, label=f"t={t:.0f}")
    if thresholds:
        ax.legend(fontsize=8, title="Candidate thresholds")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    print(f"Detections total   : {len(df):,}")
    print(f"With intensity     : {len(iv):,}")
    print(f"NaN (skipped)      : {n_nan}")
    print(f"Output             : {output_path}")
    print()
    print(f"{'Threshold':>10}  {'Below':>8}  {'Above/eq':>10}  {'% below':>8}")
    for t in thresholds:
        n_below = int((iv < t).sum())
        n_above = int((iv >= t).sum())
        pct = 100.0 * n_below / len(iv) if len(iv) else 0.0
        print(f"{t:>10.0f}  {n_below:>8,}  {n_above:>10,}  {pct:>8.1f}%")


if __name__ == "__main__":
    det_csv = cfg.OUTPUT_DIR / f"{cfg.VIDEO_PATH.stem}_detections.csv"
    out_path = cfg.OUTPUT_DIR / "intensity_histogram.png"

    if not det_csv.exists():
        print(f"Detections file not found: {det_csv}", file=sys.stderr)
        sys.exit(1)

    visualize_intensity(det_csv, out_path)
