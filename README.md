# Daphnia Tracker

## Overview

Side-view computer-vision pipeline that tracks individual *Daphnia* (water fleas) in a 165 mm × 145 mm aquarium recorded from the side at ~20 fps, then summarises the population using the five paper-aligned metrics from Cho et al. 2022 (Major axis, Speed, SD of Speed, FwdRun, YFraction). The pipeline is classical CV throughout — MOG2 background subtraction → contour + watershed splitting → SORT-style Kalman tracking with size-aware association → per-tracklet and population summaries — no deep learning. The repository is named `fish-tracker` for legacy reasons; the subject is *Daphnia*, not fish.

## Setup

Python 3.10+ (developed and tested on 3.12).

1. Clone or unzip the repository.
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS / Linux: `source .venv/bin/activate`
4. Install pinned dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Put your video at `data/your_video.mp4` (any name works; the pipeline derives output filenames from the video stem).
6. Edit `src/config.py` and update `VIDEO_PATH` to point at your file. If your video has very different resolution, lighting, ROI margins, or animal density, also retune the parameters using `notebooks/01_explore.ipynb` (see *Tuning for a new video* below).

## Running the pipeline

Run the four stages in order from the repository root. Each reads from `output/` (and `data/`) and writes to `output/`. Filenames are prefixed by the video stem; the examples below assume `data/image_3.mp4`.

```
python -m src.detection        # ~7 min on 400 frames @ 1280x1024
python -m src.tracking         # ~15 s
python -m src.analysis         # ~3 s
python -m src.visualize_tracks # ~30 s for the full annotated video
```

- **detection** — reads `data/<video>.mp4`, writes `output/<stem>_detections.csv`. Warms up the MOG2 background model on the first 60 frames, rewinds, then runs every frame through the frozen model. Slow because the warmup-rewind processes the video twice.
- **tracking** — reads `output/<stem>_detections.csv`, writes `output/<stem>_tracks.csv`. Per-frame Hungarian association on Euclidean distance plus a size cost, with Kalman prediction for coast frames.
- **analysis** — reads `output/<stem>_tracks.csv` (and the matching detections CSV for body-size joins), writes `output/<stem>_tracklet_summary.csv` and `output/<stem>_summary.csv`. Prints the seven population summary keys to stdout.
- **visualize_tracks** — reads the same two CSVs and the original video, writes `output/<stem>_annotated.mp4`. Add `--frame N` to render a single annotated PNG instead.

## Output files

All written to `output/`. `{stem}` is the video filename without the `.mp4` extension.

| File | Stage | Contents |
|---|---|---|
| `{stem}_detections.csv` | detection | One row per kept blob: `frame, x, y, w, h, area, major_axis_px, minor_axis_px, orientation_deg, mean_intensity, contrast` |
| `{stem}_tracks.csv` | tracking | One row per confirmed-track-per-frame: `frame, track_id, x, y, vx, vy` (Kalman-filtered positions and velocities) |
| `{stem}_tracklet_summary.csv` | analysis | One row per tracklet with verbose per-tracklet statistics including body-size, speed, FwdRun, angular-velocity, and YFraction columns |
| `{stem}_summary.csv` | analysis | One row, seven columns — the population-level paper metrics (see below) |
| `{stem}_annotated.mp4` | visualization | The original video with size-coloured ellipses and per-track motion trails overlaid |

## The 5 paper metrics

`{stem}_summary.csv` and the stdout report from `python -m src.analysis` are restricted to the five Cho et al. 2022 metrics. YFraction expands into three columns (top / center / bottom) so the file actually has seven keys.

- **Major axis** — population mean of each tracklet's median fitted-ellipse major-axis length (pixels). Indicates typical Daphnia body length.
- **Speed** — tracklet-length-weighted mean per-frame swimming speed across qualifying tracklets, in pixels per second.
- **SD of Speed** — tracklet-length-weighted mean of within-tracklet speed standard deviation, in pixels per second. Captures how variable each animal's speed is over its tracked lifetime.
- **FwdRun** — fraction of speed samples exceeding `1.5 ×` the population-mean speed. Two-pass: the population mean is computed first, then per-tracklet fractions are averaged.
- **YFractionTop / YFractionCenter / YFractionBottom** — population-weighted fraction of tracklet-frames spent in each vertical third of the active ROI. Sums to 1.0.

All speed and length quantities are in pixels by default. Pass `--pixels-per-mm N` to `python -m src.analysis` to add `_mm` / `_mm_s` columns to the per-tracklet CSV. The scale for the supplied tank (165 × 145 mm inside the ROI) is ~6.35 px/mm.

## Tuning for a new video

When swapping in a different recording, start in `notebooks/01_explore.ipynb`. It walks through reading video metadata, building the MOG2 model with the current parameters, plotting the contour-area histogram, overlaying the ROI mask, and counting detections on three representative frames. After settling on values, edit `src/config.py` directly — the production modules import everything from there. Re-running the notebook against the current code is the visual-regression check.

The parameters most sensitive to a new video are `MOG2_VAR_THRESHOLD`, `MIN_AREA`, `MAX_AREA`, and the four `ROI_*` margins. The Kalman noise constants (`KALMAN_*`), tracker gates (`TRACK_*`, `MAX_COAST_DISTANCE_PX`, `SIZE_COST_WEIGHT`), and behavioural thresholds (`FWDRUN_THRESHOLD_MULTIPLIER`, `MIN_DISPLACEMENT_FOR_HEADING_PX`) generally do not need per-video tuning.

## Diagnostic tools

- `src/compare_frames.py` — extracts a fixed set of frames (100, 200, 300) and writes side-by-side raw / annotated PNG pairs. Useful for spot-checking detection and tracking on representative moments.
- `src/tune_contrast.py` — sweeps the shadow-filter `margin` / `zone_width` parameters and renders annotated videos for each setting. **Partially out of date**: it does not exercise watershed splitting, the right-side shadow filter, or the degenerate-ellipse filter — those were added to `src/detection.py` after the tuning sweep was written. Use for shadow-filter exploration only.

## Known limitations

- Identity is occasionally swapped when two or more Daphnia pass within ~10 px of each other. Size-aware association reduces but does not eliminate this, and severe clumps still produce occasional ID switches.
- Per-video population metrics will fluctuate around the cross-experiment averages reported in the paper. Cho et al.'s 8.25 mm/s mean swimming speed is a dataset-wide aggregate, not a per-video target.
- Classical CV is at its ceiling here: fully overlapping Daphnia present as one connected dark region. No amount of contour math reliably recovers two centroids from one blob; watershed splitting handles touching blobs but not true occlusion.
