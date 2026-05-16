# Daphnia Tracker — CLAUDE.md

## What this project is

A side-view *Daphnia* (water flea) behaviour-tracking pipeline written in classical computer vision: OpenCV background subtraction and contour processing, watershed splitting, a SORT-style Kalman tracker with Hungarian assignment, and per-tracklet plus population-level analysis. The deliverable is the population summary printed by `python -m src.analysis` and saved to `output/{stem}_summary.csv`: five Cho et al. 2022 metrics (Major axis, Speed, SD of Speed, FwdRun, YFraction — the last expanded into top / center / bottom). The repository name `fish-tracker` is legacy; the subject is *Daphnia*. No deep learning, no GPU, no manual annotation.

## Pipeline architecture

Four stages, each a standalone module under `src/`. Inputs and outputs are CSVs in `output/`, named by the video stem.

### Stage 1: Detection (`src/detection.py`)
- **Input:** `data/<video>.mp4` (path from `cfg.VIDEO_PATH`)
- **Output:** `output/<stem>_detections.csv` (frame, x, y, w, h, area, major_axis_px, minor_axis_px, orientation_deg, mean_intensity, contrast)
- Builds a MOG2 background model with `cfg.MOG2_WARMUP_FRAMES` worth of warmup, then rewinds and runs every frame through the *frozen* model so detections exist for frame 0.
- Per frame: foreground mask → morphological open → ROI mask → contour finding → area filter → optional watershed split for large contours → ellipse fit → degenerate-ellipse rejection → shadow-filter rejection in left and right zones.
- Each surviving contour produces one detection dict with moments-centroid, bounding box, ellipse axes, mean intensity, and ring-contrast measure.

### Stage 2: Tracking (`src/tracking.py`)
- **Input:** `output/<stem>_detections.csv`
- **Output:** `output/<stem>_tracks.csv` (frame, track_id, x, y, vx, vy)
- One `KalmanTrack` per active object: state `[x, y, vx, vy]`, measurement `[x, y]`, constant-velocity model. Covariances live in `cfg.KALMAN_*`.
- Per frame: predict all tracks → Hungarian-assign detections via Euclidean position cost plus a per-pixel size-difference cost (`cfg.SIZE_COST_WEIGHT`) → gate matches by `cfg.TRACK_MAX_DISTANCE` (predicted-position gate) and `cfg.MAX_COAST_DISTANCE_PX` (last-measured-position gate) → spawn new tracks from unmatched detections → kill tracks whose `time_since_update > cfg.TRACK_MAX_AGE`.
- Each track maintains an EMA size estimate (`alpha=0.3`) updated only on real matches, used by the size cost in subsequent associations.

### Stage 3: Analysis (`src/analysis.py`)
- **Input:** `output/<stem>_tracks.csv` and `output/<stem>_detections.csv` (the latter for the body-size nearest-centroid join)
- **Outputs:**
  - `output/<stem>_tracklet_summary.csv` — one row per tracklet, verbose columns
  - `output/<stem>_summary.csv` — one row, ten columns (the five paper metrics with YFraction expanded; size and speed each emitted in both pixel units and millimetre units via `cfg.PIXELS_PER_MM`)
- `compute_per_frame_speeds` derives speed in px/s from consecutive `(x, y)` positions, dividing by the actual frame gap so single-frame coasting does not inflate speed.
- `compute_tracklet_summary` aggregates per-tracklet means, SDs, FwdRun fraction, mean angular velocity, body-size median + IQR, and vertical-third occupancy fractions.
- `compute_population_summary` reduces to the ten paper-aligned keys via tracklet-length-weighted means over qualifying tracklets (`n_frames >= cfg.MIN_TRACKLET_FRAMES_FOR_SUMMARY`); pixel-unit and millimetre-unit versions of size and speed share the underlying weighted mean.

### Stage 4: Visualization (`src/visualize_tracks.py`)
- **Input:** `output/<stem>_tracks.csv`, `output/<stem>_detections.csv`, source video
- **Output:** `output/<stem>_annotated.mp4` (or `output/<stem>_frame_N.png` with `--frame N`)
- Joins each track row to the nearest detection in the same frame (within `_MATCH_DISTANCE_PX = 20`) to recover orientation; rejects predicted-only rows. Renders an oriented ellipse per active track plus a fading motion trail. Body-size colour bins (red < p25 ≤ green < p75 ≤ blue) come from the per-tracklet major-axis medians in `_tracklet_summary.csv`.

## Config (`src/config.py`)

All tuned values live here. Modules import via `from src import config as cfg` and read at module level.

- **Paths** — `VIDEO_PATH`, `OUTPUT_DIR`, `PROJECT_ROOT`.
- **ROI** — `ROI_TOP`, `ROI_BOTTOM`, `ROI_LEFT`, `ROI_RIGHT`. Pixel margins zeroed in the foreground mask before contour detection; bound the active detection region inside the tank.
- **Spatial calibration** — `FRAME_WIDTH_PX`, `FRAME_HEIGHT_PX`, `TANK_WIDTH_MM`, `TANK_HEIGHT_MM`, `PIXELS_PER_MM`. Frame and tank dimensions feed the derived `PIXELS_PER_MM` used by analysis to emit size and speed in millimetres alongside the pixel-unit values.
- **MOG2** — `MOG2_HISTORY`, `MOG2_VAR_THRESHOLD`, `MOG2_DETECT_SHADOWS`, `MOG2_WARMUP_FRAMES`. Background-subtractor knobs; threshold is the most sensitive to lighting changes.
- **Morphology** — `MORPH_KERNEL_SIZE`, `MORPH_OPEN_ITERATIONS`. Suppress single-pixel and few-pixel noise blobs from the MOG2 mask.
- **Blob filtering** — `MIN_AREA`, `MAX_AREA`, `WATERSHED_SPLIT_THRESHOLD_PX2`. Area bounds for a valid Daphnia contour; large contours above the watershed threshold are attempted-split before being filtered.
- **Degenerate ellipse** — `MAX_MAJOR_AXIS_PX`, `MAX_ASPECT_RATIO`. Reject ellipse fits that obviously don't correspond to a real Daphnia.
- **Shadow filter** — `SHADOW_FILTER_ENABLED`, `SHADOW_FILTER_ZONE_WIDTH`, `SHADOW_FILTER_ZONE_WIDTH_RIGHT`, `SHADOW_FILTER_MARGIN`, `SHADOW_FILTER_RING_GAP_PX`, `SHADOW_FILTER_RING_THICKNESS_PX`. Rejects detections in the left and right tank-wall zones whose contour interior is not sufficiently darker than the surrounding ring (real Daphnia have strongly negative contrast; wall shadows do not).
- **Tracker** — `TRACK_MAX_AGE`, `TRACK_MIN_HITS`, `TRACK_MAX_DISTANCE`, `MAX_COAST_DISTANCE_PX`, `SIZE_COST_WEIGHT`. Lifecycle and gating for the SORT-style tracker.
- **Kalman** — `KALMAN_R_VARIANCE`, `KALMAN_Q_POS_VARIANCE`, `KALMAN_Q_VEL_VARIANCE`. Measurement and process-noise covariances driving the constant-velocity filter's responsiveness.
- **Behavioural** — `MIN_TRACKLET_FRAMES_FOR_SUMMARY`, `FWDRUN_THRESHOLD_MULTIPLIER`, `MIN_DISPLACEMENT_FOR_HEADING_PX`. Inclusion threshold for population stats and Cho et al. 2022 feature parameters.

## What NOT to do (lessons learned)

- **Don't suggest size-aware matching** — already implemented. The Hungarian cost matrix already includes a per-pixel size-difference term scaled by `cfg.SIZE_COST_WEIGHT = 1.0`. Tracks maintain an EMA major-axis estimate that updates only on real matches.
- **Don't suggest track stitching** — was tried, reverted, and the standalone module deleted. The merged-tracklet output looked stable but population metrics shifted negligibly, and the additional configuration surface was not worth the maintenance.
- **Don't suggest Mahalanobis cost** — was tried, reverted. With the current `Q_pos=0.5, Q_vel=2, R=4` calibration, the chi-square gate at `9.21` (2-dof, 99%) collapses to a position residual of ≈8–10 px in steady state and kills real reunions for Daphnia that turn or speed up. Median tracklet length dropped from 41 to 3 (exactly `TRACK_MIN_HITS`). Either widen the gate to ~16 or raise `Q_vel` substantially before considering this again.
- **Don't suggest deep learning** — out of scope. No GPU on the development machine, and the user has been explicit that this is a classical-CV project.
- **Don't back-calculate the px/mm scale from Daphnia size** — the scale comes from *tank geometry*, period. `cfg.PIXELS_PER_MM` is now a derived module-level constant computed from the ROI dimensions and the known physical tank dimensions (`cfg.TANK_WIDTH_MM = 165`, `cfg.TANK_HEIGHT_MM = 145`). It is not a tuning knob and is not derived from any assumption about Daphnia body length. If you ever need to "check the scale", verify the four `ROI_*` margins and the two `TANK_*_MM` constants, not anything about the animals.
- **Don't propose new color schemes for visualization** — several were tried and reverted. The current scheme is three bins on per-tracklet median major axis: red `< p25`, green `[p25, p75)`, blue `≥ p75`. Anything below the bins (no valid ellipse data) is gray. Boundary thresholds come from `{stem}_tracklet_summary.csv` when available, with a fallback to the detection CSV.

## Known limitations

- Identity is occasionally swapped when two or more Daphnia pass within ~10 px of each other. Size-aware association reduces but does not eliminate this; severe clumps still produce ID switches.
- Per-video population metrics fluctuate around the cross-experiment averages reported in Cho et al. The paper's 8.25 mm/s mean swimming speed is a dataset-wide aggregate, not a per-video target.
- Classical CV is at its ceiling for fully overlapping Daphnia: two animals stacked vertically present as a single connected dark region. No contour math reliably recovers two centroids from one blob; watershed splitting helps with touching blobs but not with true occlusion.

## Spatial calibration

`cfg.PIXELS_PER_MM` is a *derived* module-level constant in `src/config.py`. It is computed once at import time from four primary inputs:

- `FRAME_WIDTH_PX = 1280`, `FRAME_HEIGHT_PX = 1024` — source video dimensions.
- `ROI_TOP`, `ROI_BOTTOM`, `ROI_LEFT`, `ROI_RIGHT` — pixel margins excluded from detection; together they define the active ROI, which coincides with the inside of the physical tank.
- `TANK_WIDTH_MM = 165.0`, `TANK_HEIGHT_MM = 145.0` — known tank inside dimensions in millimetres.

The derivation:

```
roi_width_px  = FRAME_WIDTH_PX  - ROI_LEFT - ROI_RIGHT     # = 1090
roi_height_px = FRAME_HEIGHT_PX - ROI_TOP  - ROI_BOTTOM    # = 891
PIXELS_PER_MM = mean(roi_width_px / TANK_WIDTH_MM,
                    roi_height_px / TANK_HEIGHT_MM)        # ≈ 6.38
```

For the supplied recording the horizontal ratio is `1090 / 165 ≈ 6.61` and the vertical is `891 / 145 ≈ 6.15`; the mean is `≈ 6.38 px/mm`.

**To recalibrate for a different recording:** update any of the four ROI margins, the two frame-dimension constants, or the two tank-dimension constants. Never derive the scale from animal body length.

`compute_population_summary` in `src/analysis.py` reads `cfg.PIXELS_PER_MM` directly and emits the size and speed metrics in both units. The population summary CSV / stdout therefore contains both `Major axis (px)` and `Major axis (mm)`, both `Speed (px/s)` and `Speed (mm/s)`, and both `SD of Speed (px/s)` and `SD of Speed (mm/s)`. `FwdRun` and the three `YFraction*` keys are unitless and have no mm counterpart.

Coordinate and unit conventions:

- Frame: 1280 × 1024 px (default; configurable via `FRAME_*_PX`).
- `cv2.fitEllipse` returns axes in pixels.
- `cv2.moments` centroids are in pixels from the top-left of the frame (x rightward, y downward).
