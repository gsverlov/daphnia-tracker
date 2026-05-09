# Fish Tracker — CLAUDE.md

## Project Goal

Build a classical computer vision pipeline to track individual fish in a top-down aquarium video. Input: `data/image_3.mp4` (~20s, grayscale, ~99 small dark fish on a light background, static camera). Output: `output/tracks.csv` with per-fish trajectories `(frame, track_id, x, y)`, an annotated video, side-by-side comparison frames, and downstream analysis (speed, heatmaps, schooling metrics). No deep learning — background subtraction (MOG2) + blob detection + SORT-style Kalman tracker throughout.

## Current Status

| Phase | Status | Output |
|---|---|---|
| A — Exploration notebook + `src/config.py` | **done** | `notebooks/01_explore.ipynb`, `src/config.py` |
| B — Detection module | **done** | `src/detection.py`, `output/detections.csv` |
| C — Tracking module | **done** | `src/tracking.py`, `output/tracks.csv` |
| D — Analysis module | todo | `src/analysis.py`, `output/analysis_summary.csv` |
| E — Visualization | todo | `src/visualization.py`, `output/annotated.mp4`, `output/comparison_frame_*.png`, plots |
| Pipeline CLI | todo | `src/pipeline.py` |

**Phase A notes:** `src/config.py` tuned: `ROI_TOP=55`, `ROI_BOTTOM=75`, `ROI_LEFT=70`, `ROI_RIGHT=70`, `MIN_AREA=5`, `MAX_AREA=300`. MOG2 and morphology defaults retained. Expected ~85–105 detections/frame on representative frames — intentionally below the true fish count of ~99 because `MAX_AREA=300` rejects 2–3 fish clumps during dense moments. This is a deliberate tradeoff: each accepted detection corresponds to one fish, giving valid per-fish speed measurements. The Phase A success check (notebook cell `In[9]`) passes when counts are in the 80–110 range.

**Phase B notes:** `src/detection.py` exposes `build_background_subtractor`, `build_roi_mask`, `detect_frame`, `run_detection`. CLI: `python -m src.detection`. Output: ~36,000 detections on `image_3.mp4`, ~106 mean per frame, runtime ~6s.

**Phase C notes:** `src/tracking.py` exposes `KalmanTrack`, `associate_detections_to_tracks`, `Tracker`, `run_tracking`. SORT-style with constant-velocity Kalman filter, Hungarian assignment by centroid distance. CLI: `python -m src.tracking`. Output: ~42,000 track-frame rows, ~948 unique track IDs (ID switching expected per project framing), median tracklet length 22 frames, runtime ~8s.

## Tech Stack

| Library | Role |
|---|---|
| Python 3.12 | Runtime (Windows, `.venv`) |
| `opencv-python` | Background subtraction (MOG2), blob detection, video I/O, frame annotation |
| `filterpy` | Kalman filter per track (constant-velocity model) |
| `scipy` | Hungarian assignment (`linear_sum_assignment`) for detection–track matching |
| `numpy` | Array ops throughout |
| `pandas` | Detection/track DataFrames, CSV I/O |
| `matplotlib` / `seaborn` | Heatmaps, speed distributions, analysis plots |
| `tqdm` | Progress bars on long video loops |
| `jupyter` / `ipykernel` | Exploration notebooks |

## Project Layout

```
fish-tracker/
├── data/
│   └── image_3.mp4              # Input video (gitignored, ~32 MB)
├── notebooks/
│   └── 01_explore.ipynb         # Phase A: parameter tuning and exploration
├── output/                      # All generated files (gitignored)
│   ├── detections.csv           # Per-frame blob detections (frame, x, y, w, h)
│   ├── tracks.csv               # Per-fish trajectories (frame, track_id, x, y)
│   ├── analysis_summary.csv     # Per-track metrics (speed, duration, etc.)
│   ├── annotated.mp4            # Video with boxes, IDs, trails
│   ├── comparison_frame_*.png   # Side-by-side original vs annotated (sanity check)
│   ├── heatmap.png
│   └── speed_hist.png
├── src/
│   ├── __init__.py
│   ├── config.py                # All tuned parameters — single source of truth
│   ├── detection.py             # Phase B: background model + blob detector
│   ├── tracking.py              # Phase C: Kalman tracker + Hungarian assignment
│   ├── analysis.py              # Phase D: trajectory metrics and schooling
│   ├── visualization.py         # Phase E: video annotation and plots
│   └── pipeline.py              # CLI: runs full pipeline end-to-end
├── requirements.txt
├── .gitignore
├── README.md
└── CLAUDE.md
```

## Data

- **File:** `data/image_3.mp4` — ~20s clip, grayscale, static camera
- **Scene:** ~99 small dark fish on light background; stationary bubble column at top of frame; debris/particles at bottom
- **Coordinate convention:** `(x, y)` is the bounding-box centroid in pixels from the top-left corner of the frame. `w`, `h` are bounding-box width and height.
- **Units:** Speed and distance are in px and px/frame by default. `src/analysis.py` functions accept `pixels_per_mm: float | None = None`; when provided, additional `_mm` and `_mm_s` columns are emitted.
- **Frame dimensions:** Always read from the video at runtime (`cap.get(cv2.CAP_PROP_FRAME_WIDTH)` etc.). Never hardcode.

## Analysis Requirements

- **Per-track statistics are required output, not just aggregate.** The pipeline produces `output/track_summary.csv` with one row per track: `track_id`, `frame_start`, `frame_end`, `duration_frames`, `mean_speed_px_per_frame`, `max_speed_px_per_frame`, `total_distance_px`. Speed columns also emit `_mm` variants when `pixels_per_mm` is provided.
- **Single-track visualization is required.** `annotate_video()` in `src/visualization.py` accepts an optional `track_id` parameter; when provided, only that track's bounding box and trail are drawn.
- **"Following one fish" means following one track.** A track may correspond to multiple physical fish if ID swaps occur. Speed and distance metrics from a track are still valid — they reflect actual swimming motion regardless of which physical fish produced them. Long-range geographic paths may have discontinuities at ID-swap moments; this is expected and acceptable.
- **Interactive "click-to-follow" viewer** is a Phase F nice-to-have, not blocking.

## Running the Pipeline

```powershell
# Activate venv
.venv\Scripts\activate

# Full pipeline (detection → tracking → analysis → visualization)
python -m src.pipeline data/image_3.mp4

# Individual steps
python -m src.detection data/image_3.mp4      # → output/detections.csv
python -m src.tracking                         # → output/tracks.csv
python -m src.analysis                         # → output/analysis_summary.csv
python -m src.visualization data/image_3.mp4  # → output/annotated.mp4, plots

# Exploration notebook
jupyter notebook notebooks/01_explore.ipynb
```

## Conventions

- **Naming:** snake_case everywhere (files, functions, variables).
- **Parameters:** All tuned values live in `src/config.py`. Detection/tracking/analysis functions import from config at module level. Functions may accept override kwargs for testing, but production calls use config values. No parameter defaults hardcoded inside function signatures.
- **Data structures:** Functions return plain Python dicts or pandas DataFrames — no custom data classes for intermediate data.
- **Canonical CSV columns:** `frame` (int, 0-indexed), `track_id` (int), `x` (float), `y` (float), `w` (float), `h` (float). Track CSV drops `w`/`h`.
- **Output directory:** All generated files go to `output/`. Never write results next to source files.
- **SKILL.md files:** Sub-areas with their own conventions may have a `SKILL.md` inside their directory. Read it before working in that area.

## Gotchas

- **Bubbles at top:** The bubble column is a persistent false-positive source. Excluded with a hard ROI mask: the top `ROI_TOP` rows are zeroed before running the blob detector. Value determined in Phase A and stored in `src/config.py`.
- **Debris at bottom:** Excluded via `ROI_BOTTOM` rows and `MIN_AREA` threshold. Both in `src/config.py`.
- **MAX_AREA = 300 deliberately rejects 2–3 fish clumps.** Dense regions are slightly underrepresented in any single frame's detection count, but each accepted detection represents one fish, giving valid per-fish speed measurements. Raising MAX_AREA would catch more fish but contaminate speed data with clump-centroid motion. Revisit this tradeoff during analysis if it causes track fragmentation issues.
- **Windows video writer codec:** `cv2.VideoWriter` with `mp4v` may fail silently on some Windows installs. Check `writer.isOpened()` after construction; fall back to `XVID` codec with `.avi` extension if needed.
- **filterpy noise tuning:** Kalman Q (process noise) and R (measurement noise) matrices are in `src/config.py`. Too-high Q → jittery predictions. Too-low Q → slow response to direction changes.
- **Fish crossing ID switches:** SORT cannot re-identify fish after full occlusion. Some ID switches at crossings are expected and acceptable — this is a known SORT limitation.
- **Centroid distance (not IoU):** The tracker uses centroid distance as the cost metric. Fish are small and IoU degrades on near-misses where bounding boxes barely overlap.
- **ROI values must be non-zero.** If `src/config.py` ROI values are ever corrupted to 0, detection passes through the full frame including the bubble band and gravel zone, polluting tracking output. Verify with: `python -c "from src import config as cfg; print(cfg.ROI_TOP, cfg.ROI_BOTTOM, cfg.ROI_LEFT, cfg.ROI_RIGHT)"`
- **`src/visualize_tracks.py` is a pre-Phase-E inspection tool**, not the deliverable. It produces single-frame PNGs and full annotated video for tracker quality checks (`python -m src.visualize_tracks` or `--frame N`). Phase E proper (`src/visualization.py`) will produce the final comparison frames and analysis plots.

## Anti-patterns to avoid

- **Don't silently change tuned parameters.** Values in `src/config.py` are the result of deliberate tuning. If a parameter needs to change, flag it explicitly in the message to the user, with reasoning. Never edit a config value as part of an unrelated change.
- **Don't over-engineer.** Default to plain functions. Don't introduce classes unless there's clear state to encapsulate. No abstract base classes, no inheritance hierarchies, no factory patterns. If you're tempted to add abstraction "for flexibility," resist — add it when the second use case actually appears.
- **No hidden state.** Functions should be pure where possible. No mutable default arguments (e.g., `def f(x=[])`). No module-level globals that change at runtime. If a function's behavior depends on call order, that's a bug.
- **No premature optimization.** Make code correct and readable first. Don't vectorize, parallelize, or micro-optimize unless profiling shows a real bottleneck. A 2-minute runtime on the full video is acceptable.
- **Always read parameters from `src/config.py`.** Detection, tracking, and analysis modules import their parameters from config. Function signatures may accept overrides for testing, but production calls use config. Don't shadow config values with function default arguments.
- **Visualization is mandatory, not optional.** When making a change to detection, tracking, or analysis, produce an image or annotated frame demonstrating the change works on real data. "Trust me, this fix works" is not acceptable — show the before/after on a representative frame.
- **Explain reasoning for non-obvious choices.** Don't just say "this is more robust" or "this should work better." Say *why*: what failure mode it fixes, what tradeoff it makes, what evidence supports it.
- **Paste errors verbatim.** When something fails, include the actual error message and traceback. Don't paraphrase or summarize.
- **Flag drift.** If you notice you've stopped following a CLAUDE.md convention earlier in the session, call it out and correct it rather than silently continuing.

## Out of Scope

- Deep learning (YOLO, re-ID networks, DeepSORT)
- Real-time / low-latency processing
- Multi-camera setups
- Pixel → millimeter calibration (optional arg exists in `analysis.py`, no default values provided)
- 3D position estimation

## Working Style

- **Approach:** Classical CV. No ML unless explicitly decided.
- **Abstraction:** Don't introduce abstractions beyond what the current phase requires.
- **Notebooks:** The user runs notebook cells manually. Claude does not execute notebooks end-to-end or assume cells have been run.
- **Commits:** The user reviews and commits. Claude does not auto-commit.
- **Design changes:** Ask before making non-trivial design decisions mid-implementation.
- **Errors:** When reporting errors, paste the actual error message verbatim — don't paraphrase.
- **Visual regression:** Notebooks serve as the visual regression check during development. Add `tests/` only if specific logic (Kalman update, Hungarian edge cases, track lifecycle) becomes complex enough to warrant unit tests. Don't preemptively scaffold pytest.
- **SKILL.md:** If a `SKILL.md` file exists in a `src/` subdirectory, read it before working in that area.
