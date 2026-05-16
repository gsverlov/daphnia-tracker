from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
VIDEO_PATH: Path = PROJECT_ROOT / "data" / "image_3.mp4"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# ROI — pixel margins to exclude from detection.
# The corresponding rows/columns are zeroed in the foreground mask before
# contour detection. Bubble column goes at top; debris at bottom.
# ---------------------------------------------------------------------------
ROI_TOP: int = 55     # TUNED IN PHASE A — rows to exclude at top (bubble column)
ROI_BOTTOM: int = 78  # TUNED IN PHASE A — rows to exclude at bottom (debris)
ROI_LEFT: int = 90    # TUNED IN PHASE A — columns to exclude at left
ROI_RIGHT: int = 100 # TUNED IN PHASE A — columns to exclude at right

# ---------------------------------------------------------------------------
# Spatial calibration — px-to-mm scale derived from the ROI and the known
# tank dimensions. The ROI corresponds to the inside of the tank, so the
# scale is the average of the horizontal and vertical px/mm ratios.
# Update TANK_WIDTH_MM / TANK_HEIGHT_MM and FRAME_WIDTH_PX / FRAME_HEIGHT_PX
# (and the ROI margins above) if a different recording is used.
# ---------------------------------------------------------------------------
FRAME_WIDTH_PX: int = 1280   # source video frame width
FRAME_HEIGHT_PX: int = 1024  # source video frame height
TANK_WIDTH_MM: float = 165.0   # paper: 16.5 cm swimming arena width
TANK_HEIGHT_MM: float = 145.0  # paper: 14.5 cm swimming arena height
# PIXELS_PER_MM: derived from ROI dimensions and tank dimensions.
#   ROI width  = FRAME_WIDTH_PX  - ROI_LEFT - ROI_RIGHT
#   ROI height = FRAME_HEIGHT_PX - ROI_TOP  - ROI_BOTTOM
#   PIXELS_PER_MM = mean(roi_width / TANK_WIDTH_MM, roi_height / TANK_HEIGHT_MM)
_ROI_WIDTH_PX: int = FRAME_WIDTH_PX - ROI_LEFT - ROI_RIGHT
_ROI_HEIGHT_PX: int = FRAME_HEIGHT_PX - ROI_TOP - ROI_BOTTOM
PIXELS_PER_MM: float = (
    (_ROI_WIDTH_PX / TANK_WIDTH_MM) + (_ROI_HEIGHT_PX / TANK_HEIGHT_MM)
) / 2.0

# ---------------------------------------------------------------------------
# Background subtraction (MOG2)
# ---------------------------------------------------------------------------
MOG2_HISTORY: int = 500            # TUNED IN PHASE A — frame history for background model
MOG2_VAR_THRESHOLD: float = 25.0   # TUNED IN PHASE A — Mahalanobis distance threshold
MOG2_DETECT_SHADOWS: bool = False  # shadows disabled — grayscale video, shadow detection not useful
MOG2_WARMUP_FRAMES: int = 60       # TUNED IN PHASE A — frames fed before detection starts

# ---------------------------------------------------------------------------
# Morphology (applied to foreground mask to suppress noise pixels)
# ---------------------------------------------------------------------------
MORPH_KERNEL_SIZE: int = 3       # TUNED IN PHASE A — diameter of elliptical structuring element
MORPH_OPEN_ITERATIONS: int = 1   # TUNED IN PHASE A — iterations of morphological opening

# ---------------------------------------------------------------------------
# Blob filtering (areas in pixels²)
# ---------------------------------------------------------------------------
MIN_AREA: float = 5.0    # TUNED IN PHASE A — contours smaller than this are noise
MAX_AREA: float = 2000.0  # TUNED IN PHASE A — contours larger than this are clumps or debris

# Contours with area >= this threshold are attempted to be split into multiple
# sub-detections via distance-transform + watershed (likely a merged clump of
# touching Daphnia). Smaller contours are treated as a single Daphnia.
WATERSHED_SPLIT_THRESHOLD_PX2: float = 80.0

# Degenerate-ellipse filter — applied after cv2.fitEllipse, before the shadow
# filter. Real Daphnia in a 1280×1024 frame don't approach these limits; a fit
# that does is almost certainly a noisy or merged-blob contour.
MAX_MAJOR_AXIS_PX: float = 100.0  # ceiling on fitted ellipse major axis (px)
MAX_ASPECT_RATIO: float = 10.0    # ceiling on major/minor axis ratio
# ---------------------------------------------------------------------------
# Shadow filter — local-contrast rejection in left zone.
# Real Daphnia are much darker than their surroundings (strongly negative
# contrast). Shadows on the dark left wall are only slightly brighter than
# the wall (contrast near zero or positive). The filter rejects detections
# in the left zone where contrast > CONTRAST_MARGIN.
# Tuned to -20 to catch persistent top-left wall shadows while preserving
# real Daphnia near the wall.
# ---------------------------------------------------------------------------
SHADOW_FILTER_ENABLED: bool = True
SHADOW_FILTER_ZONE_WIDTH: int = 132  # filter applies where x < this
# Right-side zone width. The same ring-contrast test is applied where
# x > frame_width - SHADOW_FILTER_ZONE_WIDTH_RIGHT to catch persistent
# right-wall shadows. Uses the same margin/ring params as the left zone.
SHADOW_FILTER_ZONE_WIDTH_RIGHT: int = 118
SHADOW_FILTER_MARGIN: float = -20.0  # reject if contour-ring contrast > this
SHADOW_FILTER_RING_GAP_PX: int = 1   # pixel gap between contour and ring start
SHADOW_FILTER_RING_THICKNESS_PX: int = 5  # ring band thickness
# ---------------------------------------------------------------------------
# Tracker — placeholder values, set during Phase C
# ---------------------------------------------------------------------------
TRACK_MAX_AGE: int = 15            # TUNED IN PHASE C — frames a track survives without a match
TRACK_MIN_HITS: int = 3            # TUNED IN PHASE C — detections needed to confirm a track
TRACK_MAX_DISTANCE: float = 30.0   # TUNED IN PHASE C — max centroid distance (px) for assignment

# Extra gate on the association step: reject a (track, detection) pair when the
# detection is more than MAX_COAST_DISTANCE_PX from the track's *last measured*
# position (the last frame the track was matched to a real detection, not the
# coasted Kalman prediction). Prevents a long-coasting track from being
# reattached to a faraway detection just because its prediction drifted close.
MAX_COAST_DISTANCE_PX: float = 30.0

# Size-aware association: per-pixel cost added for each pixel of major-axis
# difference between a track's running size estimate and a candidate
# detection's major_axis_px. Pure-distance behaviour returns at 0.0. The
# weight tunes how aggressively the tracker prefers same-size matches over
# nearer-but-different-size ones; it is NOT a gate — TRACK_MAX_DISTANCE /
# MAX_COAST_DISTANCE_PX still gate on geometric distance only.
SIZE_COST_WEIGHT: float = 1.0

# Kalman filter noise covariances for the [x, y, vx, vy] constant-velocity
# model. R is the measurement noise on the detected centroid; Q is the
# per-step process noise driving the filter's responsiveness.
KALMAN_R_VARIANCE: float = 4.0       # centroid measurement variance (px²), std ≈ 2 px
KALMAN_Q_POS_VARIANCE: float = 0.5   # process noise variance on x, y (px²/frame)
KALMAN_Q_VEL_VARIANCE: float = 2.0   # process noise variance on vx, vy ((px/frame)²)

# ---------------------------------------------------------------------------
# Analysis thresholds — placeholder values, review after Phase D outputs exist.
# Reference: at ~20 fps, max Daphnia speed ~10 px/frame ≈ 200 px/s.
# ---------------------------------------------------------------------------
MIN_TRACKLET_FRAMES_FOR_SUMMARY: int = 5  # TUNED IN PHASE D — tracklets shorter than this excluded from population stats

# ---------------------------------------------------------------------------
# Cho et al. 2022 behavioural features — set in Phase D.5
# ---------------------------------------------------------------------------
FWDRUN_THRESHOLD_MULTIPLIER: float = 1.5     # TUNED IN PHASE D.5 — FwdRun: speed > N×population mean speed
MIN_DISPLACEMENT_FOR_HEADING_PX: float = 1.0  # TUNED IN PHASE D.5 — min step length (px) counted for heading
