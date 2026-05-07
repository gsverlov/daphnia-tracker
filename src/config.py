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
ROI_LEFT: int = 107    # TUNED IN PHASE A — columns to exclude at left
ROI_RIGHT: int = 107  # TUNED IN PHASE A — columns to exclude at right

# ---------------------------------------------------------------------------
# Background subtraction (MOG2)
# ---------------------------------------------------------------------------
MOG2_HISTORY: int = 500            # TUNED IN PHASE A — frame history for background model
MOG2_VAR_THRESHOLD: float = 16.0   # TUNED IN PHASE A — Mahalanobis distance threshold
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
MAX_AREA: float = 300.0  # TUNED IN PHASE A — contours larger than this are clumps or debris

# ---------------------------------------------------------------------------
# Tracker — placeholder values, set during Phase C
# ---------------------------------------------------------------------------
TRACK_MAX_AGE: int = 5             # TUNED IN PHASE C — frames a track survives without a match
TRACK_MIN_HITS: int = 3            # TUNED IN PHASE C — detections needed to confirm a track
TRACK_MAX_DISTANCE: float = 30.0   # TUNED IN PHASE C — max centroid distance (px) for assignment
