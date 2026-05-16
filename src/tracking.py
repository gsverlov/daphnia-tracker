from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src import config as cfg

# Q (process noise) and R (measurement noise) are sourced from src/config.py.
# R variance reflects ~2 px centroid jitter from contour fitting on small
# Daphnia. Q splits position vs. velocity variance: position noise is small
# (centroids barely drift between frames absent real motion), velocity noise
# is larger because Daphnia change direction frequently.


class KalmanTrack:
    """Single tracked object managed by a 2D constant-velocity Kalman filter.

    State vector : [x, y, vx, vy]  (position + velocity)
    Measurement  : [x, y]           (centroid only)

    A new track is created with hits=1 and age=1, counting the spawning
    detection as the first hit. The initial state is set directly from
    that detection; no predict/update cycle is needed for frame 0.
    """

    def __init__(self, track_id: int, detection: dict) -> None:
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.F = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=float,
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=float,
        )
        # High velocity uncertainty on init — we don't know vx/vy yet.
        self.kf.P = np.diag([10.0, 10.0, 1000.0, 1000.0])
        self.kf.Q = np.diag([
            cfg.KALMAN_Q_POS_VARIANCE,
            cfg.KALMAN_Q_POS_VARIANCE,
            cfg.KALMAN_Q_VEL_VARIANCE,
            cfg.KALMAN_Q_VEL_VARIANCE,
        ])
        self.kf.R = np.diag([cfg.KALMAN_R_VARIANCE, cfg.KALMAN_R_VARIANCE])

        self.kf.x = np.array(
            [[detection["x"]], [detection["y"]], [0.0], [0.0]]
        )

        # Position from the most recent frame this track was matched to a real
        # detection. Used as a second association gate so a long-coasting track
        # can't snap onto a faraway detection just because its Kalman
        # prediction drifted close.
        self.last_measured_x: float = float(detection["x"])
        self.last_measured_y: float = float(detection["y"])

        # Running estimate of this track's major-axis length, used as a soft
        # secondary cue in association. Initialized from the spawning
        # detection (may be NaN if that contour was too small for fitEllipse)
        # and updated with EMA alpha=0.3 on every matched detection. Not
        # updated on coast frames — there is no measurement to learn from.
        self.size_estimate: float = float(
            detection.get("major_axis_px", float("nan"))
        )

    def predict(self) -> np.ndarray:
        """Advance the Kalman filter one step and return predicted [x, y].

        Increments both age and time_since_update. Called once per frame
        on every active track before association.
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0]])

    def update(self, detection: dict) -> None:
        """Incorporate a matched detection into the filter state.

        Resets time_since_update to 0 and increments hits. Also folds the
        detection's major_axis_px into the running size estimate using an
        EMA with alpha=0.3. If the detection has NaN major_axis_px (contour
        too small for fitEllipse) the size estimate is left unchanged. If
        the prior estimate is NaN but the detection has a valid size, the
        detection size initializes the estimate.
        """
        self.kf.update(np.array([detection["x"], detection["y"]]))
        self.time_since_update = 0
        self.hits += 1
        self.last_measured_x = float(detection["x"])
        self.last_measured_y = float(detection["y"])

        det_size = float(detection.get("major_axis_px", float("nan")))
        if not np.isnan(det_size):
            if np.isnan(self.size_estimate):
                self.size_estimate = det_size
            else:
                self.size_estimate = 0.3 * det_size + 0.7 * self.size_estimate

    def get_state(self) -> dict:
        """Return the current filter estimate as a plain dict."""
        return {
            "x": float(self.kf.x[0, 0]),
            "y": float(self.kf.x[1, 0]),
            "vx": float(self.kf.x[2, 0]),
            "vy": float(self.kf.x[3, 0]),
        }


def associate_detections_to_tracks(
    detections: list[dict],
    predicted_positions: list[np.ndarray],
    max_distance: float,
    last_measured_positions: list[np.ndarray] | None = None,
    max_coast_distance: float | None = None,
    track_sizes: list[float] | None = None,
    size_cost_weight: float = 0.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimally match detections to predicted track positions.

    The assignment cost is Euclidean distance from each track's predicted
    position to each detection's centroid, optionally augmented by a
    size-difference term:

        total_cost[i, j] = dist[i, j]
                         + |track_size[i] - det_size[j]| * size_cost_weight

    The size term is skipped (treated as 0) when either side is NaN, so
    detections with no valid ellipse fit fall back to distance-only matching.
    Hungarian assignment is run on this augmented cost, but the final gates
    (max_distance, max_coast_distance) are checked against the *geometric*
    distance only — size is a tiebreaker among candidates inside the gate,
    not a gate itself.

    Args:
        detections: Detection dicts from detect_frame, each with "x", "y",
            and optionally "major_axis_px".
        predicted_positions: Per-track [x, y] predictions from KalmanTrack.predict().
        max_distance: Distance threshold in pixels; pairs whose geometric
            distance exceeds this are rejected.
        last_measured_positions: Per-track [x, y] last-measured positions,
            used by the optional coast-distance gate.
        max_coast_distance: When both last_measured_positions and this are
            provided, pairs whose detection is farther than this from the
            track's last *measured* position are rejected.
        track_sizes: Per-track running major-axis estimates, NaN-permitted.
            When None or all-NaN the size term is disabled.
        size_cost_weight: Cost added per pixel of size difference. 0 disables
            the size term entirely.

    Returns:
        matches: List of (track_idx, detection_idx) accepted pairs.
        unmatched_detection_indices: Detection indices with no accepted match.
        unmatched_track_indices: Track indices with no accepted match.
    """
    n_tracks = len(predicted_positions)
    n_dets = len(detections)

    if n_tracks == 0:
        return [], list(range(n_dets)), []
    if n_dets == 0:
        return [], [], list(range(n_tracks))

    dist = np.zeros((n_tracks, n_dets), dtype=float)
    for i, pos in enumerate(predicted_positions):
        for j, det in enumerate(detections):
            dx = pos[0] - det["x"]
            dy = pos[1] - det["y"]
            dist[i, j] = np.hypot(dx, dy)

    cost = dist.copy()
    if track_sizes is not None and size_cost_weight > 0.0:
        for i, ts in enumerate(track_sizes):
            if np.isnan(ts):
                continue
            for j, det in enumerate(detections):
                ds = det.get("major_axis_px", float("nan"))
                if np.isnan(ds):
                    continue
                cost[i, j] += abs(ts - ds) * size_cost_weight

    row_ind, col_ind = linear_sum_assignment(cost)

    unmatched_tracks = set(range(n_tracks))
    unmatched_dets = set(range(n_dets))
    matches: list[tuple[int, int]] = []

    coast_gate_enabled = (
        last_measured_positions is not None and max_coast_distance is not None
    )

    for t_idx, d_idx in zip(row_ind, col_ind):
        if dist[t_idx, d_idx] > max_distance:
            continue
        if coast_gate_enabled:
            lm = last_measured_positions[t_idx]
            det = detections[d_idx]
            coast_dist = np.hypot(lm[0] - det["x"], lm[1] - det["y"])
            if coast_dist > max_coast_distance:
                continue
        matches.append((int(t_idx), int(d_idx)))
        unmatched_tracks.discard(t_idx)
        unmatched_dets.discard(d_idx)

    return matches, sorted(unmatched_dets), sorted(unmatched_tracks)


class Tracker:
    """SORT-style multi-object tracker for fish centroids.

    Maintains a pool of KalmanTracks. Each call to update() advances the
    tracker by one frame: predict all tracks, match detections, update
    matched tracks, spawn new tracks, kill stale tracks, return confirmed
    tracks.

    Track IDs are unique integers assigned in order from 0; they are never
    reused within a session.
    """

    def __init__(self) -> None:
        self.tracks: list[KalmanTrack] = []
        self._next_id: int = 0

    def _allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, detections: list[dict]) -> list[dict]:
        """Advance the tracker by one frame.

        Args:
            detections: List of detection dicts for the current frame.
                Empty list is valid (no fish detected this frame).

        Returns:
            List of confirmed active track dicts with keys:
            track_id, x, y, vx, vy.
            A track is confirmed if hits >= cfg.TRACK_MIN_HITS (enough
            evidence) OR age <= cfg.TRACK_MIN_HITS (still in its initial
            window — keeps new fish visible from frame 1 of their track).
        """
        # 1. Predict forward all existing tracks.
        predicted = [t.predict() for t in self.tracks]
        last_measured = [
            np.array([t.last_measured_x, t.last_measured_y]) for t in self.tracks
        ]
        track_sizes = [t.size_estimate for t in self.tracks]

        # 2. Match detections to predicted positions. The coast-distance gate
        # additionally rejects pairs where the detection is far from the
        # track's last *measured* position (not the coasted prediction). The
        # size term in the assignment cost biases matching toward
        # similarly-sized candidates within the distance gate; it does not
        # change which pairs the gate accepts.
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            detections,
            predicted,
            cfg.TRACK_MAX_DISTANCE,
            last_measured_positions=last_measured,
            max_coast_distance=cfg.MAX_COAST_DISTANCE_PX,
            track_sizes=track_sizes,
            size_cost_weight=cfg.SIZE_COST_WEIGHT,
        )

        # 3. Update matched tracks with their paired detection.
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])

        # 4. Spawn new tracks from unmatched detections.
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanTrack(self._allocate_id(), detections[d_idx]))

        # 5. Kill tracks that have gone too long without a match.
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= cfg.TRACK_MAX_AGE
        ]

        # 6. Return confirmed tracks from what remains.
        return [
            {"track_id": t.track_id, **t.get_state()}
            for t in self.tracks
            if t.hits >= cfg.TRACK_MIN_HITS or t.age <= cfg.TRACK_MIN_HITS
        ]


def run_tracking(detections_df: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
    """Link per-frame detections into tracklets using the SORT tracker.

    Iterates every frame index between the first and last frame in
    detections_df — including frames with no detections — so that track
    age and time_since_update advance correctly on frames where nothing
    was detected. Skipping those frames would make tracks appear to live
    longer than they should.

    Args:
        detections_df: DataFrame from run_detection with at minimum columns
            frame, x, y. May also contain w, h, area (ignored here).
        progress: Show a tqdm progress bar.

    Returns:
        DataFrame with columns: frame, track_id, x, y, vx, vy.
    """
    if detections_df.empty:
        return pd.DataFrame(columns=["frame", "track_id", "x", "y", "vx", "vy"])

    # Major-axis is included when available so the size-aware association
    # has something to work with. Older detection CSVs without an ellipse
    # column fall back to NaN, which disables the size term per pair.
    det_cols = ["x", "y"]
    if "major_axis_px" in detections_df.columns:
        det_cols.append("major_axis_px")

    frame_to_dets: dict[int, list[dict]] = {
        int(frame_idx): group[det_cols].to_dict("records")
        for frame_idx, group in detections_df.groupby("frame")
    }

    min_frame = int(detections_df["frame"].min())
    max_frame = int(detections_df["frame"].max())

    tracker = Tracker()
    rows: list[dict] = []

    frame_iter: range | tqdm = range(min_frame, max_frame + 1)
    if progress:
        frame_iter = tqdm(frame_iter, desc="Tracking", unit="frame")

    for frame_idx in frame_iter:
        dets = frame_to_dets.get(frame_idx, [])
        for t in tracker.update(dets):
            rows.append({"frame": frame_idx, **t})

    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "vx", "vy"])


if __name__ == "__main__":
    default_det = cfg.OUTPUT_DIR / "image_3_detections.csv"
    det_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_det

    detections_df = pd.read_csv(det_path)

    t0 = time.perf_counter()
    tracks_df = run_tracking(detections_df)
    elapsed = time.perf_counter() - t0

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = det_path.stem.replace("_detections", "")
    out_path = cfg.OUTPUT_DIR / f"{stem}_tracks.csv"
    tracks_df.to_csv(out_path, index=False)

    if not tracks_df.empty:
        lengths = tracks_df.groupby("track_id")["frame"].count()
        n_tracks = len(lengths)
        mean_len = lengths.mean()
        median_len = lengths.median()
        max_len = int(lengths.max())
        long_tracks = int((lengths >= cfg.TRACK_MIN_HITS).sum())
    else:
        n_tracks = mean_len = median_len = max_len = long_tracks = 0

    print(f"Detections CSV           : {det_path}")
    print(f"Output                   : {out_path}")
    print(f"Track-frame rows         : {len(tracks_df):,}")
    print(f"Unique track IDs         : {n_tracks}")
    print(f"Tracklet length          : mean={mean_len:.1f}  median={median_len:.1f}  max={max_len}")
    print(f"Tracks >= {cfg.TRACK_MIN_HITS} frames      : {long_tracks}")
    print(f"Runtime                  : {elapsed:.1f}s")
