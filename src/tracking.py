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

# Q and R are first-pass placeholders — will be tuned once Phase C is
# functional and we can inspect track smoothness on real data.
# At that point they move into src/config.py.
_Q_DIAG = [1.0, 1.0, 1.0, 1.0]
_R_DIAG = [1.0, 1.0]


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
        self.kf.Q = np.diag(_Q_DIAG)
        self.kf.R = np.diag(_R_DIAG)

        self.kf.x = np.array(
            [[detection["x"]], [detection["y"]], [0.0], [0.0]]
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

        Resets time_since_update to 0 and increments hits.
        """
        self.kf.update(np.array([detection["x"], detection["y"]]))
        self.time_since_update = 0
        self.hits += 1

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
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimally match detections to predicted track positions by centroid distance.

    Builds an (n_tracks × n_detections) cost matrix of Euclidean distances,
    solves the linear assignment problem, then rejects any pair whose distance
    exceeds max_distance. Both members of a rejected pair are returned as
    unmatched.

    Args:
        detections: Detection dicts from detect_frame, each with "x" and "y".
        predicted_positions: Per-track [x, y] predictions from KalmanTrack.predict().
        max_distance: Distance threshold in pixels; pairs above this are rejected.

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

    cost = np.zeros((n_tracks, n_dets), dtype=float)
    for i, pos in enumerate(predicted_positions):
        for j, det in enumerate(detections):
            dx = pos[0] - det["x"]
            dy = pos[1] - det["y"]
            cost[i, j] = np.hypot(dx, dy)

    row_ind, col_ind = linear_sum_assignment(cost)

    unmatched_tracks = set(range(n_tracks))
    unmatched_dets = set(range(n_dets))
    matches: list[tuple[int, int]] = []

    for t_idx, d_idx in zip(row_ind, col_ind):
        if cost[t_idx, d_idx] <= max_distance:
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

        # 2. Match detections to predicted positions.
        matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
            detections, predicted, cfg.TRACK_MAX_DISTANCE
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

    frame_to_dets: dict[int, list[dict]] = {
        int(frame_idx): group[["x", "y"]].to_dict("records")
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
