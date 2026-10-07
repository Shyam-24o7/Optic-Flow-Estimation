"""Multi-object tracker whose Kalman state carries the log-scale of each box.

State x = [cx, cy, s, a, vx, vy, vs] with s = ln sqrt(w*h) and a = w/h.
vs = ds/dt is the looming rate: for an object approaching at constant speed,
vs = 1 / TTC. Measurements are [cx, cy, s, a] from the detector.

Each track predicts from its own state, which fixes the wrong-track lookup in
hls/ports/object_tracker.hpp.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..tracking import Detection, iou

HISTORY = 9  # boxes kept per track: current + 8 past frames (largest scale-search gap)


@dataclass(frozen=True)
class TrackerConfig:
    iou_threshold: float = 0.3
    max_lost: int = 30
    accel_px_s2: float = 200.0      # process noise: centre acceleration
    scale_accel_s2: float = 1.0     # process noise: change of looming rate
    aspect_walk: float = 0.1        # process noise: aspect random walk per sqrt(s)
    meas_center_px: float = 2.0
    meas_scale: float = 0.02
    meas_aspect: float = 0.05


def box_to_z(box) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, 0.5 * np.log(w * h), w / h])


def z_to_box(z) -> np.ndarray:
    cx, cy, s, a = z[:4]
    area_root = np.exp(s)
    w, h = area_root * np.sqrt(a), area_root / np.sqrt(a)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class ScaleTrack:
    id: int
    x: np.ndarray
    P: np.ndarray
    class_name: str
    lost: int = 0
    age: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))  # (frame_index, box)

    @property
    def bbox(self) -> np.ndarray:
        return z_to_box(self.x)

    @property
    def scale_rate(self) -> float:
        return float(self.x[6])

    @property
    def scale_rate_var(self) -> float:
        return float(self.P[6, 6])

    def box_at(self, frame_index: int) -> np.ndarray | None:
        for index, box in self.history:
            if index == frame_index:
                return box
        return None


class ScaleTracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: dict[int, ScaleTrack] = {}
        self._next_id = 0
        self._H = np.hstack([np.eye(4), np.zeros((4, 3))])
        self._R = np.diag([cfg.meas_center_px**2, cfg.meas_center_px**2, cfg.meas_scale**2, cfg.meas_aspect**2])

    def update(self, detections: list[Detection], dt: float, frame_index: int) -> dict[int, ScaleTrack]:
        for track in self.tracks.values():
            self._predict(track, dt)
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)
        for tid, d in matched:
            self._correct(self.tracks[tid], detections[d])
        for tid in unmatched_tracks:
            self.tracks[tid].lost += 1
        for d in unmatched_dets:
            self._create(detections[d])
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.lost <= self.cfg.max_lost}
        for track in self.tracks.values():
            if not track.lost:  # predicted boxes must not stand in for measurements
                track.history.append((frame_index, track.bbox))
        return self.tracks

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = dt
        Q = np.zeros((7, 7))
        for pos, vel, sigma in ((0, 4, self.cfg.accel_px_s2), (1, 5, self.cfg.accel_px_s2), (2, 6, self.cfg.scale_accel_s2)):
            q = sigma**2
            Q[pos, pos], Q[pos, vel], Q[vel, pos], Q[vel, vel] = q * dt**3 / 3, q * dt**2 / 2, q * dt**2 / 2, q * dt
        Q[3, 3] = self.cfg.aspect_walk**2 * dt
        return F, Q

    def _predict(self, track: ScaleTrack, dt: float) -> None:
        F, Q = self._transition(dt)
        track.x = F @ track.x
        track.P = F @ track.P @ F.T + Q

    def _correct(self, track: ScaleTrack, det: Detection) -> None:
        y = box_to_z(det.bbox) - self._H @ track.x
        S = self._H @ track.P @ self._H.T + self._R
        K = track.P @ self._H.T @ np.linalg.inv(S)
        track.x = track.x + K @ y
        track.P = (np.eye(7) - K @ self._H) @ track.P
        track.class_name = det.class_name
        track.lost = 0
        track.age += 1

    def _associate(self, detections):
        ids = list(self.tracks)
        if not ids or not detections:
            return [], ids, list(range(len(detections)))
        ious = np.array([[iou(self.tracks[t].bbox, d.bbox) for d in detections] for t in ids])
        matched, used_t, used_d = [], set(), set()
        for r, c in zip(*linear_sum_assignment(-ious)):
            if ious[r, c] >= self.cfg.iou_threshold:
                matched.append((ids[r], c))
                used_t.add(ids[r])
                used_d.add(c)
        return matched, [t for t in ids if t not in used_t], [d for d in range(len(detections)) if d not in used_d]

    def _create(self, det: Detection) -> None:
        x = np.zeros(7)
        x[:4] = box_to_z(det.bbox)
        P = np.diag([*np.diag(self._R), 100.0**2, 100.0**2, 1.0])
        self.tracks[self._next_id] = ScaleTrack(self._next_id, x, P, det.class_name)
        self._next_id += 1
