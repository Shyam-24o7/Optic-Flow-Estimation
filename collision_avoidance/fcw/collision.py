"""Is a tracked object on our path, and what warning level does it earn?

Course check: the lateral offset in object widths, r = (x_c - x_ref) / w, equals
X / W_obj and does not depend on distance. Extrapolated to the moment of
contact, |r + r_dot * TTC| below half the combined width means a collision.

In path: the object's lateral extent [X - W_obj/2, X + W_obj/2] must overlap our
width [-W_ego/2, W_ego/2] by at least min_overlap of the narrower one. Only the
nearest in-path object (the lead vehicle) may warn; see pipeline.py. Real streets
put parked and oncoming cars 0.5-1 m beside our path, closer than box noise lets
the at-contact test resolve, which is why a strict "directly ahead" test gates it.

Entry speed: a long vehicle beside us shows more of its side as we close in, which
widens its box towards our path and makes r shrink with no lateral motion at all.
The rear face's outer edge and the box height are free of that (the side is
further away), so q = (x_outer - x_ref) / h = X_outer / H_obj changes only with
real lateral motion; dq/dt * H_obj is the speed at which the object nears our path.
x_ref is fixed (the principal point), not the heading: the FOE jumps when a large
object hides the background, and every jump would read as lateral motion. Our own
turning is removed by undoing each frame's rotation-only image shift at the edge.

Warning levels are anchored to NHTSA's FCW test (warning by TTC 2.0-2.4 s).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

CLASS_WIDTH_M = {"car": 1.8, "truck": 2.5, "bus": 2.5, "motorcycle": 0.8, "bicycle": 0.6, "person": 0.5}
CLASS_HEIGHT_M = {"car": 1.5, "truck": 3.0, "bus": 3.0, "motorcycle": 1.5, "bicycle": 1.7, "person": 1.7}


@dataclass(frozen=True)
class CourseConfig:
    ego_width_m: float = 1.8
    margin_m: float = 0.3
    class_width_m: dict = field(default_factory=lambda: dict(CLASS_WIDTH_M))
    class_height_m: dict = field(default_factory=lambda: dict(CLASS_HEIGHT_M))
    history_s: float = 0.5
    turn_yaw_rate_rps: float = float(np.radians(3.0))
    turn_widen: float = 1.5
    min_overlap: float = 0.5        # of the narrower of our width and the object's
    in_path_frames: int = 3         # consecutive in-path frames before the lead may warn
    max_entry_m: float = 4.0        # predicted entry: only objects within this lateral distance
    entry_frames: int = 6           # consecutive predicted-entry frames before it may warn
    min_entry_speed_mps: float = 0.3  # lateral speed towards our path; box jitter of parked objects stays below


@dataclass(frozen=True)
class CourseResult:
    on_course: bool                 # at the moment of contact
    r: float
    r_contact: float | None
    threshold: float
    in_path: bool = False           # now: overlaps our width enough to be the lead vehicle
    width_m: float | None = None    # class width prior used
    entry_speed_mps: float | None = None  # outer edge's speed towards our path (see module doc)


class CourseChecker:
    def __init__(self, cfg: CourseConfig = CourseConfig()):
        self.cfg = cfg
        self.history: dict[int, deque] = {}
        self.edges: dict[int, deque] = {}
        self.rot_px: dict[int, float] = {}   # accumulated rotation-only shift of the outer edge

    def update(self, track_id: int, t: float, box, class_name: str, heading_x: float, horizon_y: float,
               ttc_s: float | None, yaw_rate_rps: float, image_wh=None, rotation_h=None, ref_x=None) -> CourseResult:
        """rotation_h: K R K^-1 of this frame's camera rotation (prev -> curr); ref_x: fixed entry-speed reference."""
        x1, y1, x2, y2 = map(float, box)
        r = ((x1 + x2) / 2 - heading_x) / max(x2 - x1, 1.0)
        hist = self.history.setdefault(track_id, deque())
        hist.append((t, r))
        while hist and t - hist[0][0] > self.cfg.history_s:
            hist.popleft()
        entry_speed = self._entry_speed(track_id, t, x1, y1, x2, y2, heading_x if ref_x is None else ref_x, class_name,
                                        image_wh, rotation_h)
        width_m = self.cfg.class_width_m.get(class_name)
        if width_m is None:
            return CourseResult(False, r, None, 0.0)
        lateral = r * width_m
        overlap = min(lateral + width_m / 2, self.cfg.ego_width_m / 2) - max(lateral - width_m / 2, -self.cfg.ego_width_m / 2)
        in_path = y2 >= horizon_y and overlap >= self.cfg.min_overlap * min(width_m, self.cfg.ego_width_m)
        threshold = 0.5 * (1.0 + (self.cfg.ego_width_m + self.cfg.margin_m) / width_m)
        if abs(yaw_rate_rps) > self.cfg.turn_yaw_rate_rps:
            threshold *= self.cfg.turn_widen
        if y2 < horizon_y or ttc_s is None:
            return CourseResult(False, r, None, threshold, in_path, width_m, entry_speed)
        r_contact = r + (_slope(hist) or 0.0) * ttc_s
        return CourseResult(abs(r_contact) < threshold, r, r_contact, threshold, in_path, width_m, entry_speed)

    def _entry_speed(self, track_id, t, x1, y1, x2, y2, ref_x, class_name, image_wh, rotation_h) -> float | None:
        edges = self.edges.setdefault(track_id, deque())
        side = 1.0 if (x1 + x2) / 2 >= ref_x else -1.0
        x_out = x2 if side > 0 else x1
        clipped = image_wh is not None and (x1 <= 1 or y1 <= 1 or x2 >= image_wh[0] - 1 or y2 >= image_wh[1] - 1)
        if clipped or (edges and edges[-1][2] != side):  # a cut-off box has no true edge; the outer edge switched
            edges.clear()
        if not edges:
            self.rot_px[track_id] = 0.0
        elif rotation_h is not None:
            p = np.asarray(rotation_h, float) @ np.array([x_out, (y1 + y2) / 2, 1.0])
            self.rot_px[track_id] += p[0] / p[2] - x_out
        if not clipped:
            edges.append((t, (x_out - self.rot_px[track_id] - ref_x) / max(y2 - y1, 1.0), side))
        while edges and t - edges[0][0] > self.cfg.history_s:
            edges.popleft()
        height_m, q_dot = self.cfg.class_height_m.get(class_name), _slope([e[:2] for e in edges])
        return None if height_m is None or q_dot is None else -side * q_dot * height_m

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        self.history = {tid: h for tid, h in self.history.items() if tid in keep}
        self.edges = {tid: h for tid, h in self.edges.items() if tid in keep}
        self.rot_px = {tid: v for tid, v in self.rot_px.items() if tid in keep}


def _slope(samples) -> float | None:
    """Least-squares d(value)/dt over (t, value) samples; None without 3 samples over a real time span."""
    if len(samples) < 3:
        return None
    ts, vs = np.array(samples, float).T
    if ts[-1] - ts[0] <= 1e-3:  # repeated timestamps happen
        return None
    tc = ts - ts.mean()
    return float((tc * (vs - vs.mean())).sum() / (tc * tc).sum())


class Level(IntEnum):
    NONE = 0
    WARNING = 1
    CRITICAL = 2


@dataclass(frozen=True)
class WarningConfig:
    warn_ttc_s: float = 2.7
    warn_frames: int = 3
    warn_clear_ttc_s: float = 3.2
    critical_ttc_s: float = 1.5
    critical_clear_ttc_s: float = 2.0
    clear_frames: int = 10
    min_methods: int = 2


class WarningFsm:
    """One per track. Raise fast, clear slowly."""

    def __init__(self, cfg: WarningConfig = WarningConfig()):
        self.cfg = cfg
        self.level = Level.NONE
        self._warn_count = 0
        self._clear_count = 0

    def step(self, on_course: bool, ttc_s: float | None, sigma_ttc_s: float | None, recent_methods: int) -> Level:
        c = self.cfg
        can_raise = on_course and ttc_s is not None and recent_methods >= c.min_methods
        critical = can_raise and ttc_s + (sigma_ttc_s or 0.0) <= c.critical_ttc_s
        warn = can_raise and ttc_s <= c.warn_ttc_s
        if self.level == Level.NONE:
            self._warn_count = self._warn_count + 1 if warn else 0
            if critical:
                self._enter(Level.CRITICAL)
            elif self._warn_count >= c.warn_frames:
                self._enter(Level.WARNING)
        elif self.level == Level.WARNING:
            if critical:
                self._enter(Level.CRITICAL)
            else:
                clearing = not on_course or ttc_s is None or ttc_s > c.warn_clear_ttc_s
                self._clear_count = self._clear_count + 1 if clearing else 0
                if self._clear_count >= c.clear_frames:
                    self._enter(Level.NONE)
        else:
            clearing = not on_course or ttc_s is None or ttc_s > c.critical_clear_ttc_s
            self._clear_count = self._clear_count + 1 if clearing else 0
            if self._clear_count >= c.clear_frames:
                self._enter(Level.WARNING)
        return self.level

    def _enter(self, level: Level) -> None:
        self.level = level
        self._warn_count = 0
        self._clear_count = 0
