"""Forward-collision-warning pipeline: the PC reference of the on-board app.

frame -> resize/gray -> flow (DIS on PC, Vitis LK on the board) + detections
      -> ScaleTracker -> ego-rotation -> 4 TTC methods per track -> fusion
      -> course check -> warning FSM
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from ..tracking import Detection
from .collision import CLASS_WIDTH_M, CourseChecker, CourseConfig, CourseResult, Level, WarningConfig, WarningFsm
from .divergence import DivergenceConfig, divergence_ttc, flow_moments, robust_mask
from .ego_rotation import EgoMotion, EgoRotationConfig, EgoRotationEstimator, rotation_divergence
from .fusion import FusionConfig, TtcEstimate, TtcFusion
from .horn import HornConfig, choose_level, horn_at_level
from .looming import looming
from .measurement import Measurement
from .scale_search import ScaleSearchConfig, choose_gap, scale_ttc
from .synth import intrinsics
from .tracker import HISTORY, ScaleTrack, ScaleTracker, TrackerConfig

FlowSource = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (prev_gray, curr_gray) -> (H, W, 2) px/frame
DetectorFn = Callable[[np.ndarray], list[Detection]]          # resized BGR frame -> detections


@dataclass(frozen=True)
class FcwConfig:
    frame_width: int = 512
    frame_height: int = 384
    fx: float = 500.0
    fy: float | None = None          # None: square pixels
    default_fps: float = 30.0        # used for the first dt only
    max_ttc_tracks: int = 16         # tracks that get TTC per frame (PL engine capacity)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    ego: EgoRotationConfig = field(default_factory=EgoRotationConfig)
    scale: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    horn: HornConfig = field(default_factory=HornConfig)
    divergence: DivergenceConfig = field(default_factory=DivergenceConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    course: CourseConfig = field(default_factory=CourseConfig)
    warning: WarningConfig = field(default_factory=WarningConfig)

    @property
    def K(self) -> np.ndarray:
        return intrinsics(self.fx, self.frame_width, self.frame_height, self.fy)


@dataclass
class ObjectResult:
    track_id: int
    bbox: np.ndarray
    class_name: str
    estimate: TtcEstimate | None
    course: CourseResult | None
    level: Level
    measurements: list[Measurement] = field(default_factory=list)  # everything produced, before gating


@dataclass
class FcwFrame:
    index: int
    time_s: float
    frame: np.ndarray            # resized BGR
    flow: np.ndarray
    ego: EgoMotion
    heading: tuple[float, float]  # FOE, or the principal point when there is none
    objects: list[ObjectResult]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def threat(self) -> ObjectResult | None:
        """The on-course object with the smallest TTC."""
        candidates = [o for o in self.objects if o.course and o.course.on_course and o.estimate and o.estimate.ttc_s]
        return min(candidates, key=lambda o: o.estimate.ttc_s, default=None)

    @property
    def level(self) -> Level:
        return max((o.level for o in self.objects), default=Level.NONE)


class DisFlow:
    """PC stand-in for the board's dense LK: OpenCV DIS optical flow."""

    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        self.dis = cv2.DISOpticalFlow_create(preset)

    def __call__(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        return self.dis.calc(prev, curr, None)


class FcwPipeline:
    def __init__(self, cfg: FcwConfig, detector: DetectorFn, flow_source: FlowSource):
        self.cfg = cfg
        self.detector = detector
        self.flow_source = flow_source
        self.K = cfg.K
        self.pp = (float(self.K[0, 2]), float(self.K[1, 2]))
        self.tracker = ScaleTracker(cfg.tracker)
        self.ego = EgoRotationEstimator(self.K, cfg.ego)
        self.fusion = TtcFusion(cfg.fusion)
        self.course = CourseChecker(cfg.course)
        self.fsms: dict[int, WarningFsm] = {}
        self.ring: deque = deque(maxlen=HISTORY)   # (frame_index, gray)
        self._index = -1
        self._prev_t: float | None = None
        self._on_course: set[int] = set()

    def process(self, frame_bgr: np.ndarray, t: float) -> FcwFrame:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        self._index += 1
        dt = t - self._prev_t if self._prev_t is not None else 0.0
        if dt <= 0:  # first frame, or a repeated/out-of-order timestamp
            dt = 1.0 / self.cfg.default_fps
        self._prev_t = t
        frame = cv2.resize(frame_bgr, (self.cfg.frame_width, self.cfg.frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = self.ring[-1][1] if self.ring else None

        mark = time.perf_counter()
        flow = self.flow_source(prev_gray, gray) if prev_gray is not None else np.zeros((*gray.shape, 2), np.float32)
        timings["flow"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        detections = [d for d in self.detector(frame) if d.class_name in self.cfg.course.class_width_m]
        timings["detection"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        prev_boxes = [tr.bbox for tr in self.tracker.tracks.values()]
        tracks = self.tracker.update(detections, dt, self._index)
        ego = self.ego.update(flow, prev_boxes) if prev_gray is not None else EgoMotion.identity()
        heading = tuple(ego.foe) if ego.foe is not None else self.pp
        yaw_rate = float(ego.rvec[1]) / dt
        self.ring.append((self._index, gray))
        timings["track_ego"] = (time.perf_counter() - mark) * 1e3

        mark = time.perf_counter()
        objects = [self._assess(track, gray, prev_gray, flow, ego, heading, yaw_rate, t, dt) for track in self._select(tracks)]
        live = list(tracks)
        self.fusion.retain(live)
        self.course.retain(live)
        self.fsms = {tid: f for tid, f in self.fsms.items() if tid in tracks}
        self._on_course = {o.track_id for o in objects if o.course and o.course.on_course}
        timings["ttc"] = (time.perf_counter() - mark) * 1e3
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return FcwFrame(self._index, t, frame, flow, ego, heading, objects, timings)

    def _select(self, tracks: dict[int, ScaleTrack]) -> list[ScaleTrack]:
        """At most max_ttc_tracks: last frame's on-course tracks first, then the largest boxes."""
        def priority(tr: ScaleTrack):
            x1, y1, x2, y2 = tr.bbox
            return (tr.id not in self._on_course, -(x2 - x1) * (y2 - y1))
        return sorted(tracks.values(), key=priority)[: self.cfg.max_ttc_tracks]

    def _gray_at(self, frame_index: int) -> np.ndarray | None:
        for index, gray in self.ring:
            if index == frame_index:
                return gray
        return None

    def _assess(self, track, gray, prev_gray, flow, ego, heading, yaw_rate, t, dt) -> ObjectResult:
        cfg = self.cfg
        box = track.bbox
        prior = self.fusion.filters.get(track.id)
        eta_prior = prior.eta if prior is not None else None
        measurements = [m for m in [looming(track)] if m is not None]
        if prev_gray is not None and not track.lost:
            k = choose_gap(eta_prior, dt, available=len(self.ring) - 1)
            gray_tk, box_tk = self._gray_at(self._index - k), track.box_at(self._index - k)
            if k and gray_tk is not None and box_tk is not None:
                m = scale_ttc(gray, gray_tk, box, box_tk, k, dt, cfg.scale)
                if m is not None:
                    measurements.append(m)
            level = choose_level(eta_prior, box, float(np.hypot(track.x[4], track.x[5])), dt)
            m = horn_at_level(prev_gray, gray, box, self.pp, level, dt, cfg.horn)
            if m is not None:
                measurements.append(m)
            mask = robust_mask(flow, box, self.pp, cfg.divergence)
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            m = divergence_ttc(flow_moments(flow, box, self.pp, mask, cfg.divergence.shrink), box,
                               rotation_divergence(ego.R, self.K, center), 1, dt, cfg.divergence)
            if m is not None:
                measurements.append(m)
        estimate = self.fusion.update(track.id, t, dt, measurements)
        ttc = estimate.ttc_s if estimate else None
        course = self.course.update(track.id, t, box, track.class_name, heading[0], heading[1], ttc, yaw_rate)
        fsm = self.fsms.setdefault(track.id, WarningFsm(cfg.warning))
        level = fsm.step(course.on_course, ttc, estimate.sigma_ttc_s if estimate else None,
                         len(estimate.recent_methods) if estimate else 0)
        return ObjectResult(track.id, box, track.class_name, estimate, course, level, measurements)
