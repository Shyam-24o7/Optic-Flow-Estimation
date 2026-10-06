# FCW Algorithm Core (Python Reference) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the PC reference of the forward-collision-warning system (tracker, ego-rotation, four TTC methods, fusion, course check, warning FSM, pipeline, CLI and evaluation scripts) as the golden model for the C++ host app and the HLS engine.

**Architecture:** A new package `collision_avoidance/fcw/` beside the legacy modules, which stay untouched because `ttc_esn` imports them. Every stage is a small module with its own frozen config dataclass, tested against synthetic scenes with exact ground truth (`fcw/synth.py`). `fcw/pipeline.py` wires the stages behind two injected callables, a detector and a flow source, so the board can later swap OpenCV DIS flow for Vitis LK and the C++ app can mirror the same structure.

**Tech Stack:** Python 3.10–3.12, numpy, OpenCV (DIS flow, RANSAC, Rodrigues), scipy (Hungarian), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-kv260-collision-warning-design.md` (diagrams: https://claude.ai/artifact/LmHw3Nxg4rDhFAWXeXXZNY)

**Provenance:** every code block below was run before this plan was written: 62 tests pass on Python 3.11 (numpy 2.1.1, scipy 1.13.1) and Python 3.14. Spec section 12 lists what prototyping changed in the design.

**This is plan 1 of 3.** Plan 2: C++ host app and board bring-up (P0, P1 on the KV260), with parity tests against this package. Plan 3: the Horn TTC engine in HLS (P2), bit-exact against `fcw/horn.py`.

## Global Constraints

- Python `>=3.10,<3.13`, numpy `>=1.26,<2.2`, opencv-python `>=4.8,<4.11`, scipy `>=1.11,<1.15` (from `pyproject.toml`). Add no dependencies.
- Do not modify `collision_avoidance/ttc.py`, `tracking.py`, `ego_motion.py`, `pipeline.py` or anything in `ttc_esn/`. The new code only imports `Detection` and `iou` from `collision_avoidance/tracking.py`.
- Conventions everywhere: camera coordinates x right, y down, z forward; a rotation R maps camera coordinates prev → curr; flow is prev → curr in px/frame; boxes are `(x1, y1, x2, y2)` float pixels of the 512×384 processing frame; η = 1/TTC in s⁻¹.
- Thresholds, verbatim from the spec: warning at τ ≤ 2.7 s for 3 frames; critical at τ + σ ≤ 1.5 s (immediate); clear at τ > 3.2 s (warning) or > 2.0 s (critical), or off course, for 10 frames; raises need ≥ 2 methods in the last 0.5 s; W_ego = 1.8 m + 0.3 m; class widths car 1.8, truck/bus 2.5, motorcycle 0.8, bicycle 0.6, person 0.5 m; corridor ×1.5 above 3°/s yaw; at most 16 tracks get TTC; 9-frame gray ring; scale gaps {1, 2, 4, 8}; scale search [0.95, 1.25], coarse 0.02, 9 fine steps of 0.004; Horn gradient |Ex| + |Ey| > 64.
- Commit messages carry no `Co-Authored-By` trailer and no Claude attribution of any kind (mandatory user instruction).
- Do not push. Pushing uses the user's Shyam-24o7 GitHub account and is the user's call.
- `pytest tests/fcw -q` stays green after every task. Evaluation scripts honour `COLLISION_WANDB=0`.

## Review Focus

Input classes the spec implies, most likely to bite first. Each has a pinning test in the task that owns the code:

1. **Repeated or out-of-order timestamps** from a live camera: no crash, fall back to the nominal frame period. Tests: `test_repeated_timestamp_does_not_divide_by_zero` (Task 10), `test_repeated_timestamps_do_not_break_the_line_fit` (Task 9).
2. **Dropped frames:** TTC must follow real timestamps, not frame counts. Test: `test_dropped_frames_use_real_timestamps` (Task 10).
3. **Boxes partly outside the frame** (objects passing at the image edge): no crash, no false warning. Test: `test_box_leaving_the_frame_is_handled` (Task 10).
4. **More than 16 objects:** all are tracked, 16 get TTC. Test: `test_ttc_is_limited_to_sixteen_tracks` (Task 10).
5. **Textureless or noisy input** (night, sky, motion blur): methods return None instead of noise, and ego-rotation holds its last estimate. Tests: `test_textureless_box_is_rejected` (Task 6), `test_holds_last_estimate_when_flow_is_noise` (Task 4).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `collision_avoidance/fcw/__init__.py` | package docstring | 2 |
| `collision_avoidance/fcw/synth.py` | synthetic scenes and exact flow with ground truth | 2 |
| `collision_avoidance/fcw/measurement.py` | `Measurement` (method, η, variance) | 3 |
| `collision_avoidance/fcw/tracker.py` | `ScaleTracker`: log-scale Kalman tracks with box history | 3 |
| `collision_avoidance/fcw/looming.py` | η from the tracker's scale rate | 3 |
| `collision_avoidance/fcw/ego_rotation.py` | rotation, FOE, stationary detection, rotation divergence | 4 |
| `collision_avoidance/fcw/scale_search.py` | scale-ratio search between frames t−k and t; frame-gap choice | 5 |
| `collision_avoidance/fcw/horn.py` | Horn direct TTC: integer sums (golden model for HLS) and solve | 6 |
| `collision_avoidance/fcw/divergence.py` | flow moments, affine divergence, RANSAC refit | 7 |
| `collision_avoidance/fcw/fusion.py` | per-track EKF on η with gating and error floors | 8 |
| `collision_avoidance/fcw/collision.py` | course check and warning FSM | 9 |
| `collision_avoidance/fcw/pipeline.py` | `FcwPipeline`, `FcwConfig`, `DisFlow` | 10 |
| `collision_avoidance/fcw/metrics.py` | exit-criteria metrics | 11 |
| `collision_avoidance/fcw/render.py` | overlay drawing | 11 |
| `collision_avoidance/fcw/__main__.py` | CLI: `python -m collision_avoidance.fcw` | 11 |
| `collision_avoidance/fcw/kitti.py` | KITTI raw timestamps, camera matrix, OXTS yaw rate | 12 |
| `eval/kitti_yaw.py`, `eval/evttc_fcw.py`, `eval/false_alarms.py` | P1 evaluation scripts | 12 |
| `tests/fcw/test_*.py` | one test file per module | 2–12 |

---

### Task 1: Sync with the reconstructed upstream and set up the environment

This clone predates `shadowPunch/Optic-Flow-Estimation` PR #1, which restored the Python pipeline, the Vitis AI results and the tests this plan builds on. Local `main` is an ancestor of upstream `main`, so the design branch rebases cleanly.

**Files:** none changed by hand.

**Interfaces:**
- Produces: a working tree with upstream's `collision_avoidance/`, `ttc_esn/`, `tests/`, `deploy/`, `hls/` plus this branch's `docs/superpowers/`.

- [ ] **Step 1: Add the upstream remote and rebase the design branch**

```bash
git remote add upstream https://github.com/shadowPunch/Optic-Flow-Estimation.git
git fetch upstream
git switch design/kv260-fcw-spec
git rebase upstream/main
```

Expected: `Successfully rebased and updated refs/heads/design/kv260-fcw-spec.` and `git log --oneline -3` shows the spec and plan commits on top of `9f86384 Rewrite README for users; move technical detail to docs/methods.md`. The empty `PWC/ptlflow` gitlink is gone (upstream deleted it).

- [ ] **Step 2: Create a virtual environment and install the package**

Windows (PowerShell or Git Bash), Python 3.11:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

Linux/macOS: `python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`. With uv, upstream's README command also works: `uv venv --python 3.12 && uv pip install -e '.[dev]'`.

Activate it for the rest of the plan (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere).

- [ ] **Step 3: Run the existing test suite**

Run: `pytest -q`
Expected: all upstream tests pass (upstream reports its suite green). Tests that download weights need network access on the first run.

- [ ] **Step 4: No commit**

The rebase rewrote only this branch's own commits; there is nothing new to commit.

---

### Task 2: Synthetic scenes with exact ground truth

Every later test measures against these scenes. The object texture is deliberately coarse (cell 24): a finer texture aliases when the object is about 40 px wide and breaks the gradient-based methods.

**Files:**
- Create: `collision_avoidance/fcw/__init__.py`
- Create: `collision_avoidance/fcw/synth.py`
- Test: `tests/fcw/test_synth.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `intrinsics(fx, width, height, fy=None) -> np.ndarray (3x3)`, `rotation_y(angle_rad) -> np.ndarray`, `texture(height, width, seed, cell=6) -> uint8 array`, `SceneConfig` (frozen dataclass: width, height, fx, fps, n_frames, obj_width_m, obj_height_m, z0_m, closing_speed_mps, lateral_m, lateral_speed_mps, obj_y_m, yaw_rate_rps, bg_depth_m, seed), `Scene` (cfg, K, frames, boxes, ttc_s, depth_m, lateral_m, rotations; property `times`), `render_scene(cfg) -> Scene`, `flow_from_motion(K, R, t, depth, width, height, step=4) -> (pts (N,2), flow (N,2))`, `dense_flow_from_points(pts, flow, width, height)`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_synth.py`:

```python
import numpy as np
import pytest

from collision_avoidance.fcw import synth


def test_box_width_scales_with_inverse_depth():
    scene = synth.render_scene(synth.SceneConfig(n_frames=20))
    w0, w19 = scene.boxes[0][2] - scene.boxes[0][0], scene.boxes[19][2] - scene.boxes[19][0]
    assert w19 / w0 == pytest.approx(scene.depth_m[0] / scene.depth_m[19], rel=1e-6)
    assert scene.ttc_s[19] == pytest.approx(scene.depth_m[19] / 15.0)


def test_relative_rotation_matches_yaw_rate():
    cfg = synth.SceneConfig(n_frames=3, yaw_rate_rps=0.3)
    scene = synth.render_scene(cfg)
    np.testing.assert_allclose(scene.rotations[0], np.eye(3))
    np.testing.assert_allclose(scene.rotations[2], synth.rotation_y(-0.3 / cfg.fps), atol=1e-12)


def test_pure_rotation_flow_does_not_depend_on_depth():
    K = synth.intrinsics(500, 512, 384)
    R = synth.rotation_y(0.01)
    _, near = synth.flow_from_motion(K, R, np.zeros(3), 5.0, 512, 384, step=32)
    _, far = synth.flow_from_motion(K, R, np.zeros(3), 500.0, 512, 384, step=32)
    np.testing.assert_allclose(near, far, atol=1e-9)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_synth.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/__init__.py`:

```python
"""Forward collision warning: scale-change TTC, rotation-only ego-motion, course check, warnings.

See docs/superpowers/specs/2026-10-06-kv260-collision-warning-design.md.
"""
```

`collision_avoidance/fcw/synth.py`:

```python
"""Synthetic scenes with exact ground truth, for testing the FCW algorithms.

A textured rectangle (the object) at depth Z(t) in front of a far textured
background plane, seen by a pinhole camera that may yaw. Every quantity the
algorithms estimate (TTC, box, camera rotation) is known exactly.

Conventions used throughout collision_avoidance.fcw:
  * camera coordinates: x right, y down, z forward;
  * a rotation R "prev -> curr" maps camera coordinates of a static point at
    the previous frame to its coordinates at the current frame: X_curr = R X_prev.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


def intrinsics(fx: float, width: int, height: int, fy: float | None = None) -> np.ndarray:
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx if fy is None else fy, height / 2.0], [0.0, 0.0, 1.0]])


def rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation about the camera y axis (yaw); positive turns the camera to the right."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def texture(height: int, width: int, seed: int, cell: int = 6) -> np.ndarray:
    coarse = np.random.default_rng(seed).uniform(0, 255, (max(2, height // cell), max(2, width // cell))).astype(np.float32)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC).clip(0, 255).astype(np.uint8)


@dataclass(frozen=True)
class SceneConfig:
    width: int = 512
    height: int = 384
    fx: float = 500.0
    fps: float = 30.0
    n_frames: int = 30
    obj_width_m: float = 1.8
    obj_height_m: float = 1.5
    z0_m: float = 30.0                 # object depth at frame 0
    closing_speed_mps: float = 15.0    # dZ/dt = -closing_speed
    lateral_m: float = 0.0             # object centre X at frame 0 (+ right)
    lateral_speed_mps: float = 0.0
    obj_y_m: float = 0.3               # object centre Y (+ down)
    yaw_rate_rps: float = 0.0          # camera yaw rate
    bg_depth_m: float = 1000.0
    seed: int = 0


@dataclass
class Scene:
    cfg: SceneConfig
    K: np.ndarray
    frames: list[np.ndarray] = field(default_factory=list)     # uint8 gray
    boxes: list[np.ndarray] = field(default_factory=list)      # (x1, y1, x2, y2) float
    ttc_s: list[float] = field(default_factory=list)           # Z / closing speed (inf if not closing)
    depth_m: list[float] = field(default_factory=list)
    lateral_m: list[float] = field(default_factory=list)
    rotations: list[np.ndarray] = field(default_factory=list)  # prev -> curr; identity at frame 0

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.frames)) / self.cfg.fps


def _plane_homography(K, R_cam_to_world, scale_m_per_px, origin_m, depth_m) -> np.ndarray:
    """Texture pixel (u, v, 1) -> image pixel, for a fronto-parallel world plane at depth_m."""
    M = np.array([[scale_m_per_px[0], 0.0, origin_m[0]], [0.0, scale_m_per_px[1], origin_m[1]], [0.0, 0.0, depth_m]])
    return K @ R_cam_to_world.T @ M


def render_scene(cfg: SceneConfig) -> Scene:
    K = intrinsics(cfg.fx, cfg.width, cfg.height)
    size = (cfg.width, cfg.height)
    obj_tex = texture(160, 192, cfg.seed + 1, cell=24)  # coarse: must not alias when the object is ~40 px wide
    bg_tex = texture(2 * cfg.height, 2 * cfg.width, cfg.seed)
    bg_span = (1.6 * cfg.width / cfg.fx * cfg.bg_depth_m, 1.6 * cfg.height / cfg.fx * cfg.bg_depth_m)
    bg_scale = (bg_span[0] / bg_tex.shape[1], bg_span[1] / bg_tex.shape[0])
    obj_scale = (cfg.obj_width_m / obj_tex.shape[1], cfg.obj_height_m / obj_tex.shape[0])
    corners = np.array([[0, 0, 1], [obj_tex.shape[1], 0, 1], [0, obj_tex.shape[0], 1], [obj_tex.shape[1], obj_tex.shape[0], 1]], float).T

    scene = Scene(cfg, K)
    prev_R = np.eye(3)
    for i in range(cfg.n_frames):
        t = i / cfg.fps
        R = rotation_y(cfg.yaw_rate_rps * t)  # camera -> world
        z = cfg.z0_m - cfg.closing_speed_mps * t
        x = cfg.lateral_m + cfg.lateral_speed_mps * t
        H_bg = _plane_homography(K, R, bg_scale, (-bg_span[0] / 2, -bg_span[1] / 2), cfg.bg_depth_m)
        H_obj = _plane_homography(K, R, obj_scale, (x - cfg.obj_width_m / 2, cfg.obj_y_m - cfg.obj_height_m / 2), z)
        image = cv2.warpPerspective(bg_tex, H_bg, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        obj = cv2.warpPerspective(obj_tex, H_obj, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(obj_tex.shape, 255, np.uint8), H_obj, size, flags=cv2.INTER_LINEAR)
        alpha = mask.astype(np.float32) / 255.0
        image = (alpha * obj + (1 - alpha) * image).round().astype(np.uint8)

        pts = H_obj @ corners
        pts = pts[:2] / pts[2]
        scene.frames.append(image)
        scene.boxes.append(np.array([pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()]))
        scene.depth_m.append(z)
        scene.lateral_m.append(x)
        scene.ttc_s.append(z / cfg.closing_speed_mps if cfg.closing_speed_mps > 0 else float("inf"))
        scene.rotations.append(R.T @ prev_R)  # X_curr = R_curr^T R_prev X_prev
        prev_R = R
    return scene


def flow_from_motion(K: np.ndarray, R: np.ndarray, t: np.ndarray, depth, width: int, height: int, step: int = 4):
    """Exact sparse flow (prev -> curr) for camera motion X_curr = R X_prev + t.

    `depth` is a scalar or a callable (x_px, y_px) -> depth in metres at the
    previous frame. Returns points (N, 2) and flow (N, 2) for a regular grid.
    """
    ys, xs = np.mgrid[step // 2:height:step, step // 2:width:step]
    xs, ys = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
    z = depth(xs, ys) if callable(depth) else np.full(xs.shape, float(depth))
    rays = np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)])
    X_curr = R @ (rays * z) + np.asarray(t, float).reshape(3, 1)
    p = K @ X_curr
    p = p[:2] / p[2]
    pts = np.stack([xs, ys], axis=1)
    return pts, p.T - pts


def dense_flow_from_points(pts: np.ndarray, flow: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scatter grid flow into a dense (H, W, 2) array (nearest grid sample), for tests."""
    out = np.zeros((height, width, 2), np.float32)
    step = int(round(pts[1, 0] - pts[0, 0])) if len(pts) > 1 else 1
    for (x, y), f in zip(pts.astype(int), flow):
        out[max(0, y - step // 2):y + step - step // 2, max(0, x - step // 2):x + step - step // 2] = f
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_synth.py -v`
Expected: `3 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/__init__.py collision_avoidance/fcw/synth.py tests/fcw/test_synth.py
git commit -m "Add synthetic FCW scenes with exact ground truth"
```

---

### Task 3: Measurement type, log-scale tracker and looming

Each track predicts from its own state, which removes the wrong-track lookup in `hls/ports/object_tracker.hpp`. The scale rate σ̇ of the state is the looming estimate of 1/TTC.

**Files:**
- Create: `collision_avoidance/fcw/measurement.py`
- Create: `collision_avoidance/fcw/tracker.py`
- Create: `collision_avoidance/fcw/looming.py`
- Test: `tests/fcw/test_tracker.py`

**Interfaces:**
- Consumes: `Detection(bbox, class_name, confidence)` and `iou(a, b)` from `collision_avoidance/tracking.py`; `synth.render_scene` (tests).
- Produces: `METHODS = ("looming", "scale", "horn", "divergence")`; `Measurement(method: str, eta: float, var: float)` with property `ttc_s`; `HISTORY = 9`; `TrackerConfig`; `box_to_z(box)`, `z_to_box(z)`; `ScaleTrack(id, x (7,), P (7,7), class_name, lost, age, history)` with properties `bbox`, `scale_rate`, `scale_rate_var` and method `box_at(frame_index) -> box | None`; `ScaleTracker(cfg).update(detections, dt, frame_index) -> dict[int, ScaleTrack]` and attribute `tracks`; `looming(track) -> Measurement | None` (None until 3 detections and while lost).

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_tracker.py`:

```python
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.looming import looming
from collision_avoidance.fcw.tracker import ScaleTracker, box_to_z, z_to_box
from collision_avoidance.tracking import Detection

DT = 1 / 30


def det(box, name="car"):
    return Detection(np.asarray(box, float), name, 0.9)


def test_box_state_round_trip():
    box = np.array([100.0, 50.0, 180.0, 110.0])
    np.testing.assert_allclose(z_to_box(box_to_z(box)), box)


def test_scale_rate_converges_to_inverse_ttc():
    scene = synth.render_scene(synth.SceneConfig(n_frames=40, z0_m=40.0))
    tracker = ScaleTracker()
    for i, box in enumerate(scene.boxes):
        tracks = tracker.update([det(box)], DT, i)
    (track,) = tracks.values()
    assert track.scale_rate == pytest.approx(1 / scene.ttc_s[-1], rel=0.1)
    m = looming(track)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[-1], rel=0.1)


def test_each_track_predicts_from_its_own_state():
    # Regression for hls/ports/object_tracker.hpp, which predicted from the wrong track.
    tracker = ScaleTracker()
    for i in range(10):
        tracker.update([det([10 + 2 * i, 10, 50 + 2 * i, 40]), det([300, 200 - i, 400, 260 - i], "truck")], DT, i)
    small, large = tracker.tracks[0], tracker.tracks[1]
    assert small.class_name == "car" and large.class_name == "truck"
    np.testing.assert_allclose(small.bbox, [28, 10, 68, 40], atol=1.0)
    np.testing.assert_allclose(large.bbox, [300, 191, 400, 251], atol=1.0)


def test_lost_tracks_expire_and_history_is_kept():
    tracker = ScaleTracker()
    tracker.update([det([10, 10, 50, 50])], DT, 0)
    for i in range(1, 33):
        tracker.update([], DT, i)
    assert tracker.tracks == {}
    tracker.update([det([10, 10, 50, 50])], DT, 40)
    track = next(iter(tracker.tracks.values()))
    assert track.box_at(40) is not None and track.box_at(39) is None


def test_looming_needs_a_few_detections():
    tracker = ScaleTracker()
    tracks = tracker.update([det([10, 10, 50, 50])], DT, 0)
    assert looming(next(iter(tracks.values()))) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_tracker.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.looming'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/measurement.py`:

```python
"""The common output of every TTC method: an inverse TTC with its variance."""

from dataclasses import dataclass

METHODS = ("looming", "scale", "horn", "divergence")


@dataclass(frozen=True)
class Measurement:
    method: str   # one of METHODS
    eta: float    # inverse TTC in 1/s; > 0 approaching, <= 0 receding or static
    var: float    # variance of eta in 1/s^2

    @property
    def ttc_s(self) -> float | None:
        return 1.0 / self.eta if self.eta > 0 else None
```

`collision_avoidance/fcw/tracker.py`:

```python
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
```

`collision_avoidance/fcw/looming.py`:

```python
"""Looming: the tracker's log-scale rate is the inverse TTC."""

from .measurement import Measurement
from .tracker import ScaleTrack

MIN_AGE = 3  # the scale rate is meaningless until a few detections have been fused


def looming(track: ScaleTrack) -> Measurement | None:
    if track.age < MIN_AGE or track.lost:
        return None
    return Measurement("looming", track.scale_rate, track.scale_rate_var)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_tracker.py -v`
Expected: `5 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/measurement.py collision_avoidance/fcw/tracker.py collision_avoidance/fcw/looming.py tests/fcw/test_tracker.py
git commit -m "Add log-scale Kalman tracker and looming TTC"
```

---

### Task 4: Ego-rotation and heading from flow

Two prototype lessons are built in. A rotation-only homography is preferred when it explains at least 90% of the essential matrix's inliers, since E is degenerate without translation. Confidence is E's RANSAC inlier share, because `recoverPose`'s mask also drops far background.

**Files:**
- Create: `collision_avoidance/fcw/ego_rotation.py`
- Test: `tests/fcw/test_ego_rotation.py`

**Interfaces:**
- Consumes: `synth.intrinsics`, `synth.rotation_y`, `synth.flow_from_motion` (tests).
- Produces: `EgoRotationConfig`; `EgoMotion(R, rvec, foe, inlier_ratio, stationary, valid)` with `omega_rps(dt)` and `EgoMotion.identity()`; `grid_samples(flow, exclude_boxes, cfg)`; `estimate(prev_pts, curr_pts, K, cfg) -> EgoMotion | None`; `EgoRotationEstimator(K, cfg).update(flow, exclude_boxes) -> EgoMotion`; `rotation_divergence(R, K, center_xy) -> float` (per-frame divergence the rotation adds; equals 3(x_n·ω_y − y_n·ω_x) for small rotations).

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_ego_rotation.py`:

```python
import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.ego_rotation import EgoRotationEstimator, rotation_divergence

W, H = 512, 384
K = synth.intrinsics(500, W, H)


def dense_flow(R, t, depth):
    _, flow = synth.flow_from_motion(K, R, t, depth, W, H, step=1)
    return flow.reshape(H, W, 2).astype(np.float32)


def road_depth(xs, ys):
    """Varied depths so translation produces parallax (a flat wall would be degenerate)."""
    rng = np.random.default_rng(0)
    return 8.0 + 60.0 * rng.random(xs.shape)


def angle_deg(R):
    return np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))


@pytest.mark.parametrize("yaw_deg", [0.0, 0.4, -0.8])
def test_recovers_rotation_and_heading_while_driving_forward(yaw_deg):
    R = synth.rotation_y(np.radians(yaw_deg))
    t = np.array([0.0, 0.0, -0.5])  # camera moves 0.5 m forward: points come closer
    ego = EgoRotationEstimator(K).update(dense_flow(R, t, road_depth), [])
    assert ego.valid and not ego.stationary
    assert angle_deg(ego.R @ R.T) < 0.1
    np.testing.assert_allclose(ego.foe, (K @ t)[:2] / (K @ t)[2], atol=3.0)


def test_pure_rotation_is_reported_as_stationary():
    R = synth.rotation_y(np.radians(0.5))
    ego = EgoRotationEstimator(K).update(dense_flow(R, np.zeros(3), 30.0), [])
    assert ego.valid and ego.stationary and ego.foe is None
    assert angle_deg(ego.R @ R.T) < 0.05


def test_excluded_boxes_ignore_a_moving_object():
    R, t = synth.rotation_y(np.radians(0.3)), np.array([0.0, 0.0, -0.5])
    flow = dense_flow(R, t, road_depth)
    flow[100:300, 150:350] += np.array([6.0, -2.0], np.float32)  # an independently moving truck
    ego = EgoRotationEstimator(K).update(flow, [(150, 100, 350, 300)])
    assert angle_deg(ego.R @ R.T) < 0.1


def test_holds_last_estimate_when_flow_is_noise():
    est = EgoRotationEstimator(K)
    good = est.update(dense_flow(synth.rotation_y(np.radians(0.3)), np.array([0, 0, -0.5]), road_depth), [])
    noise = np.random.default_rng(1).normal(0, 5, (H, W, 2)).astype(np.float32)
    held = est.update(noise, [])
    assert not held.valid
    np.testing.assert_allclose(held.R, good.R)


@pytest.mark.parametrize("center", [(256, 192), (400, 100), (60, 330)])
def test_rotation_divergence_matches_small_angle_formula(center):
    rvec = np.array([0.004, -0.006, 0.002])
    R = cv2.Rodrigues(rvec)[0]
    xn, yn = (center[0] - K[0, 2]) / K[0, 0], (center[1] - K[1, 2]) / K[1, 1]
    assert rotation_divergence(R, K, center) == pytest.approx(3 * (xn * rvec[1] - yn * rvec[0]), rel=0.05, abs=1e-4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_ego_rotation.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.ego_rotation'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/ego_rotation.py`:

```python
"""Camera rotation and heading (focus of expansion) from a dense flow field.

Only rotation is ever removed from the flow: forward translation is what makes
obstacles ahead expand, so it is the collision signal and must stay.

Rotation R maps camera coordinates prev -> curr (see synth.py for conventions).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EgoRotationConfig:
    grid_step_px: int = 16
    box_margin: float = 0.10          # enlarge excluded boxes by this fraction per side
    hood_rows: int = 0                # bottom image rows hidden by the car's hood
    ransac_threshold_px: float = 1.0
    ransac_confidence: float = 0.999
    rotation_only_tol: float = 0.01   # max |singular value - 1| of K^-1 H K for a pure rotation
    rotation_only_share: float = 0.9  # homography must explain this share of E's inliers
    min_inlier_ratio: float = 0.5
    max_jump_rad: float = np.radians(1.0)  # per-frame change of rotation angle accepted
    min_points: int = 30
    min_foe_tz: float = 0.5           # |t_z| of the unit translation needed for a usable FOE
    foe_smoothing: float = 0.5        # EMA weight of the newest FOE


@dataclass
class EgoMotion:
    R: np.ndarray                 # 3x3, prev -> curr
    rvec: np.ndarray              # Rodrigues vector of R, radians over the frame pair
    foe: np.ndarray | None        # (x, y) px; None when stationary or not moving forward
    inlier_ratio: float
    stationary: bool
    valid: bool                   # False: low confidence, R held from an earlier frame

    def omega_rps(self, dt: float) -> np.ndarray:
        return self.rvec / dt

    @staticmethod
    def identity() -> "EgoMotion":
        return EgoMotion(np.eye(3), np.zeros(3), None, 0.0, True, False)


def grid_samples(flow: np.ndarray, exclude_boxes, cfg: EgoRotationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Matched points (prev, curr) on a regular grid, skipping boxes and the hood."""
    h, w = flow.shape[:2]
    s = cfg.grid_step_px
    ys, xs = np.mgrid[s // 2:h - cfg.hood_rows:s, s // 2:w:s]
    keep = np.ones(xs.shape, bool)
    for x1, y1, x2, y2 in exclude_boxes:
        mx, my = cfg.box_margin * (x2 - x1), cfg.box_margin * (y2 - y1)
        keep &= ~((xs >= x1 - mx) & (xs <= x2 + mx) & (ys >= y1 - my) & (ys <= y2 + my))
    xs, ys = xs[keep], ys[keep]
    prev = np.stack([xs, ys], axis=1).astype(np.float64)
    return prev, prev + flow[ys, xs].astype(np.float64)


def _rotation_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest rotation to K^-1 H K and how far that matrix is from a rotation."""
    A = np.linalg.inv(K) @ H @ K
    A = A / np.cbrt(np.linalg.det(A))
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R, float(np.abs(sv - 1.0).max())


def _angle(R: np.ndarray) -> float:
    return float(np.linalg.norm(cv2.Rodrigues(R)[0]))


def estimate(prev_pts: np.ndarray, curr_pts: np.ndarray, K: np.ndarray, cfg: EgoRotationConfig) -> EgoMotion | None:
    """One unsmoothed estimate, or None when the points cannot support one."""
    n = len(prev_pts)
    if n < cfg.min_points:
        return None
    E, e_mask = cv2.findEssentialMat(prev_pts, curr_pts, K, cv2.RANSAC, cfg.ransac_confidence, cfg.ransac_threshold_px)
    e_ratio = 0.0 if E is None else float(e_mask.sum()) / n
    # Pure rotation (or a stopped car) is degenerate for E: prefer the rotation
    # homography when it explains about as many points as E does.
    H, h_mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, cfg.ransac_threshold_px)
    if H is not None:
        R_h, deviation = _rotation_from_homography(H, K)
        h_ratio = float(h_mask.sum()) / n
        if deviation < cfg.rotation_only_tol and h_ratio >= cfg.rotation_only_share * e_ratio:
            return EgoMotion(R_h, cv2.Rodrigues(R_h)[0].ravel(), None, h_ratio, True, True)
    if E is None:
        return None
    E = E[:3]  # findEssentialMat may stack several solutions
    # recoverPose's own mask also drops points beyond 50 baselines (far background),
    # so the RANSAC inlier share of E is the confidence measure.
    _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, K, mask=e_mask.copy())
    ratio = e_ratio
    t = t.ravel() / np.linalg.norm(t)
    foe = None
    if abs(t[2]) >= cfg.min_foe_tz:
        p = K @ t
        foe = p[:2] / p[2]
    return EgoMotion(R, cv2.Rodrigues(R)[0].ravel(), foe, ratio, False, True)


class EgoRotationEstimator:
    """Per-frame estimate with outlier holding and FOE smoothing."""

    def __init__(self, K: np.ndarray, cfg: EgoRotationConfig = EgoRotationConfig()):
        self.K = K
        self.cfg = cfg
        self.last = EgoMotion.identity()

    def update(self, flow: np.ndarray, exclude_boxes) -> EgoMotion:
        prev_pts, curr_pts = grid_samples(flow, exclude_boxes, self.cfg)
        est = estimate(prev_pts, curr_pts, self.K, self.cfg)
        if est is None or est.inlier_ratio < self.cfg.min_inlier_ratio:
            return self._hold()
        if self.last.valid and abs(_angle(est.R) - _angle(self.last.R)) > self.cfg.max_jump_rad:
            return self._hold()
        if est.foe is not None and self.last.foe is not None:
            a = self.cfg.foe_smoothing
            est.foe = a * est.foe + (1 - a) * self.last.foe
        self.last = est
        return est

    def _hold(self) -> EgoMotion:
        held = EgoMotion(self.last.R, self.last.rvec, self.last.foe, 0.0, self.last.stationary, False)
        return held


def rotation_divergence(R: np.ndarray, K: np.ndarray, center_xy) -> float:
    """Divergence (per frame) that the rotation alone adds to the flow at a pixel.

    Computed from the exact rotation homography H = K R K^-1 by central
    differences. For a small rotation vector w it equals
    3 * (x_n * w_y - y_n * w_x) with normalised coordinates x_n, y_n.
    """
    H = K @ R @ np.linalg.inv(K)
    cx, cy = map(float, center_xy)

    def mapped(x, y):
        p = H @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    d = 1.0
    dgx = (mapped(cx + d, cy)[0] - mapped(cx - d, cy)[0]) / (2 * d)
    dgy = (mapped(cx, cy + d)[1] - mapped(cx, cy - d)[1]) / (2 * d)
    return float(dgx + dgy - 2.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_ego_rotation.py -v`
Expected: `9 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/ego_rotation.py tests/fcw/test_ego_rotation.py
git commit -m "Add rotation-only ego-motion estimation from flow"
```

---

### Task 5: Scale-ratio search

**Files:**
- Create: `collision_avoidance/fcw/scale_search.py`
- Test: `tests/fcw/test_scale_search.py`

**Interfaces:**
- Consumes: `Measurement` (Task 3); `synth.render_scene` (tests).
- Produces: `ScaleSearchConfig`; `crop(gray, center, size_wh, patch)`; `ncc(a, b)`; `search_scale(gray_t, gray_tk, box_t, box_tk, cfg) -> (s, peak_ncc) | None`; `scale_ttc(gray_t, gray_tk, box_t, box_tk, k, dt, cfg) -> Measurement | None` (method `"scale"`, η = (s − 1)/(k·dt)); `choose_gap(eta_estimate, dt, available, min_expansion=0.02) -> int` (0 when no history).

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_scale_search.py`:

```python
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.scale_search import choose_gap, scale_ttc

DT = 1 / 30


@pytest.mark.parametrize("k", [2, 4, 8])
def test_recovers_ttc_of_approaching_object(k):
    scene = synth.render_scene(synth.SceneConfig(n_frames=25))
    t = 24
    m = scale_ttc(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k], k, DT)
    assert m is not None
    assert m.ttc_s == pytest.approx(scene.ttc_s[t], rel=0.05)


def test_static_object_has_no_expansion():
    scene = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0))
    m = scale_ttc(scene.frames[8], scene.frames[0], scene.boxes[8], scene.boxes[0], 8, DT)
    assert m is not None and abs(m.eta) < 0.05


def test_sideways_motion_does_not_fake_an_approach():
    scene = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0, lateral_speed_mps=4.0))
    m = scale_ttc(scene.frames[8], scene.frames[0], scene.boxes[8], scene.boxes[0], 8, DT)
    assert m is not None and abs(m.eta) < 0.05


def test_tiny_box_is_rejected():
    scene = synth.render_scene(synth.SceneConfig(n_frames=5, z0_m=200.0))
    assert scale_ttc(scene.frames[4], scene.frames[0], scene.boxes[4], scene.boxes[0], 4, DT) is None


def test_gap_choice():
    assert choose_gap(None, DT, available=8) == 8
    assert choose_gap(1.0, DT, available=8) == 1     # 1/s * 1/30 s = 3.3 % >= 2 %
    assert choose_gap(0.25, DT, available=8) == 4    # 2 frames: 1.7 %, 4 frames: 3.3 %
    assert choose_gap(0.25, DT, available=3) == 2    # largest gap the history allows
    assert choose_gap(0.25, DT, available=0) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_scale_search.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.scale_search'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/scale_search.py`:

```python
"""Scale-ratio search: how much did the object grow between frame t-k and t?

The box at t (shrunk to stay on the object) is resampled to a fixed patch.
Crops of frame t-k around the earlier box centre, sized 1/s of that, are
compared with zero-mean NCC for a range of scales s. For constant closing
speed s = Z(t-k) / Z(t), so TTC(t) = k*dt / (s - 1) and eta = (s - 1) / (k*dt).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .measurement import Measurement


@dataclass(frozen=True)
class ScaleSearchConfig:
    patch: int = 32
    shrink: float = 0.10       # crop (1 - shrink) of the box width and height
    s_min: float = 0.95
    s_max: float = 1.25
    coarse_step: float = 0.02  # 16 coarse scales over [s_min, s_max]
    fine_step: float = 0.004   # 9 fine scales around the best coarse one
    fine_count: int = 9
    min_ncc: float = 0.5
    min_box_px: float = 12.0


def crop(gray: np.ndarray, center, size_wh, patch: int) -> np.ndarray:
    """Bilinear resample of a (w, h) window centred at `center` to patch x patch."""
    w, h = size_wh
    sx, sy = w / patch, h / patch
    # patch pixel (i, j) centre -> image point center + ((i + 0.5) * s - w / 2)
    M = np.array([[sx, 0.0, center[0] - w / 2 + 0.5 * sx - 0.5], [0.0, sy, center[1] - h / 2 + 0.5 * sy - 0.5]])
    return cv2.warpAffine(gray, M, (patch, patch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / denom if denom > 0 else 0.0


def _parabola_peak(c_m: float, c_0: float, c_p: float) -> float:
    """Offset (in steps, within [-0.5, 0.5]) of the maximum of a parabola through 3 samples."""
    denom = c_m - 2 * c_0 + c_p
    return 0.0 if denom >= 0 else float(np.clip(0.5 * (c_m - c_p) / denom, -0.5, 0.5))


def search_scale(gray_t, gray_tk, box_t, box_tk, cfg: ScaleSearchConfig = ScaleSearchConfig()) -> tuple[float, float] | None:
    """Best scale s = size(t) / size(t-k) and its NCC, or None."""
    w_t, h_t = (box_t[2] - box_t[0]) * (1 - cfg.shrink), (box_t[3] - box_t[1]) * (1 - cfg.shrink)
    if min(w_t, h_t) < cfg.min_box_px:
        return None
    c_t = ((box_t[0] + box_t[2]) / 2, (box_t[1] + box_t[3]) / 2)
    c_tk = ((box_tk[0] + box_tk[2]) / 2, (box_tk[1] + box_tk[3]) / 2)
    template = crop(gray_t, c_t, (w_t, h_t), cfg.patch)

    def score(s: float) -> float:
        return ncc(template, crop(gray_tk, c_tk, (w_t / s, h_t / s), cfg.patch))

    coarse = np.arange(cfg.s_min, cfg.s_max + 1e-9, cfg.coarse_step)
    best = float(coarse[int(np.argmax([score(s) for s in coarse]))])
    half = cfg.fine_count // 2
    fine = best + cfg.fine_step * np.arange(-half, half + 1)
    scores = np.array([score(s) for s in fine])
    i = int(np.argmax(scores))
    if i in (0, len(fine) - 1):
        return None  # peak not bracketed: outside the fine window
    s_star = float(fine[i] + cfg.fine_step * _parabola_peak(scores[i - 1], scores[i], scores[i + 1]))
    if not cfg.s_min < s_star < cfg.s_max:
        return None
    return s_star, float(scores[i])


def scale_ttc(gray_t, gray_tk, box_t, box_tk, k: int, dt: float, cfg: ScaleSearchConfig = ScaleSearchConfig()) -> Measurement | None:
    """Inverse TTC from the growth of the box between frames t-k and t (dt = frame period)."""
    found = search_scale(gray_t, gray_tk, box_t, box_tk, cfg)
    if found is None:
        return None
    s, peak = found
    if peak < cfg.min_ncc:
        return None
    span = k * dt
    sigma_s = cfg.fine_step * (1.0 + 10.0 * (1.0 - peak))
    return Measurement("scale", (s - 1.0) / span, (sigma_s / span) ** 2)


def choose_gap(eta_estimate: float | None, dt: float, available: int, min_expansion: float = 0.02) -> int:
    """Smallest gap in {1, 2, 4, 8} whose expected expansion reaches min_expansion."""
    gaps = [k for k in (1, 2, 4, 8) if k <= available]
    if not gaps:
        return 0
    if eta_estimate is None or eta_estimate <= 0:
        return gaps[-1]
    for k in gaps:
        if eta_estimate * k * dt >= min_expansion:
            return k
    return gaps[-1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_scale_search.py -v`
Expected: `7 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/scale_search.py tests/fcw/test_scale_search.py
git commit -m "Add scale-ratio search TTC"
```

---

### Task 6: Horn direct TTC (integer golden model)

`horn_sums` is the golden model for the HLS engine (plan 3): `test_integer_sums_match_a_pixel_loop` pins its exact integer definition. Sideways motion over about 1 px per frame needs a coarser pyramid level, which `choose_level` picks.

**Files:**
- Create: `collision_avoidance/fcw/horn.py`
- Test: `tests/fcw/test_horn.py`

**Interfaces:**
- Consumes: `Measurement` (Task 3); `synth.render_scene` (tests).
- Produces: `HORN_TERMS` (10 names), `SOBEL_GAIN = 16`, `HornConfig`; `gradients(prev, curr) -> (ex, ey, et)` int64; `inner_box(box, shrink, width, height)`; `horn_sums(prev, curr, box, principal_point, cfg) -> (sums int64 (10,), n) | None`; `horn_solve(sums, n, dt, cfg) -> Measurement | None` (method `"horn"`); `choose_level(eta_estimate, box, center_speed_px_s, dt, levels=3) -> int`; `downsample(gray, level)`; `horn_at_level(prev, curr, box, principal_point, level, dt, cfg) -> Measurement | None`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_horn.py`:

```python
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.horn import HornConfig, choose_level, gradients, horn_at_level, horn_solve, horn_sums

DT = 1 / 30
PP = (256, 192)


def horn_ttc(scene, t):
    sums, n = horn_sums(scene.frames[t - 1], scene.frames[t], scene.boxes[t], PP)
    return horn_solve(sums, n, DT)


def test_recovers_ttc_of_approaching_object():
    scene = synth.render_scene(synth.SceneConfig(n_frames=21))
    m = horn_ttc(scene, 20)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[20], rel=0.1)


def test_sideways_motion_needs_a_coarser_level():
    # 1.5 m/s sideways at 11 m is ~2.3 px/frame: too much for level 0, fine at level 2.
    scene = synth.render_scene(synth.SceneConfig(n_frames=21, z0_m=15.0, closing_speed_mps=6.0, lateral_m=-1.0, lateral_speed_mps=1.5))
    box = scene.boxes[20]
    speed = abs((box[0] + box[2]) - (scene.boxes[19][0] + scene.boxes[19][2])) / 2 / DT
    level = choose_level(1 / scene.ttc_s[20], box, speed, DT)
    assert level == 2
    m = horn_at_level(scene.frames[19], scene.frames[20], box, PP, level, DT)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[20], rel=0.15)


def test_textureless_box_is_rejected():
    flat = np.full((384, 512), 128, np.uint8)
    found = horn_sums(flat, flat, (100, 100, 200, 200), PP)
    assert found is not None
    assert horn_solve(*found, DT) is None


def test_integer_sums_match_a_pixel_loop():
    """Pins the exact integer definition the HLS engine must reproduce."""
    rng = np.random.default_rng(3)
    prev = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    curr = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    box, pp = (10, 8, 30, 28), (25, 20)
    cfg = HornConfig(shrink=0.0, grad_threshold_l1=0)
    sums, n = horn_sums(prev, curr, box, pp, cfg)

    ex, ey, et = gradients(prev, curr)
    expected = np.zeros(10, np.int64)
    count = 0
    for y in range(8, 28):
        for x in range(10, 30):
            gx, gy, gt = int(ex[y, x]), int(ey[y, x]), int(et[y, x])
            if abs(gx) + abs(gy) <= 0:
                continue
            g = (x - 25) * gx + (y - 20) * gy
            expected += np.array([gx * gx, gx * gy, gx * g, gy * gy, gy * g, g * g, gx * gt, gy * gt, g * gt, gt * gt])
            count += 1
    assert sums.dtype == np.int64 and n == count
    np.testing.assert_array_equal(sums, expected)


def test_level_choice_keeps_motion_under_a_pixel():
    box = (0, 0, 120, 90)  # half-diagonal 75 px
    assert choose_level(None, box, 0.0, DT) == 0
    assert choose_level(1.0, box, 0.0, DT) == 2   # 75 * 1/30 = 2.5 px -> /4 at level 2
    assert choose_level(0.2, box, 30.0, DT) == 1  # 0.5 + 1.0 = 1.5 px -> /2 at level 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_horn.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.horn'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/horn.py`:

```python
"""Horn direct time-to-contact, as integer sums the PL engine can produce.

Model for a flat object facing the camera (Horn, Fang & Masaki 2007):
    A*Ex + B*Ey + C*G + Et = 0,   G = x*Ex + y*Ey,
with x, y measured from the principal point. Moving the origin only
reparameterises A and B, so every box can share per-pixel products.

Integer definitions (the HLS engine must match these bit for bit):
    S  = prev + curr                       (uint8 + uint8, 9 bit)
    Ex = Sobel_x(S), Ey = Sobel_y(S)       (3x3, 12 bit signed; interior pixels only)
    Et = curr - prev                       (9 bit signed)
Sobel of the frame sum is 16x the mean-intensity derivative, so the fitted C'
is 1/16 of the expansion rate per frame: C = 16 * C'.
The 10 sums per box: Ex^2, ExEy, ExG, Ey^2, EyG, G^2, ExEt, EyEt, GEt, Et^2.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .measurement import Measurement

HORN_TERMS = ("ExEx", "ExEy", "ExG", "EyEy", "EyG", "GG", "ExEt", "EyEt", "GEt", "EtEt")
SOBEL_GAIN = 16


@dataclass(frozen=True)
class HornConfig:
    grad_threshold_l1: int = 64     # |Ex| + |Ey| must exceed this (Sobel-of-sum units)
    shrink: float = 0.10            # use (1 - shrink) of the box, to stay on the object
    max_condition: float = 1e4      # of the column-normalised normal matrix
    max_rel_residual: float = 0.5   # residual / sum(Et^2)
    min_pixels: int = 100


def gradients(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ex, Ey, Et as int64 arrays; Ex/Ey are 0 on the 1-pixel border."""
    s = prev.astype(np.int64) + curr.astype(np.int64)
    ex = np.zeros_like(s)
    ey = np.zeros_like(s)
    ex[1:-1, 1:-1] = (s[:-2, 2:] + 2 * s[1:-1, 2:] + s[2:, 2:]) - (s[:-2, :-2] + 2 * s[1:-1, :-2] + s[2:, :-2])
    ey[1:-1, 1:-1] = (s[2:, :-2] + 2 * s[2:, 1:-1] + s[2:, 2:]) - (s[:-2, :-2] + 2 * s[:-2, 1:-1] + s[:-2, 2:])
    return ex, ey, curr.astype(np.int64) - prev.astype(np.int64)


def inner_box(box, shrink: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer pixel range [x1, x2) x [y1, y2) of the shrunk box, clipped to the Sobel interior."""
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(1, int(np.ceil(x1 + mx))), max(1, int(np.ceil(y1 + my)))
    ix2, iy2 = min(width - 1, int(np.floor(x2 - mx))), min(height - 1, int(np.floor(y2 - my)))
    return (ix1, iy1, ix2, iy2) if ix2 > ix1 and iy2 > iy1 else None


def horn_sums(prev: np.ndarray, curr: np.ndarray, box, principal_point, cfg: HornConfig = HornConfig()) -> tuple[np.ndarray, int] | None:
    """The 10 int64 sums of HORN_TERMS over the box, and the pixel count."""
    h, w = curr.shape
    rng = inner_box(box, cfg.shrink, w, h)
    if rng is None:
        return None
    x1, y1, x2, y2 = rng
    ex, ey, et = gradients(prev[y1 - 1:y2 + 1, x1 - 1:x2 + 1], curr[y1 - 1:y2 + 1, x1 - 1:x2 + 1])
    ex, ey, et = ex[1:-1, 1:-1], ey[1:-1, 1:-1], et[1:-1, 1:-1]
    px, py = int(round(principal_point[0])), int(round(principal_point[1]))
    ys, xs = np.mgrid[y1:y2, x1:x2]
    g = (xs - px) * ex + (ys - py) * ey
    keep = (np.abs(ex) + np.abs(ey)) > cfg.grad_threshold_l1
    ex, ey, et, g = ex[keep], ey[keep], et[keep], g[keep]
    sums = np.array([(ex * ex).sum(), (ex * ey).sum(), (ex * g).sum(), (ey * ey).sum(), (ey * g).sum(),
                     (g * g).sum(), (ex * et).sum(), (ey * et).sum(), (g * et).sum(), (et * et).sum()], dtype=np.int64)
    return sums, int(keep.sum())


def horn_solve(sums: np.ndarray, n: int, dt: float, cfg: HornConfig = HornConfig()) -> Measurement | None:
    """Inverse TTC from the box sums; None when the fit is unreliable."""
    if n < cfg.min_pixels:
        return None
    s = sums.astype(np.float64)
    M = np.array([[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]])
    q = np.array([s[6], s[7], s[8]])
    diag = np.sqrt(np.diag(M))
    if np.any(diag <= 0):
        return None
    Mn = M / np.outer(diag, diag)
    if np.linalg.cond(Mn) > cfg.max_condition:
        return None
    p = np.linalg.solve(M, -q)                       # (A', B', C')
    residual = s[9] + 2 * p @ q + p @ M @ p          # sum of squared equation errors
    if s[9] <= 0 or residual / s[9] > cfg.max_rel_residual:
        return None
    sigma2 = max(residual, 0.0) / max(n - 3, 1)
    var_c = sigma2 * np.linalg.inv(M)[2, 2]
    eta = SOBEL_GAIN * p[2] / dt
    return Measurement("horn", float(eta), float(SOBEL_GAIN**2 * var_c / dt**2))


def choose_level(eta_estimate: float | None, box, center_speed_px_s: float, dt: float, levels: int = 3) -> int:
    """Pyramid level at which the motion inside the box stays under about 1 px."""
    half_diag = 0.5 * float(np.hypot(box[2] - box[0], box[3] - box[1]))
    motion = (abs(eta_estimate) if eta_estimate else 0.0) * dt * half_diag + center_speed_px_s * dt
    level = 0
    while motion / 2**level > 1.0 and level < levels - 1:
        level += 1
    return level


def downsample(gray: np.ndarray, level: int) -> np.ndarray:
    for _ in range(level):
        gray = cv2.pyrDown(gray)
    return gray


def horn_at_level(prev: np.ndarray, curr: np.ndarray, box, principal_point, level: int, dt: float,
                  cfg: HornConfig = HornConfig()) -> Measurement | None:
    """horn_sums + horn_solve on pyramid level `level` (box and principal point scaled to it)."""
    f = 2.0**-level
    found = horn_sums(downsample(prev, level), downsample(curr, level), np.asarray(box, float) * f,
                      (principal_point[0] * f, principal_point[1] * f), cfg)
    return None if found is None else horn_solve(*found, dt, cfg)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_horn.py -v`
Expected: `5 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/horn.py tests/fcw/test_horn.py
git commit -m "Add Horn direct TTC with integer box sums"
```

---

### Task 7: LK divergence with rotation removal

The CPU-side refit is RANSAC. An iterated least-squares refit was tried first and failed `test_robust_refit_ignores_background_pixels`: the first fit is dragged by the background before any residual test can reject it.

**Files:**
- Create: `collision_avoidance/fcw/divergence.py`
- Test: `tests/fcw/test_divergence.py`

**Interfaces:**
- Consumes: `Measurement` (Task 3); `ego_rotation.rotation_divergence` and `synth` (tests).
- Produces: `FLOW_TERMS` (12 names), `DivergenceConfig`; `flow_moments(flow, box, principal_point, mask=None, shrink=0.10) -> np.ndarray (12,) | None`; `affine_from_moments(m) -> (a (3,), b (3,)) | None`; `divergence_ttc(m, box, rotation_div, k, dt, cfg) -> Measurement | None` (method `"divergence"`, η = div/(2·k·dt)); `robust_mask(flow, box, principal_point, cfg) -> bool mask (H, W)`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_divergence.py`:

```python
import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.divergence import divergence_ttc, flow_moments, robust_mask
from collision_avoidance.fcw.ego_rotation import rotation_divergence

DT = 1 / 30
W, H = 512, 384
K = synth.intrinsics(500, W, H)
PP = (K[0, 2], K[1, 2])
BOX = (180, 140, 300, 230)


def looming_flow(rate, center=(240, 185), lateral=(0.0, 0.0)):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.stack([(xs - center[0]) * rate + lateral[0], (ys - center[1]) * rate + lateral[1]], axis=-1)


def test_recovers_ttc_from_expanding_flow():
    rate = 0.025  # per frame -> TTC = 40 frames = 1.333 s
    m = divergence_ttc(flow_moments(looming_flow(rate, lateral=(2.0, -1.0)), BOX, PP), BOX, 0.0, 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=1e-6)


def test_camera_rotation_is_removed():
    rate = 0.02
    R = cv2.Rodrigues(np.array([0.01, -0.015, 0.0]))[0]
    _, rot = synth.flow_from_motion(K, R, np.zeros(3), 50.0, W, H, step=1)
    flow = looming_flow(rate) + rot.reshape(H, W, 2)
    center = ((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2)
    m = divergence_ttc(flow_moments(flow, BOX, PP), BOX, rotation_divergence(R, K, center), 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=0.02)
    uncorrected = divergence_ttc(flow_moments(flow, BOX, PP), BOX, 0.0, 1, DT)
    assert abs(uncorrected.ttc_s - 1 / (rate * 30)) > 0.05  # the correction matters here


def test_robust_refit_ignores_background_pixels():
    rate = 0.02
    flow = looming_flow(rate)
    flow[140:160, 180:300] = (5.0, 0.0)  # background leaking into the box top
    mask = robust_mask(flow, BOX, PP)
    m = divergence_ttc(flow_moments(flow, BOX, PP, mask), BOX, 0.0, 1, DT)
    assert m.ttc_s == pytest.approx(1 / (rate * 30), rel=0.02)


def test_empty_box_returns_none():
    assert flow_moments(looming_flow(0.02), (600, 10, 700, 50), PP) is None
    assert divergence_ttc(None, BOX, 0.0, 1, DT) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_divergence.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.divergence'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/divergence.py`:

```python
"""Flow divergence per box from an affine fit, after removing camera rotation.

Inside a box, u = a0 + a1*x + a2*y and v = b0 + b1*x + b2*y; div = a1 + b2.
For an object approaching at constant speed, the translational flow of a
fronto-parallel surface has div = 2 / TTC_frames, so eta = div / (2 * k * dt).

The fit needs only 12 sums per box (FLOW_TERMS), which the PL engine
accumulates in the same pass as the Horn sums.
"""

from dataclasses import dataclass

import numpy as np

from .measurement import Measurement

FLOW_TERMS = ("n", "x", "y", "xx", "xy", "yy", "u", "v", "xu", "yu", "xv", "yv")


@dataclass(frozen=True)
class DivergenceConfig:
    shrink: float = 0.10
    flow_noise_px: float = 0.3   # assumed per-pixel flow noise, sets the variance
    inlier_px: float = 1.0       # robust refit: flow residual that still counts as the object
    ransac_iterations: int = 64
    ransac_stride: int = 2       # subsample pixels for the RANSAC hypotheses
    min_pixels: int = 50


def flow_moments(flow: np.ndarray, box, principal_point, mask: np.ndarray | None = None, shrink: float = 0.10) -> np.ndarray | None:
    """The 12 FLOW_TERMS sums over the shrunk box (float64), or None if empty."""
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    mx, my = shrink / 2 * (x2 - x1), shrink / 2 * (y2 - y1)
    ix1, iy1 = max(0, int(np.ceil(x1 + mx))), max(0, int(np.ceil(y1 + my)))
    ix2, iy2 = min(w, int(np.floor(x2 - mx))), min(h, int(np.floor(y2 - my)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    ys, xs = np.mgrid[iy1:iy2, ix1:ix2]
    x = (xs - principal_point[0]).astype(np.float64)
    y = (ys - principal_point[1]).astype(np.float64)
    u = flow[iy1:iy2, ix1:ix2, 0].astype(np.float64)
    v = flow[iy1:iy2, ix1:ix2, 1].astype(np.float64)
    keep = np.ones(x.shape, bool) if mask is None else mask[iy1:iy2, ix1:ix2]
    x, y, u, v = x[keep], y[keep], u[keep], v[keep]
    return np.array([x.size, x.sum(), y.sum(), (x * x).sum(), (x * y).sum(), (y * y).sum(),
                     u.sum(), v.sum(), (x * u).sum(), (y * u).sum(), (x * v).sum(), (y * v).sum()])


def affine_from_moments(m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(a0, a1, a2), (b0, b1, b2) of the least-squares affine flow."""
    n, sx, sy, sxx, sxy, syy, su, sv, sxu, syu, sxv, syv = m
    A = np.array([[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]])
    if n < 3 or abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, [su, sxu, syu]), np.linalg.solve(A, [sv, sxv, syv])


def divergence_ttc(m: np.ndarray, box, rotation_div: float, k: int, dt: float,
                   cfg: DivergenceConfig = DivergenceConfig()) -> Measurement | None:
    """Inverse TTC from flow moments; rotation_div is subtracted (see ego_rotation.rotation_divergence)."""
    if m is None or m[0] < cfg.min_pixels:
        return None
    fit = affine_from_moments(m)
    if fit is None:
        return None
    a, b = fit
    div = a[1] + b[2] - rotation_div
    w, h = box[2] - box[0], box[3] - box[1]
    var_div = cfg.flow_noise_px**2 * 12.0 / m[0] * (1.0 / w**2 + 1.0 / h**2)
    span = 2.0 * k * dt
    return Measurement("divergence", float(div / span), float(var_div / span**2))


def robust_mask(flow: np.ndarray, box, principal_point, cfg: DivergenceConfig = DivergenceConfig()) -> np.ndarray:
    """Pixels that follow the dominant affine flow in the box (RANSAC; CPU-side refit).

    A plain least-squares fit is dragged towards background pixels leaking into
    the box before any residual test can reject them, so hypotheses come from
    random 3-pixel samples and the one with most inliers wins.
    """
    h, w = flow.shape[:2]
    mask = np.ones((h, w), bool)
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return mask
    ys, xs = np.mgrid[y1:y2, x1:x2]
    x, y = xs - principal_point[0], ys - principal_point[1]
    u, v = flow[y1:y2, x1:x2, 0], flow[y1:y2, x1:x2, 1]
    s = cfg.ransac_stride
    X = np.stack([np.ones(x[::s, ::s].size), x[::s, ::s].ravel(), y[::s, ::s].ravel()], axis=1)
    U, V = u[::s, ::s].ravel(), v[::s, ::s].ravel()
    rng = np.random.default_rng(0)
    best, best_count = None, -1
    for _ in range(cfg.ransac_iterations):
        idx = rng.choice(len(X), 3, replace=False)
        try:
            a = np.linalg.solve(X[idx], U[idx])
            b = np.linalg.solve(X[idx], V[idx])
        except np.linalg.LinAlgError:
            continue
        count = int((np.hypot(X @ a - U, X @ b - V) < cfg.inlier_px).sum())
        if count > best_count:
            best, best_count = (a, b), count
    if best is None:
        return mask
    a, b = best
    r = np.hypot(u - (a[0] + a[1] * x + a[2] * y), v - (b[0] + b[1] * x + b[2] * y))
    mask[y1:y2, x1:x2] = r < cfg.inlier_px
    return mask
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_divergence.py -v`
Expected: `4 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/divergence.py tests/fcw/test_divergence.py
git commit -m "Add affine-flow divergence TTC with rotation removal"
```

---

### Task 8: Per-track fusion of inverse TTC

The `rel_sigma_floor` defaults came from the prototype. On a synthetic approach, Horn read 10–30% fast while claiming about 1% uncertainty, and dominated the fused TTC until its variance was floored. Task 13 replaces the floors' companion `var_scale` with values calibrated on EvTTC.

**Files:**
- Create: `collision_avoidance/fcw/fusion.py`
- Test: `tests/fcw/test_fusion.py`

**Interfaces:**
- Consumes: `Measurement`, `METHODS` (Task 3).
- Produces: `FusionConfig` (process_var_per_s, gate_sigma2, max_consecutive_rejects, min_eta, recent_window_s, var_scale, rel_sigma_floor, abs_sigma_floor); `TtcEstimate(track_id, eta, eta_var, ttc_s, sigma_ttc_s, measurements, recent_methods)`; `EtaFilter`; `TtcFusion(cfg).update(track_id, t, dt, measurements) -> TtcEstimate | None`, `.retain(track_ids)`, attribute `filters: dict[int, EtaFilter]` (each with `.eta`).

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_fusion.py`:

```python
import numpy as np
import pytest

from collision_avoidance.fcw.fusion import FusionConfig, TtcFusion
from collision_avoidance.fcw.measurement import Measurement

DT = 1 / 30


def test_tracks_constant_closing_speed():
    fusion, rng = TtcFusion(), np.random.default_rng(0)
    tau0 = 3.0
    for i in range(60):
        tau = tau0 - i * DT
        ms = [Measurement("scale", 1 / tau + rng.normal(0, 0.03), 0.03**2),
              Measurement("horn", 1 / tau + rng.normal(0, 0.05), 0.05**2)]
        est = fusion.update(7, i * DT, DT, ms)
    assert est.ttc_s == pytest.approx(tau, rel=0.05)
    assert est.recent_methods == {"scale", "horn"}


def test_outlier_is_gated():
    fusion = TtcFusion()
    for i in range(20):
        fusion.update(1, i * DT, DT, [Measurement("scale", 0.5, 0.01**2)])
    est = fusion.update(1, 20 * DT, DT, [Measurement("horn", 5.0, 0.05**2)])
    assert est.measurements == ()
    assert est.ttc_s == pytest.approx(2.0, rel=0.05)


def test_reinitialises_after_repeated_rejections():
    fusion = TtcFusion(FusionConfig(max_consecutive_rejects=3))
    for i in range(10):
        fusion.update(1, i * DT, DT, [Measurement("scale", 0.2, 0.01**2)])
    for i in range(10, 15):
        est = fusion.update(1, i * DT, DT, [Measurement("scale", 2.0, 0.01**2)])
    assert est.eta == pytest.approx(2.0, rel=0.05)


def test_receding_object_has_no_ttc():
    est = TtcFusion().update(1, 0.0, DT, [Measurement("looming", -0.3, 0.01)])
    assert est.ttc_s is None and est.eta < 0


def test_retain_drops_filters():
    fusion = TtcFusion()
    fusion.update(1, 0.0, DT, [Measurement("looming", 0.3, 0.01)])
    fusion.update(2, 0.0, DT, [Measurement("looming", 0.3, 0.01)])
    fusion.retain([2])
    assert set(fusion.filters) == {2}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_fusion.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.fusion'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/fusion.py`:

```python
"""Per-track fusion of inverse-TTC measurements with a one-state EKF.

The state is eta = 1/TTC. With constant closing speed TTC shrinks by dt each
step, so eta' = eta / (1 - eta*dt). Working in eta keeps receding and static
objects (eta <= 0) finite and makes measurement noise roughly Gaussian.
"""

from dataclasses import dataclass, field

import numpy as np

from .measurement import METHODS, Measurement


@dataclass(frozen=True)
class FusionConfig:
    process_var_per_s: float = 0.25        # growth of var(eta) per second (braking, accelerating)
    gate_sigma2: float = 9.0               # Mahalanobis gate (3 sigma)
    max_consecutive_rejects: int = 5       # then re-initialise from the measurements
    min_eta: float = 0.05                  # below this the object is "not approaching" (TTC > 20 s)
    recent_window_s: float = 0.5           # for "methods that contributed recently"
    var_scale: dict = field(default_factory=lambda: {m: 1.0 for m in METHODS})  # calibrated per method (P1)
    # Least-squares variances capture noise, not model error. Floor each method's
    # sigma at this fraction of |eta| (plus abs_sigma_floor). Defaults come from the
    # synthetic-scene bias of each method; P1 calibration on EvTTC replaces them.
    rel_sigma_floor: dict = field(default_factory=lambda: {"looming": 0.0, "scale": 0.05, "horn": 0.15, "divergence": 0.30})
    abs_sigma_floor: float = 0.02


@dataclass(frozen=True)
class TtcEstimate:
    track_id: int
    eta: float
    eta_var: float
    ttc_s: float | None
    sigma_ttc_s: float | None
    measurements: tuple[Measurement, ...]   # accepted this frame
    recent_methods: frozenset[str]         # methods accepted within recent_window_s


class EtaFilter:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.eta: float | None = None
        self.var = 0.0
        self.rejects = 0
        self.last_accept: dict[str, float] = {}

    def predict(self, dt: float) -> None:
        if self.eta is None:
            return
        denom = max(1.0 - self.eta * dt, 0.1)
        jac = 1.0 / denom**2
        self.eta = self.eta / denom
        self.var = jac * jac * self.var + self.cfg.process_var_per_s * dt

    def update(self, t: float, measurements: list[Measurement]) -> list[Measurement]:
        scaled = [self._effective(m) for m in measurements]
        if self.eta is None or self.rejects >= self.cfg.max_consecutive_rejects:
            return self._initialise(t, scaled)
        accepted = []
        for m in scaled:
            s = self.var + m.var
            innovation = m.eta - self.eta
            if innovation * innovation / s > self.cfg.gate_sigma2:
                continue
            gain = self.var / s
            self.eta += gain * innovation
            self.var *= 1.0 - gain
            self.last_accept[m.method] = t
            accepted.append(m)
        self.rejects = 0 if accepted or not scaled else self.rejects + 1
        return accepted

    def _effective(self, m: Measurement) -> Measurement:
        floor = self.cfg.rel_sigma_floor.get(m.method, 0.0) * abs(m.eta) + self.cfg.abs_sigma_floor
        return Measurement(m.method, m.eta, max(m.var * self.cfg.var_scale.get(m.method, 1.0), floor * floor))

    def _initialise(self, t: float, measurements: list[Measurement]) -> list[Measurement]:
        if not measurements:
            return []
        w = np.array([1.0 / m.var for m in measurements])
        self.eta = float(np.dot(w, [m.eta for m in measurements]) / w.sum())
        self.var = float(1.0 / w.sum())
        self.rejects = 0
        for m in measurements:
            self.last_accept[m.method] = t
        return measurements

    def recent_methods(self, t: float) -> frozenset[str]:
        return frozenset(m for m, ts in self.last_accept.items() if t - ts <= self.cfg.recent_window_s)


class TtcFusion:
    def __init__(self, cfg: FusionConfig = FusionConfig()):
        self.cfg = cfg
        self.filters: dict[int, EtaFilter] = {}

    def update(self, track_id: int, t: float, dt: float, measurements: list[Measurement]) -> TtcEstimate | None:
        f = self.filters.setdefault(track_id, EtaFilter(self.cfg))
        f.predict(dt)
        accepted = f.update(t, measurements)
        if f.eta is None:
            return None
        approaching = f.eta > self.cfg.min_eta
        return TtcEstimate(
            track_id, f.eta, f.var,
            1.0 / f.eta if approaching else None,
            float(np.sqrt(f.var)) / f.eta**2 if approaching else None,
            tuple(accepted), f.recent_methods(t),
        )

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        self.filters = {tid: f for tid, f in self.filters.items() if tid in keep}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_fusion.py -v`
Expected: `5 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/fusion.py tests/fcw/test_fusion.py
git commit -m "Add per-track eta fusion with gating and error floors"
```

---

### Task 9: Course check and warning FSM

**Files:**
- Create: `collision_avoidance/fcw/collision.py`
- Test: `tests/fcw/test_collision.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CLASS_WIDTH_M`; `CourseConfig`; `CourseResult(on_course, r, r_contact, threshold)`; `CourseChecker(cfg).update(track_id, t, box, class_name, heading_x, horizon_y, ttc_s, yaw_rate_rps) -> CourseResult`, `.retain(track_ids)`; `Level` (IntEnum NONE=0, WARNING=1, CRITICAL=2); `WarningConfig`; `WarningFsm(cfg).step(on_course, ttc_s, sigma_ttc_s, recent_methods: int) -> Level`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_collision.py`:

```python
import pytest

from collision_avoidance.fcw.collision import CourseChecker, Level, WarningFsm

DT = 1 / 30


def run_course(checker, boxes, heading_x=256.0, horizon_y=192.0, ttc=2.0, yaw=0.0, name="car"):
    result = None
    for i, box in enumerate(boxes):
        result = checker.update(1, i * DT, box, name, heading_x, horizon_y, ttc, yaw)
    return result


def test_object_straight_ahead_is_on_course():
    assert run_course(CourseChecker(), [(226, 200, 286, 240)] * 10).on_course


def test_parked_car_beside_lane_is_not_on_course():
    # Static object 3 m to the right: r = X / W stays 3 / 1.8 while the box grows.
    boxes = [(256 + 500 * 2.1 / z, 200, 256 + 500 * 3.9 / z, 240) for z in (20 - 0.5 * i for i in range(10))]
    res = run_course(CourseChecker(), boxes)
    assert not res.on_course and res.r == pytest.approx(3 / 1.8, rel=1e-6)


def test_object_cutting_in_is_on_course():
    # Starts 2.5 m right, moves left 2 m/s: at contact (TTC 2 s) it is at -1.5 m... just inside.
    boxes = []
    for i in range(15):
        x, z = 2.5 - 2.0 * i * DT, 20.0
        boxes.append((256 + 500 * (x - 0.9) / z, 200, 256 + 500 * (x + 0.9) / z, 240))
    assert run_course(CourseChecker(), boxes, ttc=1.5).on_course


def test_repeated_timestamps_do_not_break_the_line_fit():
    checker = CourseChecker()
    for _ in range(5):
        res = checker.update(1, 0.0, (226, 200, 286, 240), "car", 256.0, 192.0, 2.0, 0.0)
    assert res.on_course


def test_objects_above_horizon_and_unknown_classes_are_ignored():
    assert not run_course(CourseChecker(), [(226, 100, 286, 150)] * 5).on_course
    assert not run_course(CourseChecker(), [(226, 200, 286, 240)] * 5, name="dog").on_course


def test_turning_widens_the_corridor():
    box = [(256 + 500 * 1.5 / 20, 200, 256 + 500 * 3.3 / 20, 240)]  # X = 2.4 m: r = 1.33, threshold 1.08 (x1.5 = 1.63)
    assert not run_course(CourseChecker(), box * 5).on_course
    assert run_course(CourseChecker(), box * 5, yaw=0.1).on_course


def steps(fsm, inputs):
    return [fsm.step(*args) for args in inputs]


def test_warning_needs_three_frames_and_two_methods():
    fsm = WarningFsm()
    assert steps(fsm, [(True, 2.5, 0.1, 1)] * 5)[-1] == Level.NONE
    assert steps(fsm, [(True, 2.5, 0.1, 2)] * 3) == [Level.NONE, Level.NONE, Level.WARNING]


def test_critical_is_immediate_and_clears_slowly():
    fsm = WarningFsm()
    assert fsm.step(True, 1.2, 0.2, 2) == Level.CRITICAL
    levels = steps(fsm, [(True, 2.5, 0.1, 2)] * 10)
    assert levels[:9] == [Level.CRITICAL] * 9 and levels[9] == Level.WARNING
    levels = steps(fsm, [(True, 3.5, 0.1, 2)] * 10)
    assert levels[-1] == Level.NONE


def test_warning_clears_when_object_leaves_path():
    fsm = WarningFsm()
    steps(fsm, [(True, 2.5, 0.1, 2)] * 3)
    assert steps(fsm, [(False, 2.5, 0.1, 2)] * 10)[-1] == Level.NONE
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_collision.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.collision'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/collision.py`:

```python
"""Is a tracked object on our path, and what warning level does it earn?

Course check: the lateral offset in object widths, r = (x_c - x_F) / w, equals
X / W_obj and does not depend on distance. Extrapolated to the moment of
contact, |r + r_dot * TTC| below half the combined width means a collision.

Warning levels are anchored to NHTSA's FCW test (warning by TTC 2.0-2.4 s).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

CLASS_WIDTH_M = {"car": 1.8, "truck": 2.5, "bus": 2.5, "motorcycle": 0.8, "bicycle": 0.6, "person": 0.5}


@dataclass(frozen=True)
class CourseConfig:
    ego_width_m: float = 1.8
    margin_m: float = 0.3
    class_width_m: dict = field(default_factory=lambda: dict(CLASS_WIDTH_M))
    history_s: float = 0.5
    turn_yaw_rate_rps: float = float(np.radians(3.0))
    turn_widen: float = 1.5


@dataclass(frozen=True)
class CourseResult:
    on_course: bool
    r: float
    r_contact: float | None
    threshold: float


class CourseChecker:
    def __init__(self, cfg: CourseConfig = CourseConfig()):
        self.cfg = cfg
        self.history: dict[int, deque] = {}

    def update(self, track_id: int, t: float, box, class_name: str, heading_x: float, horizon_y: float,
               ttc_s: float | None, yaw_rate_rps: float) -> CourseResult:
        x1, y1, x2, y2 = map(float, box)
        r = ((x1 + x2) / 2 - heading_x) / max(x2 - x1, 1.0)
        hist = self.history.setdefault(track_id, deque())
        hist.append((t, r))
        while hist and t - hist[0][0] > self.cfg.history_s:
            hist.popleft()
        width_m = self.cfg.class_width_m.get(class_name)
        if width_m is None:
            return CourseResult(False, r, None, 0.0)
        threshold = 0.5 * (1.0 + (self.cfg.ego_width_m + self.cfg.margin_m) / width_m)
        if abs(yaw_rate_rps) > self.cfg.turn_yaw_rate_rps:
            threshold *= self.cfg.turn_widen
        if y2 < horizon_y or ttc_s is None:
            return CourseResult(False, r, None, threshold)
        r_dot = 0.0
        ts, rs = np.array(hist).T
        if len(hist) >= 3 and ts[-1] - ts[0] > 1e-3:  # needs a real time span (repeated timestamps happen)
            tc = ts - ts.mean()
            r_dot = float((tc * (rs - rs.mean())).sum() / (tc * tc).sum())
        r_contact = r + r_dot * ttc_s
        return CourseResult(abs(r_contact) < threshold, r, r_contact, threshold)

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        self.history = {tid: h for tid, h in self.history.items() if tid in keep}


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_collision.py -v`
Expected: `9 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/collision.py tests/fcw/test_collision.py
git commit -m "Add FOE-relative course check and warning state machine"
```

---

### Task 10: FCW pipeline

On the prototype's synthetic head-on approach this pipeline warns at true TTC 2.73 s and goes critical at 1.27 s. The test bounds are looser than that (warning at a true TTC between 2.0 and 3.0 s) so they don't depend on DIS flow details across OpenCV versions.

**Files:**
- Create: `collision_avoidance/fcw/pipeline.py`
- Test: `tests/fcw/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces: `FlowSource = Callable[[prev_gray, curr_gray], flow (H, W, 2)]`; `DetectorFn = Callable[[bgr_512x384], list[Detection]]`; `FcwConfig` (frame size, fx, fy, default_fps, max_ttc_tracks and every sub-config) with property `K`; `ObjectResult(track_id, bbox, class_name, estimate, course, level, measurements)`; `FcwFrame(index, time_s, frame, flow, ego, heading, objects, timings_ms)` with properties `threat` and `level`; `DisFlow()` (PC flow source); `FcwPipeline(cfg, detector, flow_source).process(frame_bgr, t) -> FcwFrame`, attribute `tracker`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_pipeline.py`:

```python
import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.tracking import Detection


def run_scene(scene_cfg: synth.SceneConfig, class_name: str = "car"):
    scene = synth.render_scene(scene_cfg)
    frames = iter(range(len(scene.frames)))
    current = {"i": 0}

    def detector(_frame):
        return [Detection(scene.boxes[current["i"]], class_name, 0.9)]

    pipeline = FcwPipeline(FcwConfig(fx=scene_cfg.fx), detector, DisFlow())
    results = []
    for i in frames:
        current["i"] = i
        results.append(pipeline.process(cv2.cvtColor(scene.frames[i], cv2.COLOR_GRAY2BGR), i / scene_cfg.fps))
    return scene, results


def first_frame_at(results, level):
    return next((r.index for r in results if r.level >= level), None)


def test_head_on_approach_warns_in_time():
    scene, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    warn = first_frame_at(results, Level.WARNING)
    crit = first_frame_at(results, Level.CRITICAL)
    assert warn is not None and scene.ttc_s[warn] >= 2.0          # NHTSA: warned before TTC 2.0 s
    assert scene.ttc_s[warn] <= 3.0                                # and not absurdly early
    assert crit is not None and scene.ttc_s[crit] >= 1.0
    final = results[-1].objects[0].estimate
    assert final.ttc_s == pytest.approx(scene.ttc_s[-1], rel=0.15)


def test_car_in_the_next_lane_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0, lateral_m=3.5))
    assert first_frame_at(results, Level.WARNING) is None


def test_static_object_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=40, z0_m=20.0, closing_speed_mps=0.0))
    assert first_frame_at(results, Level.WARNING) is None
    assert results[-1].objects[0].estimate.ttc_s is None


def test_dropped_frames_use_real_timestamps():
    scene = synth.render_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    for current in range(0, 80, 2):  # every other frame arrives
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
    assert result.objects[0].estimate.ttc_s == pytest.approx(scene.ttc_s[78], rel=0.15)


def test_repeated_timestamp_does_not_divide_by_zero():
    scene = synth.render_scene(synth.SceneConfig(n_frames=4))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    for gray in scene.frames:
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.0)
    assert all(np.isfinite(v) for v in result.timings_ms.values())


def test_box_leaving_the_frame_is_handled():
    _, results = run_scene(synth.SceneConfig(n_frames=30, z0_m=12.0, closing_speed_mps=3.0, lateral_m=-2.5, lateral_speed_mps=-3.0))
    assert results[-1].objects[0].bbox[0] < 0  # partly outside on the left
    assert first_frame_at(results, Level.WARNING) is None


def test_ttc_is_limited_to_sixteen_tracks():
    boxes = [np.array([10 + 30 * (i % 10), 100 + 60 * (i // 10), 35 + 30 * (i % 10), 150 + 60 * (i // 10)], float) for i in range(20)]
    gray = synth.texture(384, 512, 0)
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(b, "car", 0.9) for b in boxes], DisFlow())
    for i in range(3):
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), i / 30)
    assert len(pipeline.tracker.tracks) == 20 and len(result.objects) == 16
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_pipeline.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'collision_avoidance.fcw.pipeline'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/pipeline.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_pipeline.py -v`
Expected: `7 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/pipeline.py tests/fcw/test_pipeline.py
git commit -m "Add FCW pipeline wiring tracker, ego-rotation, TTC, fusion and warnings"
```

---

### Task 11: Metrics, overlay and command-line tool

After Step 4, also smoke-test the CLI on any dash-cam clip (it loads YOLOv9t through upstream's `Detector`): `COLLISION_WANDB=0 python -m collision_avoidance.fcw --source <clip.mp4> --fx 500 --no-display --max-frames 100`. Expected: a final line `frames 100 | mean ... ms/frame | warnings raised ... | critical frames ...`.

**Files:**
- Create: `collision_avoidance/fcw/metrics.py`
- Create: `collision_avoidance/fcw/render.py`
- Create: `collision_avoidance/fcw/__main__.py`
- Test: `tests/fcw/test_metrics_render.py`

**Interfaces:**
- Consumes: `FcwPipeline`, `FcwFrame`, `DisFlow`, `FcwConfig` (Task 10); `Level` (Task 9).
- Produces: `metrics.median_relative_error`, `coverage`, `first_warning_ttc(levels, gt_ttc, level=Level.WARNING)`, `warned_in_time(first_warning_ttcs, deadline_s=2.0)`, `warning_raises(levels)`, `false_alarms_per_10min(levels, duration_s)`, `variance_scale(records)`, `rms(a, b)`; `render.draw(result, fx) -> image`; `__main__.parse_args(argv)`, `make_detector(weights, conf, device)`, `main(argv)`.

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_metrics_render.py`:

```python
import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import metrics, synth
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.fcw.render import draw
from collision_avoidance.tracking import Detection

N, W, C = Level.NONE, Level.WARNING, Level.CRITICAL


def test_median_relative_error_skips_missing_estimates():
    assert metrics.median_relative_error([1.1, np.nan, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(0.05)
    assert metrics.coverage([1.0, np.nan]) == 0.5


def test_first_warning_and_in_time_share():
    gt = [3.0, 2.8, 2.6, 2.4, 2.2]
    assert metrics.first_warning_ttc([N, N, W, W, C], gt) == 2.6
    assert metrics.first_warning_ttc([N, N, N, N, C], gt, Level.CRITICAL) == 2.2
    assert metrics.first_warning_ttc([N] * 5, gt) is None
    assert metrics.warned_in_time([2.6, 1.9, None, 2.0]) == 0.5


def test_false_alarm_rate_counts_raises_not_frames():
    levels = [N, W, W, W, N, N, C, C, N, W]
    assert metrics.warning_raises(levels) == 3
    assert metrics.false_alarms_per_10min(levels, duration_s=300.0) == pytest.approx(6.0)


def test_variance_scale():
    records = [("horn", 1.2, 0.01, 1.0), ("horn", 0.8, 0.01, 1.0), ("scale", 1.0, 0.04, 1.0)]
    assert metrics.variance_scale(records) == pytest.approx({"horn": 4.0, "scale": 0.0})


def test_render_draws_on_a_copy():
    scene = synth.render_scene(synth.SceneConfig(n_frames=3))
    pipe = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    result = None
    for i, gray in enumerate(scene.frames):
        result = pipe.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), i / 30)
    image = draw(result, 500.0)
    assert image.shape == result.frame.shape
    assert not np.array_equal(image, result.frame)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_metrics_render.py -v`
Expected: FAIL during collection with `ImportError: cannot import name 'metrics' from 'collision_avoidance.fcw'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/metrics.py`:

```python
"""Evaluation metrics for the phase exit criteria (spec section 9)."""

import numpy as np

from .collision import Level


def median_relative_error(estimated, truth) -> float:
    """Median |est - gt| / gt over pairs where both are finite (NaN marks 'no estimate')."""
    est, gt = np.asarray(estimated, float), np.asarray(truth, float)
    ok = np.isfinite(est) & np.isfinite(gt) & (gt > 0)
    return float(np.median(np.abs(est[ok] - gt[ok]) / gt[ok])) if ok.any() else float("nan")


def coverage(estimated) -> float:
    est = np.asarray(estimated, float)
    return float(np.isfinite(est).mean()) if est.size else 0.0


def first_warning_ttc(levels, gt_ttc, level: Level = Level.WARNING) -> float | None:
    """Ground-truth TTC at the first frame whose level reaches `level` (None if never)."""
    for lv, tau in zip(levels, gt_ttc):
        if lv >= level:
            return float(tau)
    return None


def warned_in_time(first_warning_ttcs, deadline_s: float = 2.0) -> float:
    """Share of approach events warned while the true TTC was still >= deadline_s."""
    if not first_warning_ttcs:
        return float("nan")
    return float(np.mean([t is not None and t >= deadline_s for t in first_warning_ttcs]))


def warning_raises(levels) -> int:
    """Number of NONE -> WARNING-or-higher transitions in a per-frame level sequence."""
    lv = np.asarray([int(x) for x in levels])
    return int(((lv[1:] >= Level.WARNING) & (lv[:-1] < Level.WARNING)).sum() + (lv[:1] >= Level.WARNING).sum())


def false_alarms_per_10min(levels, duration_s: float) -> float:
    return warning_raises(levels) / (duration_s / 600.0) if duration_s > 0 else float("nan")


def variance_scale(records) -> dict[str, float]:
    """Per method, mean normalised squared error (eta - eta_gt)^2 / var.

    records: iterable of (method, eta, var, eta_gt). A well-calibrated method
    gives 1; use the result as FusionConfig.var_scale.
    """
    out: dict[str, list[float]] = {}
    for method, eta, var, eta_gt in records:
        if var > 0 and np.isfinite(eta_gt):
            out.setdefault(method, []).append((eta - eta_gt) ** 2 / var)
    return {m: float(np.mean(v)) for m, v in out.items()}


def rms(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")
```

`collision_avoidance/fcw/render.py`:

```python
"""Overlay for the FCW output: boxes by warning level, TTC labels, heading and corridor."""

import cv2
import numpy as np

from .collision import Level
from .pipeline import FcwFrame

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {Level.NONE: (0, 200, 0), Level.WARNING: (0, 165, 255), Level.CRITICAL: (0, 0, 255)}  # BGR
OFF_COURSE = (160, 160, 160)
CAMERA_HEIGHT_M = 1.3   # drawing only: where the corridor meets the image bottom
HALF_CORRIDOR_M = 1.05  # (ego width + margin) / 2


def draw(result: FcwFrame, fx: float) -> np.ndarray:
    image = result.frame.copy()
    h, w = image.shape[:2]
    hx, hy = map(float, result.heading)
    if h - 1 > hy:
        z_bottom = fx * CAMERA_HEIGHT_M / (h - 1 - hy)
        dx = fx * HALF_CORRIDOR_M / z_bottom
        for side in (-1, 1):
            cv2.line(image, (int(hx), int(hy)), (int(hx + side * dx), h - 1), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(image, (int(hx), int(hy)), (255, 255, 0), cv2.MARKER_CROSS, 12, 2)

    for obj in result.objects:
        x1, y1, x2, y2 = map(int, obj.bbox)
        on_course = obj.course is not None and obj.course.on_course
        color = COLORS[obj.level] if on_course or obj.level > Level.NONE else OFF_COURSE
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if obj.level else 1)
        ttc = obj.estimate.ttc_s if obj.estimate else None
        label = f"{obj.class_name} {ttc:.1f}s" if ttc is not None else obj.class_name
        cv2.putText(image, label, (x1, max(12, y1 - 4)), FONT, 0.45, color, 1, cv2.LINE_AA)

    level = result.level
    if level > Level.NONE:
        text = "BRAKE" if level == Level.CRITICAL else "COLLISION WARNING"
        cv2.putText(image, text, (10, 30), FONT, 0.8, COLORS[level], 2, cv2.LINE_AA)
    total = result.timings_ms.get("total", 0.0)
    cv2.putText(image, f"{total:.0f} ms", (w - 70, 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image
```

`collision_avoidance/fcw/__main__.py`:

```python
"""Run the forward-collision-warning pipeline on a video file or webcam.

    python -m collision_avoidance.fcw --source dashcam.mp4 --fx 500
    python -m collision_avoidance.fcw --source 0 --output run.mp4
    python -m collision_avoidance.fcw --source clip.mp4 --no-display --max-frames 300

--fx is the focal length in pixels of the 512x384 processing frame. For a
camera with horizontal field of view F: fx = 256 / tan(F / 2).
Keys (display mode): ESC quit, P pause.
"""

import argparse
import time

import cv2
import numpy as np

from .collision import Level
from .metrics import warning_raises
from .pipeline import DisFlow, FcwConfig, FcwPipeline
from .render import draw


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or webcam index")
    p.add_argument("--fx", type=float, default=500.0, help="focal length (px) at 512x384")
    p.add_argument("--fy", type=float, help="vertical focal length (px) if pixels are not square")
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", help="write the annotated video here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args(argv)


def make_detector(weights: str, conf: float, device: str):
    from ..detection import Detector  # imports ultralytics and torch
    from ..flow import resolve_device

    return Detector(weights, conf, resolve_device(device))


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = FcwConfig(fx=args.fx, fy=args.fy)
    live = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source '{args.source}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = FcwPipeline(cfg, make_detector(args.yolo_weights, args.conf, args.device), DisFlow())
    writer, levels, totals = None, [], []
    start = time.perf_counter()
    try:
        while not args.max_frames or len(levels) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - start if live else len(levels) / fps
            result = pipeline.process(frame, t)
            levels.append(result.level)
            totals.append(result.timings_ms["total"])
            if args.no_display and not args.output:
                continue
            image = draw(result, cfg.fx)
            if args.output:
                if writer is None:
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, image.shape[1::-1])
                writer.write(image)
            if not args.no_display:
                cv2.imshow("Forward collision warning", image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("p"):
                    cv2.waitKey(0)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"frames {len(levels)} | mean {np.mean(totals[1:]) if len(totals) > 1 else 0:.1f} ms/frame | "
          f"warnings raised {warning_raises(levels)} | critical frames {sum(lv == Level.CRITICAL for lv in levels)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_metrics_render.py -v`
Expected: `5 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/metrics.py collision_avoidance/fcw/render.py collision_avoidance/fcw/__main__.py tests/fcw/test_metrics_render.py
git commit -m "Add FCW metrics, overlay and CLI"
```

---

### Task 12: KITTI helpers and evaluation scripts

The scripts need datasets, so only the KITTI parsing helpers get unit tests. After Step 4, check that the scripts at least parse their arguments: `python eval/kitti_yaw.py --help`, `python eval/evttc_fcw.py --help`, `python eval/false_alarms.py --help` (each prints usage).

**Files:**
- Create: `collision_avoidance/fcw/kitti.py`
- Create: `eval/kitti_yaw.py`
- Create: `eval/evttc_fcw.py`
- Create: `eval/false_alarms.py`
- Test: `tests/fcw/test_kitti.py`

**Interfaces:**
- Consumes: `metrics` (Task 11), `FcwPipeline`/`FcwConfig`/`DisFlow` (Task 10), `EgoRotationEstimator` (Task 4); upstream `ttc_esn.evttc` (`load_gt`, `load_annotations`, `video_fps`, `estimate_sync_offset`, `iter_left_frames`, `LEFT_RGB_FX`) and `collision_avoidance.telemetry.start_run`.
- Produces: `kitti.parse_timestamps(path)`, `camera_matrix(calib_path, camera="02")`, `yaw_rates(oxts_dir)`, `load_drive(drive_dir)`; three runnable scripts (Task 13 runs them).

- [ ] **Step 1: Write the failing test**

`tests/fcw/test_kitti.py`:

```python
import numpy as np
import pytest

from collision_avoidance.fcw import kitti


def test_timestamps_are_relative_seconds(tmp_path):
    f = tmp_path / "timestamps.txt"
    f.write_text("2011-09-26 13:02:25.964389698\n2011-09-26 13:02:26.068550587\n2011-09-26 13:03:00.000000000\n")
    np.testing.assert_allclose(kitti.parse_timestamps(f), [0.0, 0.104160889, 34.035610302], atol=1e-6)


def test_camera_matrix_reads_p_rect_02(tmp_path):
    f = tmp_path / "calib_cam_to_cam.txt"
    f.write_text("calib_time: x\nP_rect_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                 "P_rect_02: 7.215377e+02 0 6.095593e+02 4.485728e+01 0 7.215377e+02 1.728540e+02 2.163791e-01 0 0 1 2.745884e-03\n")
    K = kitti.camera_matrix(f)
    assert K[0, 0] == pytest.approx(721.5377) and K[0, 2] == pytest.approx(609.5593) and K[1, 2] == pytest.approx(172.854)


def test_yaw_rate_is_field_22(tmp_path):
    (tmp_path / "data").mkdir()
    for i, wu in enumerate((0.01, -0.02)):
        values = np.zeros(30)
        values[22] = wu
        (tmp_path / "data" / f"{i:010d}.txt").write_text(" ".join(map(str, values)))
    np.testing.assert_allclose(kitti.yaw_rates(tmp_path), [0.01, -0.02])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/fcw/test_kitti.py -v`
Expected: FAIL during collection with `ImportError: cannot import name 'kitti' from 'collision_avoidance.fcw'`

- [ ] **Step 3: Write the implementation**

`collision_avoidance/fcw/kitti.py`:

```python
"""KITTI raw helpers: timestamps, rectified camera matrix and OXTS yaw rate.

Layout (from the KITTI raw download):
    <date>/calib_cam_to_cam.txt
    <date>/<date>_drive_NNNN_sync/image_02/data/*.png, image_02/timestamps.txt
    <date>/<date>_drive_NNNN_sync/oxts/data/*.txt (30 values per line)
OXTS field 22 is wu, the angular rate about the vehicle's upward axis (rad/s,
positive turning left). With this package's convention (R maps camera
coordinates prev -> curr, camera y pointing down) a left turn gives a
positive rotation about camera y, so wu compares directly with rvec[1] / dt.
"""

from pathlib import Path

import numpy as np

OXTS_WU = 22


def parse_timestamps(path: Path) -> np.ndarray:
    """Seconds since the first line, from lines like '2011-09-26 13:02:25.964389698'."""
    seconds = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            h, m, s = line.split()[1].split(":")
            seconds.append(int(h) * 3600 + int(m) * 60 + float(s))
    t = np.array(seconds)
    return t - t[0]


def camera_matrix(calib_path: Path, camera: str = "02") -> np.ndarray:
    for line in Path(calib_path).read_text().split("\n"):
        if line.startswith(f"P_rect_{camera}:"):
            return np.array(line.split(":")[1].split(), float).reshape(3, 4)[:, :3]
    raise ValueError(f"P_rect_{camera} not found in {calib_path}")


def yaw_rates(oxts_dir: Path) -> np.ndarray:
    files = sorted(Path(oxts_dir, "data").glob("*.txt"))
    return np.array([float(f.read_text().split()[OXTS_WU]) for f in files])


def load_drive(drive_dir: Path):
    """(image paths, image times [s], K, OXTS yaw rate per frame [rad/s])."""
    drive_dir = Path(drive_dir)
    images = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    times = parse_timestamps(drive_dir / "image_02" / "timestamps.txt")
    K = camera_matrix(drive_dir.parent / "calib_cam_to_cam.txt")
    return images, times, K, yaw_rates(drive_dir / "oxts")
```

`eval/kitti_yaw.py`:

```python
"""Ego-rotation accuracy: estimated yaw rate vs KITTI raw OXTS (P1 exit: RMS < 1 deg/s).

    python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_0005_sync [more drives...]

Runs at the native KITTI resolution with the rectified camera matrix; flow is
OpenCV DIS (the PC stand-in for the board's dense LK).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from collision_avoidance.fcw import kitti
from collision_avoidance.fcw.ego_rotation import EgoRotationEstimator
from collision_avoidance.fcw.metrics import rms
from collision_avoidance.fcw.pipeline import DisFlow
from collision_avoidance.telemetry import start_run


def evaluate_drive(drive_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(estimated, ground-truth) yaw rates in deg/s for every frame pair with a valid estimate."""
    images, times, K, wu = kitti.load_drive(drive_dir)
    flow_source, estimator = DisFlow(), EgoRotationEstimator(K)
    est, gt = [], []
    prev = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    for i in range(1, len(images)):
        curr = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
        ego = estimator.update(flow_source(prev, curr), [])
        dt = times[i] - times[i - 1]
        if ego.valid and dt > 0:
            est.append(np.degrees(ego.rvec[1] / dt))
            gt.append(np.degrees(0.5 * (wu[i] + wu[i - 1])))
        prev = curr
    return np.array(est), np.array(gt)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("drives", nargs="+", type=Path)
    args = p.parse_args(argv)
    run = start_run("fcw-eval-kitti-yaw", {"drives": [d.name for d in args.drives]}, tags=["fcw", "eval", "kitti"])
    all_est, all_gt = [], []
    for drive in args.drives:
        est, gt = evaluate_drive(drive)
        all_est.append(est)
        all_gt.append(gt)
        print(f"{drive.name:40s} pairs={len(est):5d} rms={rms(est, gt):.3f} deg/s")
    total = rms(np.concatenate(all_est), np.concatenate(all_gt))
    print(f"{'overall':40s} rms={total:.3f} deg/s  (P1 exit: < 1.0)")
    if run is not None:
        run.summary.update({"yaw_rms_deg_s": total})
        run.finish()


if __name__ == "__main__":
    main()
```

`eval/evttc_fcw.py`:

```python
"""TTC accuracy, warning lead time and variance calibration on EvTTC car-rear approaches.

    python -m ttc_esn.evttc --root data/evttc          # download once (upstream tool)
    python eval/evttc_fcw.py --data data/evttc

P1 exits: fused median relative TTC error <= 20 %; >= 90 % of approach events
warned while ground-truth TTC >= 2.0 s. Also prints FusionConfig.var_scale
values calibrated from the per-method errors.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from collision_avoidance.fcw import metrics
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.measurement import METHODS
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.telemetry import start_run
from collision_avoidance.tracking import iou
from ttc_esn import evttc

PANEL_W, PANEL_H = 1920, 1200
MIN_TARGET_IOU = 0.3


def config_for_evttc() -> FcwConfig:
    """EvTTC's 1920x1200 left panel resized to 512x384 makes pixels non-square."""
    return FcwConfig(fx=evttc.LEFT_RGB_FX * 512 / PANEL_W, fy=evttc.LEFT_RGB_FX * 384 / PANEL_H)


def target_object(result, box_512):
    scored = [(iou(o.bbox, box_512), o) for o in result.objects]
    best = max(scored, key=lambda s: s[0], default=(0.0, None))
    return best[1] if best[0] >= MIN_TARGET_IOU else None


def evaluate_sequence(seq_dir: Path, detector, warmup: int = 20) -> dict:
    gt, annotations = evttc.load_gt(seq_dir), evttc.load_annotations(seq_dir)
    fps = evttc.video_fps(seq_dir / "video.mp4")
    sync = evttc.estimate_sync_offset(annotations, gt, fps)
    first = math.ceil((gt.t.iloc[0] - sync.offset_s) * fps)
    last = math.floor((gt.t.iloc[-1] - sync.offset_s) * fps)
    scale = np.array([512 / PANEL_W, 384 / PANEL_H] * 2)
    anchors = np.array(sorted(annotations))
    pipeline = FcwPipeline(config_for_evttc(), detector, DisFlow())

    rows = []
    for index, panel in evttc.iter_left_frames(seq_dir / "video.mp4"):
        if index < first - warmup:
            continue
        if index > last:
            break
        result = pipeline.process(panel, index / fps)
        if index < first:
            continue
        nearest = int(anchors[np.abs(anchors - index).argmin()])
        obj = target_object(result, annotations[nearest] * scale)
        gt_ttc = float(np.interp(index / fps + sync.offset_s, gt.t, gt.ttc))
        rows.append({
            "gt_ttc": gt_ttc,
            "fused": obj.estimate.ttc_s if obj and obj.estimate and obj.estimate.ttc_s else np.nan,
            "level": obj.level if obj else Level.NONE,
            "measurements": [] if obj is None else [(m.method, m.eta, m.var) for m in obj.measurements],
        })
    return {"name": seq_dir.name, "rows": rows}


def summarise(results: list[dict]) -> dict:
    gt = np.array([r["gt_ttc"] for res in results for r in res["rows"]])
    fused = np.array([r["fused"] for res in results for r in res["rows"]])
    summary = {"fused_median_rel_err": metrics.median_relative_error(fused, gt), "fused_coverage": metrics.coverage(fused)}
    records = []
    for method in METHODS:
        per = np.full(len(gt), np.nan)
        k = 0
        for res in results:
            for r in res["rows"]:
                for name, eta, var in r["measurements"]:
                    if name == method:
                        per[k] = 1.0 / eta if eta > 0 else np.nan
                        records.append((name, eta, var, 1.0 / r["gt_ttc"]))
                k += 1
        summary[f"{method}_median_rel_err"] = metrics.median_relative_error(per, gt)
        summary[f"{method}_coverage"] = metrics.coverage(per)
    summary["var_scale"] = metrics.variance_scale(records)
    events = [metrics.first_warning_ttc([r["level"] for r in res["rows"]], [r["gt_ttc"] for r in res["rows"]])
              for res in results if min(r["gt_ttc"] for r in res["rows"]) < 2.0]
    summary["warned_by_2s"] = metrics.warned_in_time(events)
    summary["approach_events"] = len(events)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/evttc"))
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from collision_avoidance.fcw.__main__ import make_detector

    detector = make_detector(args.yolo_weights, 0.5, args.device)
    seq_dirs = sorted(d for d in args.data.iterdir() if (d / "video.mp4").exists() and (d / "gt_ttc.csv").exists())
    run = start_run("fcw-eval-evttc", {"sequences": [d.name for d in seq_dirs]}, tags=["fcw", "eval", "evttc"])
    results = [evaluate_sequence(d, detector) for d in seq_dirs]
    for res in results:
        one = summarise([res])
        print(f"{res['name']:24s} fused_err={one['fused_median_rel_err']:.3f} coverage={one['fused_coverage']:.2f}")
    summary = summarise(results)
    print(json.dumps(summary, indent=2))
    print("P1 exits: fused_median_rel_err <= 0.20, warned_by_2s >= 0.90")
    if run is not None:
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        run.finish()


if __name__ == "__main__":
    main()
```

`eval/false_alarms.py`:

```python
"""False-alarm rate on normal driving (P1 exit: < 1 warning raise per 10 minutes).

    python eval/false_alarms.py --videos data/normal_driving --fx 500

Every video in the folder must be free of real near-collisions; each
NONE -> WARNING transition then counts as a false alarm.
"""

import argparse
from pathlib import Path

import cv2

from collision_avoidance.fcw.metrics import warning_raises
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.telemetry import start_run

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def count_raises(video: Path, cfg: FcwConfig, detector) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline, levels = FcwPipeline(cfg, detector, DisFlow()), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        levels.append(pipeline.process(frame, len(levels) / fps).level)
    cap.release()
    return warning_raises(levels), len(levels) / fps


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--fx", type=float, default=500.0)
    p.add_argument("--yolo-weights", default="yolov9t.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from collision_avoidance.fcw.__main__ import make_detector

    detector = make_detector(args.yolo_weights, 0.5, args.device)
    videos = sorted(v for v in args.videos.iterdir() if v.suffix.lower() in VIDEO_SUFFIXES)
    run = start_run("fcw-eval-false-alarms", {"videos": [v.name for v in videos], "fx": args.fx}, tags=["fcw", "eval"])
    total_raises, total_s = 0, 0.0
    for video in videos:
        raises, seconds = count_raises(video, FcwConfig(fx=args.fx), detector)
        total_raises, total_s = total_raises + raises, total_s + seconds
        print(f"{video.name:40s} {seconds:7.1f} s  raises={raises}")
    rate = total_raises / (total_s / 600.0) if total_s else float("nan")
    print(f"total {total_s / 60:.1f} min, {total_raises} raises -> {rate:.2f} per 10 min  (P1 exit: < 1)")
    if run is not None:
        run.summary.update({"false_alarms_per_10min": rate, "minutes": total_s / 60})
        run.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/fcw/test_kitti.py -v`
Expected: `3 passed`

Then run the whole new suite: `pytest tests/fcw -q` (all green).

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/kitti.py eval/kitti_yaw.py eval/evttc_fcw.py eval/false_alarms.py tests/fcw/test_kitti.py
git commit -m "Add KITTI helpers and FCW evaluation scripts"
```

---

### Task 13: Run the P1 PC evaluations and calibrate the fusion

This task produces numbers, not features. It closes the PC half of the P1 exit criteria; the on-board half is plan 2.

**Files:**
- Create: `docs/superpowers/reports/2026-10-p1-pc-evaluation.md`
- Modify: `collision_avoidance/fcw/fusion.py` (the `var_scale` default only)

**Interfaces:**
- Consumes: the three scripts from Task 12.
- Produces: calibrated `FusionConfig.var_scale` defaults; a report the user can quote.

- [ ] **Step 1: Get the data**

- EvTTC (upstream downloader): `python -m ttc_esn.evttc --root data/evttc`
- KITTI raw: download at least `2011_09_26_drive_0005_sync`, `_0014_sync` and `_0091_sync` with their `2011_09_26_calib.zip` from https://www.cvlibs.net/datasets/kitti/raw_data.php, unpacked to `data/kitti/2011_09_26/`.
- Normal driving: put at least 30 minutes of dash-cam clips without near-collisions in `data/normal_driving/`. Upstream already uses Nexar dash-cam clips; any similar footage works.

- [ ] **Step 2: Run the three evaluations**

```bash
export COLLISION_WANDB=0   # or leave W&B on; upstream logs every run there
python eval/kitti_yaw.py data/kitti/2011_09_26/2011_09_26_drive_0005_sync data/kitti/2011_09_26/2011_09_26_drive_0014_sync data/kitti/2011_09_26/2011_09_26_drive_0091_sync
python eval/evttc_fcw.py --data data/evttc
python eval/false_alarms.py --videos data/normal_driving --fx 500
```

Expected output lines to copy into the report:
- `overall  rms=... deg/s  (P1 exit: < 1.0)`
- the JSON summary from `evttc_fcw.py`, with `fused_median_rel_err`, per-method errors, `var_scale` and `warned_by_2s`
- `total ... min, ... raises -> ... per 10 min  (P1 exit: < 1)`

- [ ] **Step 3: Calibrate the fusion**

Replace the `var_scale` default in `FusionConfig` with the `var_scale` values printed by `evttc_fcw.py`, rounded to 2 decimals, keeping every method key. For example, if it printed `{"looming": 0.8, "scale": 1.7, "horn": 6.2, "divergence": 3.1}`:

```python
    var_scale: dict = field(default_factory=lambda: {"looming": 0.8, "scale": 1.7, "horn": 6.2, "divergence": 3.1})
```

Then rerun `python eval/evttc_fcw.py --data data/evttc` and `pytest tests/fcw -q`. Expected: tests green; `fused_median_rel_err` no worse than before calibration.

- [ ] **Step 4: Write the report**

`docs/superpowers/reports/2026-10-p1-pc-evaluation.md` gets one table per exit criterion with the value, the target and pass/fail, plus the before/after calibration numbers. Name the datasets and the commit hash it was measured at (`git rev-parse --short HEAD`). Any criterion that fails is listed with the measured value, not rounded toward the target.

- [ ] **Step 5: Commit**

```bash
git add collision_avoidance/fcw/fusion.py docs/superpowers/reports/2026-10-p1-pc-evaluation.md
git commit -m "Calibrate FCW fusion on EvTTC and record P1 PC evaluation"
```
