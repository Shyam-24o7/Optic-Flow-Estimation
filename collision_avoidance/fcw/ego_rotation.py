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
    foe_smoothing: float = 0.05       # EMA weight of the newest FOE (~0.7 s): follows bends, not per-frame jitter


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
