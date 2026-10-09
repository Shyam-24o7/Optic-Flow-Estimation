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
    ground: bool = False               # textured road plane below the horizon
    ego_speed_mps: float = 0.0         # our forward speed: moves the road (objects use closing_speed)
    camera_height_m: float = 1.05      # road at Y = camera_height (object bottom with the defaults)


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

    ground_tex = texture(2048, 512, cfg.seed + 2, cell=6)
    g_half_width, g_length = 15.0, 150.0
    g_scale = (2 * g_half_width / ground_tex.shape[1], g_length / ground_tex.shape[0])

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
        if cfg.ground:
            image = _composite_ground(image, ground_tex, g_scale, g_half_width, K, R, cfg, cfg.ego_speed_mps * t, size)
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


def _composite_ground(image, tex, scale, half_width, K, R, cfg, travelled_m, size):
    """Road plane Y = camera_height; texture row v is at world Z = v * scale_z, seen from Z = travelled_m."""
    near = 1.0
    v_min = int(np.ceil((travelled_m + near) / scale[1]))   # rows in front of the camera only
    part = tex[v_min:]
    M = np.array([[scale[0], 0.0, -half_width], [0.0, 0.0, cfg.camera_height_m], [0.0, scale[1], scale[1] * v_min - travelled_m]])
    H = K @ R.T @ M
    road = cv2.warpPerspective(part, H, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(part.shape, 255, np.uint8), H, size, flags=cv2.INTER_LINEAR)
    alpha = mask.astype(np.float32) / 255.0
    return (alpha * road + (1 - alpha) * image).round().astype(np.uint8)


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
