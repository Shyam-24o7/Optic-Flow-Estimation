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
