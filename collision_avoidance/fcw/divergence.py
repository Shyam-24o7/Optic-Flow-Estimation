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


_LCG_MASK = (1 << 64) - 1


def ransac_samples(n: int, iterations: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """Index triples from a fixed 64-bit LCG (Knuth MMIX constants).

    Deliberately simple so the C++ port and the HLS engine draw exactly the same
    samples as this reference: numpy's generator cannot be reproduced there.
    """
    state = (seed ^ 0x853C49E6748FEA9B) & _LCG_MASK
    out = []
    for _ in range(iterations):
        triple: list[int] = []
        while len(triple) < 3:
            state = (state * 6364136223846793005 + 1442695040888963407) & _LCG_MASK
            value = (state >> 33) % n
            if value not in triple:
                triple.append(value)
        out.append(tuple(triple))
    return out


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
    best, best_count = None, -1
    for sample in ransac_samples(len(X), cfg.ransac_iterations):
        idx = list(sample)
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
