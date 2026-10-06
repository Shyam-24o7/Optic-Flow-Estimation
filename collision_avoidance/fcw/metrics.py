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
