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
