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
