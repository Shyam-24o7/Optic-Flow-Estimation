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


def test_ransac_samples_are_reproducible_across_languages():
    # Fixed 64-bit LCG so the C++ port (and HLS) draw exactly the same RANSAC samples.
    from collision_avoidance.fcw.divergence import ransac_samples
    samples = ransac_samples(1000, 4, seed=0)
    assert samples == ransac_samples(1000, 4, seed=0)
    assert all(len(set(s)) == 3 and all(0 <= i < 1000 for i in s) for s in samples)
    assert samples[0] == (502, 397, 989)   # pinned: the C++ test checks the same first triple
