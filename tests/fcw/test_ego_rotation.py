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


def test_pose_from_an_inlier_subset_matches_the_full_set():
    # recoverPose only picks among four decompositions of E; a subset of inliers decides it as well.
    from collision_avoidance.fcw.ego_rotation import EgoRotationConfig
    R, t = synth.rotation_y(np.radians(0.6)), np.array([0.0, 0.0, -0.5])
    flow = dense_flow(R, t, road_depth)
    full = EgoRotationEstimator(K, EgoRotationConfig(pose_points=0)).update(flow, [])
    fast = EgoRotationEstimator(K, EgoRotationConfig(pose_points=64)).update(flow, [])
    assert angle_deg(full.R @ fast.R.T) < 0.01
    np.testing.assert_allclose(fast.foe, full.foe, atol=0.5)
