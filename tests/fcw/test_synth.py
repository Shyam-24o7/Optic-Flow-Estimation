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
