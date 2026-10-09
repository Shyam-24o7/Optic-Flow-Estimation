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


def test_ground_plane_moves_only_when_we_drive():
    still = synth.render_scene(synth.SceneConfig(n_frames=3, ground=True, ego_speed_mps=0.0, closing_speed_mps=0.0))
    moving = synth.render_scene(synth.SceneConfig(n_frames=3, ground=True, ego_speed_mps=10.0, closing_speed_mps=10.0))
    road = (slice(300, 384), slice(0, 120))          # below the horizon, left of the object
    assert np.array_equal(still.frames[0][road], still.frames[2][road])
    assert not np.array_equal(moving.frames[0][road], moving.frames[2][road])
    sky = (slice(0, 150), slice(0, 512))              # above the horizon: the far background only
    assert np.abs(moving.frames[0][sky].astype(int) - moving.frames[2][sky].astype(int)).max() <= 2
