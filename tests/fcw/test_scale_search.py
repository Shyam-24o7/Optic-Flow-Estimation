import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.scale_search import choose_gap, scale_ttc

DT = 1 / 30


@pytest.mark.parametrize("k", [2, 4, 8])
def test_recovers_ttc_of_approaching_object(k):
    scene = synth.render_scene(synth.SceneConfig(n_frames=25))
    t = 24
    m = scale_ttc(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k], k, DT)
    assert m is not None
    assert m.ttc_s == pytest.approx(scene.ttc_s[t], rel=0.05)


def test_static_object_has_no_expansion():
    scene = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0))
    m = scale_ttc(scene.frames[8], scene.frames[0], scene.boxes[8], scene.boxes[0], 8, DT)
    assert m is not None and abs(m.eta) < 0.05


def test_sideways_motion_does_not_fake_an_approach():
    scene = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0, lateral_speed_mps=4.0))
    m = scale_ttc(scene.frames[8], scene.frames[0], scene.boxes[8], scene.boxes[0], 8, DT)
    assert m is not None and abs(m.eta) < 0.05


def test_tiny_box_is_rejected():
    scene = synth.render_scene(synth.SceneConfig(n_frames=5, z0_m=200.0))
    assert scale_ttc(scene.frames[4], scene.frames[0], scene.boxes[4], scene.boxes[0], 4, DT) is None


def test_gap_choice():
    assert choose_gap(None, DT, available=8) == 8
    assert choose_gap(1.0, DT, available=8) == 1     # 1/s * 1/30 s = 3.3 % >= 2 %
    assert choose_gap(0.25, DT, available=8) == 4    # 2 frames: 1.7 %, 4 frames: 3.3 %
    assert choose_gap(0.25, DT, available=3) == 2    # largest gap the history allows
    assert choose_gap(0.25, DT, available=0) == 0
