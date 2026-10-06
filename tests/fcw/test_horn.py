import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.horn import HornConfig, choose_level, gradients, horn_at_level, horn_solve, horn_sums

DT = 1 / 30
PP = (256, 192)


def horn_ttc(scene, t):
    sums, n = horn_sums(scene.frames[t - 1], scene.frames[t], scene.boxes[t], PP)
    return horn_solve(sums, n, DT)


def test_recovers_ttc_of_approaching_object():
    scene = synth.render_scene(synth.SceneConfig(n_frames=21))
    m = horn_ttc(scene, 20)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[20], rel=0.1)


def test_sideways_motion_needs_a_coarser_level():
    # 1.5 m/s sideways at 11 m is ~2.3 px/frame: too much for level 0, fine at level 2.
    scene = synth.render_scene(synth.SceneConfig(n_frames=21, z0_m=15.0, closing_speed_mps=6.0, lateral_m=-1.0, lateral_speed_mps=1.5))
    box = scene.boxes[20]
    speed = abs((box[0] + box[2]) - (scene.boxes[19][0] + scene.boxes[19][2])) / 2 / DT
    level = choose_level(1 / scene.ttc_s[20], box, speed, DT)
    assert level == 2
    m = horn_at_level(scene.frames[19], scene.frames[20], box, PP, level, DT)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[20], rel=0.15)


def test_textureless_box_is_rejected():
    flat = np.full((384, 512), 128, np.uint8)
    found = horn_sums(flat, flat, (100, 100, 200, 200), PP)
    assert found is not None
    assert horn_solve(*found, DT) is None


def test_integer_sums_match_a_pixel_loop():
    """Pins the exact integer definition the HLS engine must reproduce."""
    rng = np.random.default_rng(3)
    prev = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    curr = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    box, pp = (10, 8, 30, 28), (25, 20)
    cfg = HornConfig(shrink=0.0, grad_threshold_l1=0)
    sums, n = horn_sums(prev, curr, box, pp, cfg)

    ex, ey, et = gradients(prev, curr)
    expected = np.zeros(10, np.int64)
    count = 0
    for y in range(8, 28):
        for x in range(10, 30):
            gx, gy, gt = int(ex[y, x]), int(ey[y, x]), int(et[y, x])
            if abs(gx) + abs(gy) <= 0:
                continue
            g = (x - 25) * gx + (y - 20) * gy
            expected += np.array([gx * gx, gx * gy, gx * g, gy * gy, gy * g, g * g, gx * gt, gy * gt, g * gt, gt * gt])
            count += 1
    assert sums.dtype == np.int64 and n == count
    np.testing.assert_array_equal(sums, expected)


def test_level_choice_keeps_motion_under_a_pixel():
    box = (0, 0, 120, 90)  # half-diagonal 75 px
    assert choose_level(None, box, 0.0, DT) == 0
    assert choose_level(1.0, box, 0.0, DT) == 2   # 75 * 1/30 = 2.5 px -> /4 at level 2
    assert choose_level(0.2, box, 30.0, DT) == 1  # 0.5 + 1.0 = 1.5 px -> /2 at level 1
