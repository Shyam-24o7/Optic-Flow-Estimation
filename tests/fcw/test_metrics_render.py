import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import metrics, synth
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.fcw.render import draw
from collision_avoidance.tracking import Detection

N, W, C = Level.NONE, Level.WARNING, Level.CRITICAL


def test_median_relative_error_skips_missing_estimates():
    assert metrics.median_relative_error([1.1, np.nan, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(0.05)
    assert metrics.coverage([1.0, np.nan]) == 0.5


def test_first_warning_and_in_time_share():
    gt = [3.0, 2.8, 2.6, 2.4, 2.2]
    assert metrics.first_warning_ttc([N, N, W, W, C], gt) == 2.6
    assert metrics.first_warning_ttc([N, N, N, N, C], gt, Level.CRITICAL) == 2.2
    assert metrics.first_warning_ttc([N] * 5, gt) is None
    assert metrics.warned_in_time([2.6, 1.9, None, 2.0]) == 0.5


def test_false_alarm_rate_counts_raises_not_frames():
    levels = [N, W, W, W, N, N, C, C, N, W]
    assert metrics.warning_raises(levels) == 3
    assert metrics.false_alarms_per_10min(levels, duration_s=300.0) == pytest.approx(6.0)


def test_variance_scale():
    records = [("horn", 1.2, 0.01, 1.0), ("horn", 0.8, 0.01, 1.0), ("scale", 1.0, 0.04, 1.0)]
    assert metrics.variance_scale(records) == pytest.approx({"horn": 4.0, "scale": 0.0})


def test_render_draws_on_a_copy():
    scene = synth.render_scene(synth.SceneConfig(n_frames=3))
    pipe = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    result = None
    for i, gray in enumerate(scene.frames):
        result = pipe.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), i / 30)
    image = draw(result, 500.0)
    assert image.shape == result.frame.shape
    assert not np.array_equal(image, result.frame)
