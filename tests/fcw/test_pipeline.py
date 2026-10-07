import cv2
import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.collision import Level
from collision_avoidance.fcw.pipeline import DisFlow, FcwConfig, FcwPipeline
from collision_avoidance.tracking import Detection


def run_scene(scene_cfg: synth.SceneConfig, class_name: str = "car"):
    scene = synth.render_scene(scene_cfg)
    frames = iter(range(len(scene.frames)))
    current = {"i": 0}

    def detector(_frame):
        return [Detection(scene.boxes[current["i"]], class_name, 0.9)]

    pipeline = FcwPipeline(FcwConfig(fx=scene_cfg.fx), detector, DisFlow())
    results = []
    for i in frames:
        current["i"] = i
        results.append(pipeline.process(cv2.cvtColor(scene.frames[i], cv2.COLOR_GRAY2BGR), i / scene_cfg.fps))
    return scene, results


def first_frame_at(results, level):
    return next((r.index for r in results if r.level >= level), None)


def test_head_on_approach_warns_in_time():
    scene, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    warn = first_frame_at(results, Level.WARNING)
    crit = first_frame_at(results, Level.CRITICAL)
    assert warn is not None and scene.ttc_s[warn] >= 2.0          # NHTSA: warned before TTC 2.0 s
    assert scene.ttc_s[warn] <= 3.0                                # and not absurdly early
    assert crit is not None and scene.ttc_s[crit] >= 1.0
    final = results[-1].objects[0].estimate
    assert final.ttc_s == pytest.approx(scene.ttc_s[-1], rel=0.15)


def test_car_in_the_next_lane_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0, lateral_m=3.5))
    assert first_frame_at(results, Level.WARNING) is None


def test_static_object_never_warns():
    _, results = run_scene(synth.SceneConfig(n_frames=40, z0_m=20.0, closing_speed_mps=0.0))
    assert first_frame_at(results, Level.WARNING) is None
    assert results[-1].objects[0].estimate.ttc_s is None


def test_dropped_frames_use_real_timestamps():
    scene = synth.render_scene(synth.SceneConfig(n_frames=80, z0_m=50.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    for current in range(0, 80, 2):  # every other frame arrives
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
    assert result.objects[0].estimate.ttc_s == pytest.approx(scene.ttc_s[78], rel=0.15)


def test_repeated_timestamp_does_not_divide_by_zero():
    scene = synth.render_scene(synth.SceneConfig(n_frames=4))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    for gray in scene.frames:
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.0)
    assert all(np.isfinite(v) for v in result.timings_ms.values())


def test_box_leaving_the_frame_is_handled():
    _, results = run_scene(synth.SceneConfig(n_frames=30, z0_m=12.0, closing_speed_mps=3.0, lateral_m=-2.5, lateral_speed_mps=-3.0))
    assert results[-1].objects[0].bbox[0] < 0  # partly outside on the left
    assert first_frame_at(results, Level.WARNING) is None


def test_ttc_is_limited_to_sixteen_tracks():
    boxes = [np.array([10 + 30 * (i % 10), 100 + 60 * (i // 10), 35 + 30 * (i % 10), 150 + 60 * (i // 10)], float) for i in range(20)]
    gray = synth.texture(384, 512, 0)
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(b, "car", 0.9) for b in boxes], DisFlow())
    for i in range(3):
        result = pipeline.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), i / 30)
    assert len(pipeline.tracker.tracks) == 20 and len(result.objects) == 16


def test_scale_search_uses_the_real_time_span_when_frames_arrive_irregularly():
    # Every third frame is lost: gaps alternate between 1/30 s and 2/30 s, so k * (latest dt) is not the span.
    scene = synth.render_scene(synth.SceneConfig(n_frames=60, z0_m=40.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)], DisFlow())
    errors = []
    for current in [i for i in range(60) if i % 3 != 2]:
        result = pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30)
        scale = [m for m in result.objects[0].measurements if m.method == "scale"]
        if current >= 30 and scale:
            errors.append(abs(scale[0].ttc_s - scene.ttc_s[current]) / scene.ttc_s[current])
    assert errors and max(errors) < 0.1


def test_out_of_order_timestamp_does_not_inflate_the_next_step():
    scene = synth.render_scene(synth.SceneConfig(n_frames=4))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[0], "car", 0.9)], DisFlow())
    times = [0.0, 1 / 30, 0.0, 2 / 30]  # the third frame arrives with a stale timestamp
    results = [pipeline.process(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), t) for g, t in zip(scene.frames, times)]
    assert results[2].dt_s == pytest.approx(1 / 30)   # fallback to the nominal period
    assert results[3].dt_s == pytest.approx(1 / 30)   # measured from 1/30, not from the stale 0.0


def test_lost_track_cannot_escalate_or_hold_a_warning():
    # The object is detected until true TTC is ~2 s, then the detector loses it.
    scene = synth.render_scene(synth.SceneConfig(n_frames=70, z0_m=50.0, closing_speed_mps=15.0))
    pipeline = FcwPipeline(FcwConfig(), lambda f: [Detection(scene.boxes[current], "car", 0.9)] if current < 40 else [], DisFlow())
    levels = []
    for current in range(70):
        levels.append(pipeline.process(cv2.cvtColor(scene.frames[current], cv2.COLOR_GRAY2BGR), current / 30).level)
    assert Level.CRITICAL not in levels[40:]   # no escalation from extrapolated predictions
    assert levels[-1] == Level.NONE            # cleared while the track is lost, before it is deleted
