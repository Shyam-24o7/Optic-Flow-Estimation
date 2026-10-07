import numpy as np
import pytest

from collision_avoidance.fcw import synth
from collision_avoidance.fcw.looming import looming
from collision_avoidance.fcw.tracker import ScaleTracker, box_to_z, z_to_box
from collision_avoidance.tracking import Detection

DT = 1 / 30


def det(box, name="car"):
    return Detection(np.asarray(box, float), name, 0.9)


def test_box_state_round_trip():
    box = np.array([100.0, 50.0, 180.0, 110.0])
    np.testing.assert_allclose(z_to_box(box_to_z(box)), box)


def test_scale_rate_converges_to_inverse_ttc():
    scene = synth.render_scene(synth.SceneConfig(n_frames=40, z0_m=40.0))
    tracker = ScaleTracker()
    for i, box in enumerate(scene.boxes):
        tracks = tracker.update([det(box)], DT, i)
    (track,) = tracks.values()
    assert track.scale_rate == pytest.approx(1 / scene.ttc_s[-1], rel=0.1)
    m = looming(track)
    assert m is not None and m.ttc_s == pytest.approx(scene.ttc_s[-1], rel=0.1)


def test_each_track_predicts_from_its_own_state():
    # Regression for hls/ports/object_tracker.hpp, which predicted from the wrong track.
    tracker = ScaleTracker()
    for i in range(10):
        tracker.update([det([10 + 2 * i, 10, 50 + 2 * i, 40]), det([300, 200 - i, 400, 260 - i], "truck")], DT, i)
    small, large = tracker.tracks[0], tracker.tracks[1]
    assert small.class_name == "car" and large.class_name == "truck"
    np.testing.assert_allclose(small.bbox, [28, 10, 68, 40], atol=1.0)
    np.testing.assert_allclose(large.bbox, [300, 191, 400, 251], atol=1.0)


def test_lost_tracks_expire_and_history_is_kept():
    tracker = ScaleTracker()
    tracker.update([det([10, 10, 50, 50])], DT, 0)
    for i in range(1, 33):
        tracker.update([], DT, i)
    assert tracker.tracks == {}
    tracker.update([det([10, 10, 50, 50])], DT, 40)
    track = next(iter(tracker.tracks.values()))
    assert track.box_at(40) is not None and track.box_at(39) is None


def test_looming_needs_a_few_detections():
    tracker = ScaleTracker()
    tracks = tracker.update([det([10, 10, 50, 50])], DT, 0)
    assert looming(next(iter(tracks.values()))) is None


def test_history_keeps_only_detected_boxes():
    # A lost track's predicted box must not stand in for a measurement in a later scale search.
    tracker = ScaleTracker()
    for i in range(5):
        tracker.update([det([100, 100, 160, 150])], DT, i)
    tracker.update([], DT, 5)
    track = tracker.update([det([100, 100, 160, 150])], DT, 6)[0]
    assert track.box_at(4) is not None and track.box_at(6) is not None
    assert track.box_at(5) is None
