import pytest

from collision_avoidance.fcw.collision import CourseChecker, Level, WarningFsm

DT = 1 / 30


def run_course(checker, boxes, heading_x=256.0, horizon_y=192.0, ttc=2.0, yaw=0.0, name="car"):
    result = None
    for i, box in enumerate(boxes):
        result = checker.update(1, i * DT, box, name, heading_x, horizon_y, ttc, yaw)
    return result


def test_object_straight_ahead_is_on_course():
    assert run_course(CourseChecker(), [(226, 200, 286, 240)] * 10).on_course


def test_parked_car_beside_lane_is_not_on_course():
    # Static object 3 m to the right: r = X / W stays 3 / 1.8 while the box grows.
    boxes = [(256 + 500 * 2.1 / z, 200, 256 + 500 * 3.9 / z, 240) for z in (20 - 0.5 * i for i in range(10))]
    res = run_course(CourseChecker(), boxes)
    assert not res.on_course and res.r == pytest.approx(3 / 1.8, rel=1e-6)


def test_object_cutting_in_is_on_course():
    # Starts 2.5 m right, moves left 2 m/s: at contact (TTC 2 s) it is at -1.5 m... just inside.
    boxes = []
    for i in range(15):
        x, z = 2.5 - 2.0 * i * DT, 20.0
        boxes.append((256 + 500 * (x - 0.9) / z, 200, 256 + 500 * (x + 0.9) / z, 240))
    assert run_course(CourseChecker(), boxes, ttc=1.5).on_course


def test_repeated_timestamps_do_not_break_the_line_fit():
    checker = CourseChecker()
    for _ in range(5):
        res = checker.update(1, 0.0, (226, 200, 286, 240), "car", 256.0, 192.0, 2.0, 0.0)
    assert res.on_course


def test_objects_above_horizon_and_unknown_classes_are_ignored():
    assert not run_course(CourseChecker(), [(226, 100, 286, 150)] * 5).on_course
    assert not run_course(CourseChecker(), [(226, 200, 286, 240)] * 5, name="dog").on_course


def test_turning_widens_the_corridor():
    box = [(256 + 500 * 1.5 / 20, 200, 256 + 500 * 3.3 / 20, 240)]  # X = 2.4 m: r = 1.33, threshold 1.08 (x1.5 = 1.63)
    assert not run_course(CourseChecker(), box * 5).on_course
    assert run_course(CourseChecker(), box * 5, yaw=0.1).on_course


def steps(fsm, inputs):
    return [fsm.step(*args) for args in inputs]


def test_warning_needs_three_frames_and_two_methods():
    fsm = WarningFsm()
    assert steps(fsm, [(True, 2.5, 0.1, 1)] * 5)[-1] == Level.NONE
    assert steps(fsm, [(True, 2.5, 0.1, 2)] * 3) == [Level.NONE, Level.NONE, Level.WARNING]


def test_critical_is_immediate_and_clears_slowly():
    fsm = WarningFsm()
    assert fsm.step(True, 1.2, 0.2, 2) == Level.CRITICAL
    levels = steps(fsm, [(True, 2.5, 0.1, 2)] * 10)
    assert levels[:9] == [Level.CRITICAL] * 9 and levels[9] == Level.WARNING
    levels = steps(fsm, [(True, 3.5, 0.1, 2)] * 10)
    assert levels[-1] == Level.NONE


def test_warning_clears_when_object_leaves_path():
    fsm = WarningFsm()
    steps(fsm, [(True, 2.5, 0.1, 2)] * 3)
    assert steps(fsm, [(False, 2.5, 0.1, 2)] * 10)[-1] == Level.NONE


@pytest.mark.parametrize("lateral_m, expected", [(0.0, True), (0.8, True), (1.0, False), (2.4, False)])
def test_in_path_needs_half_our_width_of_overlap(lateral_m, expected):
    # Car (1.8 m) at 20 m; our car is 1.8 m wide, so half-overlap means |X| <= 0.9 m.
    box = (256 + 500 * (lateral_m - 0.9) / 20, 200, 256 + 500 * (lateral_m + 0.9) / 20, 240)
    assert CourseChecker().update(1, 0.0, box, "car", 256.0, 192.0, 2.0, 0.0).in_path is expected
