import numpy as np
import pytest

from collision_avoidance.fcw import kitti


def test_timestamps_are_relative_seconds(tmp_path):
    f = tmp_path / "timestamps.txt"
    f.write_text("2011-09-26 13:02:25.964389698\n2011-09-26 13:02:26.068550587\n2011-09-26 13:03:00.000000000\n")
    np.testing.assert_allclose(kitti.parse_timestamps(f), [0.0, 0.104160889, 34.035610302], atol=1e-6)


def test_camera_matrix_reads_p_rect_02(tmp_path):
    f = tmp_path / "calib_cam_to_cam.txt"
    f.write_text("calib_time: x\nP_rect_00: 1 0 0 0 0 1 0 0 0 0 1 0\n"
                 "P_rect_02: 7.215377e+02 0 6.095593e+02 4.485728e+01 0 7.215377e+02 1.728540e+02 2.163791e-01 0 0 1 2.745884e-03\n")
    K = kitti.camera_matrix(f)
    assert K[0, 0] == pytest.approx(721.5377) and K[0, 2] == pytest.approx(609.5593) and K[1, 2] == pytest.approx(172.854)


def test_yaw_rate_is_field_22(tmp_path):
    (tmp_path / "data").mkdir()
    for i, wu in enumerate((0.01, -0.02)):
        values = np.zeros(30)
        values[22] = wu
        (tmp_path / "data" / f"{i:010d}.txt").write_text(" ".join(map(str, values)))
    np.testing.assert_allclose(kitti.yaw_rates(tmp_path), [0.01, -0.02])
