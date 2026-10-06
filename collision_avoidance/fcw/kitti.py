"""KITTI raw helpers: timestamps, rectified camera matrix and OXTS yaw rate.

Layout (from the KITTI raw download):
    <date>/calib_cam_to_cam.txt
    <date>/<date>_drive_NNNN_sync/image_02/data/*.png, image_02/timestamps.txt
    <date>/<date>_drive_NNNN_sync/oxts/data/*.txt (30 values per line)
OXTS field 22 is wu, the angular rate about the vehicle's upward axis (rad/s,
positive turning left). With this package's convention (R maps camera
coordinates prev -> curr, camera y pointing down) a left turn gives a
positive rotation about camera y, so wu compares directly with rvec[1] / dt.
"""

from pathlib import Path

import numpy as np

OXTS_WU = 22


def parse_timestamps(path: Path) -> np.ndarray:
    """Seconds since the first line, from lines like '2011-09-26 13:02:25.964389698'."""
    seconds = []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            h, m, s = line.split()[1].split(":")
            seconds.append(int(h) * 3600 + int(m) * 60 + float(s))
    t = np.array(seconds)
    return t - t[0]


def camera_matrix(calib_path: Path, camera: str = "02") -> np.ndarray:
    for line in Path(calib_path).read_text().split("\n"):
        if line.startswith(f"P_rect_{camera}:"):
            return np.array(line.split(":")[1].split(), float).reshape(3, 4)[:, :3]
    raise ValueError(f"P_rect_{camera} not found in {calib_path}")


def yaw_rates(oxts_dir: Path) -> np.ndarray:
    files = sorted(Path(oxts_dir, "data").glob("*.txt"))
    return np.array([float(f.read_text().split()[OXTS_WU]) for f in files])


def load_drive(drive_dir: Path):
    """(image paths, image times [s], K, OXTS yaw rate per frame [rad/s])."""
    drive_dir = Path(drive_dir)
    images = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    times = parse_timestamps(drive_dir / "image_02" / "timestamps.txt")
    K = camera_matrix(drive_dir.parent / "calib_cam_to_cam.txt")
    return images, times, K, yaw_rates(drive_dir / "oxts")
