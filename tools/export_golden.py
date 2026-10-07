"""Export golden vectors from the Python FCW reference for the C++ parity tests (plan 2).

    python tools/export_golden.py --out host/tests/golden            # every module
    python tools/export_golden.py --out host/tests/golden --only horn

Each module writes host/tests/golden/<module>.yml.gz (OpenCV FileStorage), read
by host/tests/golden.hpp. int64 values are written as decimal strings because
FileStorage integers are 32-bit.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


class Writer:
    def __init__(self, out_dir: Path, name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{name}.yml.gz"
        self.fs = cv2.FileStorage(str(self.path), cv2.FILE_STORAGE_WRITE)

    def mat(self, key: str, value) -> None:
        self.fs.write(key, np.ascontiguousarray(value))

    def num(self, key: str, value) -> None:
        self.fs.write(key, float(value))

    def i64(self, key: str, value) -> None:
        self.fs.write(key, str(int(value)))

    def str(self, key: str, value: str) -> None:
        self.fs.write(key, value)

    def close(self) -> None:
        self.fs.release()
        print("wrote", self.path)


def export_harness(out: Path) -> None:
    w = Writer(out, "harness")
    w.mat("m", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]]))
    w.num("x", 3.25)
    w.i64("big", 6_000_000_000)
    w.close()


def _dets(boxes, names=None):
    from collision_avoidance.tracking import Detection
    return [Detection(np.asarray(b, float), (names or ["car"] * len(boxes))[i], 0.9) for i, b in enumerate(boxes)]


def _track_state(w, key, tracks):
    ids = sorted(tracks)
    w.mat(f"{key}_ids", np.array(ids, np.float64).reshape(-1, 1) if ids else np.zeros((0, 1)))
    for tid in ids:
        t = tracks[tid]
        w.mat(f"{key}_t{tid}_x", t.x.reshape(-1, 1))
        w.mat(f"{key}_t{tid}_P", t.P)
        w.num(f"{key}_t{tid}_lost", t.lost)
        w.num(f"{key}_t{tid}_age", t.age)
        hist = np.array([[i, *b] for i, b in t.history], np.float64) if t.history else np.zeros((0, 5))
        w.mat(f"{key}_t{tid}_hist", hist)


def export_tracker(out: Path) -> None:
    """Three detection sequences replayed frame by frame; full tracker state after every frame."""
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.looming import looming
    from collision_avoidance.fcw.tracker import ScaleTracker

    dt = 1 / 30
    scene = synth.render_scene(synth.SceneConfig(n_frames=40, z0_m=40.0))
    cases = {
        "approach": [_dets([b]) for b in scene.boxes],
        "two": [_dets([[10 + 2 * i, 10, 50 + 2 * i, 40], [300, 200 - i, 400, 260 - i]], ["car", "truck"]) for i in range(10)],
        "lost": [_dets([[100, 100, 160, 150]])] * 5 + [[]] + [_dets([[100, 100, 160, 150]])],
    }
    w = Writer(out, "tracker")
    w.num("dt", dt)
    for name, frames in cases.items():
        tracker = ScaleTracker()
        w.num(f"{name}_frames", len(frames))
        for i, dets in enumerate(frames):
            w.mat(f"{name}_f{i}_dets", np.array([d.bbox for d in dets], np.float64).reshape(-1, 4))
            w.str(f"{name}_f{i}_cls", ",".join(d.class_name for d in dets))
            tracks = tracker.update(dets, dt, i)
            _track_state(w, f"{name}_f{i}", tracks)
            for tid, t in tracks.items():
                m = looming(t)
                if m is not None:
                    w.num(f"{name}_f{i}_t{tid}_loom_eta", m.eta)
                    w.num(f"{name}_f{i}_t{tid}_loom_var", m.var)
    w.close()


EXPORTERS = {"harness": export_harness, "tracker": export_tracker}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("host/tests/golden"))
    p.add_argument("--only", nargs="*", choices=sorted(EXPORTERS))
    args = p.parse_args(argv)
    for name in args.only or EXPORTERS:
        EXPORTERS[name](args.out)


if __name__ == "__main__":
    main()
