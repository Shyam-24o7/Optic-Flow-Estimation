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

    def png(self, key: str, image) -> None:
        """Lossless image next to the YAML; the key stores the file name."""
        name = f"{self.path.name.split('.')[0]}_{key}.png"
        cv2.imwrite(str(self.path.parent / name), image)
        self.fs.write(key, name)

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


def _ego(w, key, ego):
    w.mat(f"{key}_R", ego.R)
    w.num(f"{key}_valid", ego.valid)
    w.num(f"{key}_stationary", ego.stationary)
    w.num(f"{key}_has_foe", ego.foe is not None)
    if ego.foe is not None:
        w.mat(f"{key}_foe", np.asarray(ego.foe, float).reshape(1, 2))


def export_ego(out: Path) -> None:
    """Grid samples per case (all the estimator sees), the resulting EgoMotion, and a small sampling case."""
    import cv2 as _cv2
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.ego_rotation import EgoRotationConfig, EgoRotationEstimator, grid_samples, rotation_divergence

    W, H = 512, 384
    K = synth.intrinsics(500, W, H)
    cfg = EgoRotationConfig()

    def dense(R, t, depth):
        _, f = synth.flow_from_motion(K, R, t, depth, W, H, step=1)
        return f.reshape(H, W, 2).astype(np.float32)

    def road(xs, ys):
        return 8.0 + 60.0 * np.random.default_rng(0).random(xs.shape)

    fwd = np.array([0.0, 0.0, -0.5])
    truck = dense(synth.rotation_y(np.radians(0.3)), fwd, road)
    truck[100:300, 150:350] += np.array([6.0, -2.0], np.float32)
    good = dense(synth.rotation_y(np.radians(0.3)), fwd, road)
    noise = np.random.default_rng(1).normal(0, 5, (H, W, 2)).astype(np.float32)
    # Each case is a sequence of (flow, excluded boxes) fed to one estimator.
    cases = {
        "yaw0": [(dense(synth.rotation_y(0.0), fwd, road), [])],
        "yaw04": [(dense(synth.rotation_y(np.radians(0.4)), fwd, road), [])],
        "yawm08": [(dense(synth.rotation_y(np.radians(-0.8)), fwd, road), [])],
        "pure": [(dense(synth.rotation_y(np.radians(0.5)), np.zeros(3), 30.0), [])],
        "truck": [(truck, [(150, 100, 350, 300)])],
        "hold": [(good, []), (noise, [])],
    }
    w = Writer(out, "ego")
    w.mat("K", K)
    for name, seq in cases.items():
        est = EgoRotationEstimator(K)
        w.num(f"{name}_steps", len(seq))
        for i, (flow, boxes) in enumerate(seq):
            prev, curr = grid_samples(flow, boxes, cfg)
            w.mat(f"{name}_s{i}_prev", prev)
            w.mat(f"{name}_s{i}_curr", curr)
            _ego(w, f"{name}_s{i}", est.update(flow, boxes))
    small = np.random.default_rng(2).normal(0, 1, (48, 64, 2)).astype(np.float32)
    w.mat("small_flow", small)
    prev, curr = grid_samples(small, [(10, 5, 30, 20)], EgoRotationConfig(grid_step_px=8, hood_rows=4))
    w.mat("small_prev", prev)
    w.mat("small_curr", curr)
    for j, c in enumerate([(256, 192), (400, 100), (60, 330)]):
        R = _cv2.Rodrigues(np.array([0.004, -0.006, 0.002]))[0]
        w.mat(f"div{j}_R", R)
        w.mat(f"div{j}_c", np.array([c], float))
        w.num(f"div{j}_value", rotation_divergence(R, K, c))
    w.close()


def export_scale(out: Path) -> None:
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.scale_search import choose_gap, scale_ttc, search_scale

    dt = 1 / 30
    approach = synth.render_scene(synth.SceneConfig(n_frames=25))
    static = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0))
    sideways = synth.render_scene(synth.SceneConfig(n_frames=9, closing_speed_mps=0.0, z0_m=15.0, lateral_speed_mps=4.0))
    tiny = synth.render_scene(synth.SceneConfig(n_frames=5, z0_m=200.0))
    cases = [("k2", approach, 24, 2), ("k4", approach, 24, 4), ("k8", approach, 24, 8),
             ("static", static, 8, 8), ("sideways", sideways, 8, 8), ("tiny", tiny, 4, 4)]
    w = Writer(out, "scale")
    w.num("dt", dt)
    w.num("cases", len(cases))
    for i, (name, scene, t, k) in enumerate(cases):
        key = f"c{i}"
        w.str(f"{key}_name", name)
        w.png(f"{key}_gray_t", scene.frames[t])
        w.png(f"{key}_gray_tk", scene.frames[t - k])
        w.mat(f"{key}_box_t", np.asarray(scene.boxes[t], float).reshape(1, 4))
        w.mat(f"{key}_box_tk", np.asarray(scene.boxes[t - k], float).reshape(1, 4))
        w.num(f"{key}_k", k)
        found = search_scale(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k])
        w.num(f"{key}_found", found is not None)
        if found is not None:
            w.num(f"{key}_s", found[0])
            w.num(f"{key}_peak", found[1])
        m = scale_ttc(scene.frames[t], scene.frames[t - k], scene.boxes[t], scene.boxes[t - k], k, dt)
        w.num(f"{key}_has_m", m is not None)
        if m is not None:
            w.num(f"{key}_eta", m.eta)
            w.num(f"{key}_var", m.var)
    gaps = [(None, 8), (1.0, 8), (0.25, 8), (0.25, 3), (0.25, 0), (0.6, 8), (-0.2, 8)]
    w.mat("gaps_in", np.array([[np.nan if e is None else e, a] for e, a in gaps]))
    w.mat("gaps_out", np.array([[choose_gap(e, dt, available=a)] for e, a in gaps], float))
    w.close()


def export_horn(out: Path) -> None:
    from collision_avoidance.fcw import synth
    from collision_avoidance.fcw.horn import HornConfig, choose_level, horn_at_level, horn_solve, horn_sums

    dt = 1 / 30
    rng = np.random.default_rng(3)
    rand_prev = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    rand_curr = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    approach = synth.render_scene(synth.SceneConfig(n_frames=21))
    side = synth.render_scene(synth.SceneConfig(n_frames=21, z0_m=15.0, closing_speed_mps=6.0, lateral_m=-1.0, lateral_speed_mps=1.5))
    flat = np.full((384, 512), 128, np.uint8)
    # 2-px checkerboard in both frames: maximal |Ex| and |Ey| everywhere, so the largest sums.
    checker = np.kron((np.indices((192, 256)).sum(axis=0) % 2) * 255, np.ones((2, 2))).astype(np.uint8)
    cases = [
        ("pixel_loop", rand_prev, rand_curr, (10, 8, 30, 28), (25, 20), HornConfig(shrink=0.0, grad_threshold_l1=0), 0),
        ("approach", approach.frames[19], approach.frames[20], tuple(approach.boxes[20]), (256, 192), HornConfig(), 0),
        ("flat", flat, flat, (100, 100, 200, 200), (256, 192), HornConfig(), 0),
        ("saturated", checker, checker, (1, 1, 511, 383), (256, 192), HornConfig(shrink=0.0, grad_threshold_l1=0), 0),
        ("side_level2", side.frames[19], side.frames[20], tuple(side.boxes[20]), (256, 192), HornConfig(), 2),
    ]
    w = Writer(out, "horn")
    w.num("dt", dt)
    w.num("cases", len(cases))
    for i, (name, prev, curr, box, pp, cfg, level) in enumerate(cases):
        k = f"c{i}"
        w.str(f"{k}_name", name)
        w.png(f"{k}_prev", prev)
        w.png(f"{k}_curr", curr)
        w.mat(f"{k}_box", np.array([box], float))
        w.mat(f"{k}_pp", np.array([pp], float))
        w.num(f"{k}_shrink", cfg.shrink)
        w.num(f"{k}_thr", cfg.grad_threshold_l1)
        w.num(f"{k}_level", level)
        if level == 0:
            found = horn_sums(prev, curr, box, pp, cfg)
            w.num(f"{k}_has_sums", found is not None)
            if found is not None:
                sums, n = found
                for j, v in enumerate(sums):
                    w.i64(f"{k}_sum{j}", v)
                w.i64(f"{k}_n", n)
                m = horn_solve(sums, n, dt, cfg)
            else:
                m = None
        else:
            m = horn_at_level(prev, curr, box, pp, level, dt, cfg)
        w.num(f"{k}_has_m", m is not None)
        if m is not None:
            w.num(f"{k}_eta", m.eta)
            w.num(f"{k}_var", m.var)
    levels = [(None, (0, 0, 120, 90), 0.0), (1.0, (0, 0, 120, 90), 0.0), (0.2, (0, 0, 120, 90), 30.0), (3.0, (0, 0, 400, 300), 200.0)]
    w.mat("levels_in", np.array([[np.nan if e is None else e, *b, v] for e, b, v in levels]))
    w.mat("levels_out", np.array([[choose_level(e, b, v, dt)] for e, b, v in levels], float))
    w.close()


EXPORTERS = {"harness": export_harness, "tracker": export_tracker, "ego": export_ego, "scale": export_scale, "horn": export_horn}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("host/tests/golden"))
    p.add_argument("--only", nargs="*", choices=sorted(EXPORTERS))
    args = p.parse_args(argv)
    for name in args.only or EXPORTERS:
        EXPORTERS[name](args.out)


if __name__ == "__main__":
    main()
