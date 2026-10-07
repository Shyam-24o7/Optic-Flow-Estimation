import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evttc_sequence_without_annotations_is_skipped(tmp_path):
    evttc_fcw = load_script("evttc_fcw")
    seq = tmp_path / "seq"
    (seq / "annotations").mkdir(parents=True)
    (seq / "gt_ttc.csv").write_text("0 0.0 20.0 -5.0 4.0\n1 0.01 19.95 -5.0 3.99\n")
    writer = cv2.VideoWriter(str(seq / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    result = evttc_fcw.evaluate_sequence(seq, detector=None)
    assert result["rows"] == [] and "annotations" in result["skipped"]
    summary = evttc_fcw.summarise([result])
    assert summary["approach_events"] == 0
