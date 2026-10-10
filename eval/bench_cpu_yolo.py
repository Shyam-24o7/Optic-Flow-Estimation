"""YOLOv9t on the CPU (no DPU): ms per 512x384 frame, one thread and all threads.

    python eval/bench_cpu_yolo.py --source clip.mp4 --frames 30

PyTorch's CPU kernels (oneDNN) are vectorised, so this is the best case for a
CPU-only detector; a hand-written C convolution would be slower.
"""

import argparse
import time

import cv2
import numpy as np
import torch


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--weights", default="yolov9t.pt")
    args = p.parse_args(argv)
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.source)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (512, 384), interpolation=cv2.INTER_AREA))
    model = YOLO(args.weights)
    for threads in (1, torch.get_num_threads()):
        torch.set_num_threads(threads)
        model.predict(frames[0], imgsz=(384, 512), device="cpu", verbose=False)  # warm-up
        times = []
        for f in frames:
            t0 = time.perf_counter()
            model.predict(f, imgsz=(384, 512), device="cpu", verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"YOLOv9t CPU, {threads} thread(s): {np.mean(times):.1f} ms/frame (median {np.median(times):.1f})")


if __name__ == "__main__":
    main()
