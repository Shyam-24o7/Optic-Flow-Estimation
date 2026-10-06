"""Overlay for the FCW output: boxes by warning level, TTC labels, heading and corridor."""

import cv2
import numpy as np

from .collision import Level
from .pipeline import FcwFrame

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {Level.NONE: (0, 200, 0), Level.WARNING: (0, 165, 255), Level.CRITICAL: (0, 0, 255)}  # BGR
OFF_COURSE = (160, 160, 160)
CAMERA_HEIGHT_M = 1.3   # drawing only: where the corridor meets the image bottom
HALF_CORRIDOR_M = 1.05  # (ego width + margin) / 2


def draw(result: FcwFrame, fx: float) -> np.ndarray:
    image = result.frame.copy()
    h, w = image.shape[:2]
    hx, hy = map(float, result.heading)
    if h - 1 > hy:
        z_bottom = fx * CAMERA_HEIGHT_M / (h - 1 - hy)
        dx = fx * HALF_CORRIDOR_M / z_bottom
        for side in (-1, 1):
            cv2.line(image, (int(hx), int(hy)), (int(hx + side * dx), h - 1), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(image, (int(hx), int(hy)), (255, 255, 0), cv2.MARKER_CROSS, 12, 2)

    for obj in result.objects:
        x1, y1, x2, y2 = map(int, obj.bbox)
        on_course = obj.course is not None and obj.course.on_course
        color = COLORS[obj.level] if on_course or obj.level > Level.NONE else OFF_COURSE
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if obj.level else 1)
        ttc = obj.estimate.ttc_s if obj.estimate else None
        label = f"{obj.class_name} {ttc:.1f}s" if ttc is not None else obj.class_name
        cv2.putText(image, label, (x1, max(12, y1 - 4)), FONT, 0.45, color, 1, cv2.LINE_AA)

    level = result.level
    if level > Level.NONE:
        text = "BRAKE" if level == Level.CRITICAL else "COLLISION WARNING"
        cv2.putText(image, text, (10, 30), FONT, 0.8, COLORS[level], 2, cv2.LINE_AA)
    total = result.timings_ms.get("total", 0.0)
    cv2.putText(image, f"{total:.0f} ms", (w - 70, 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image
