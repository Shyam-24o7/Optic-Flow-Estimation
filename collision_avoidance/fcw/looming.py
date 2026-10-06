"""Looming: the tracker's log-scale rate is the inverse TTC."""

from .measurement import Measurement
from .tracker import ScaleTrack

MIN_AGE = 3  # the scale rate is meaningless until a few detections have been fused


def looming(track: ScaleTrack) -> Measurement | None:
    if track.age < MIN_AGE or track.lost:
        return None
    return Measurement("looming", track.scale_rate, track.scale_rate_var)
