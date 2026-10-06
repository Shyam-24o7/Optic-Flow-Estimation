"""The common output of every TTC method: an inverse TTC with its variance."""

from dataclasses import dataclass

METHODS = ("looming", "scale", "horn", "divergence")


@dataclass(frozen=True)
class Measurement:
    method: str   # one of METHODS
    eta: float    # inverse TTC in 1/s; > 0 approaching, <= 0 receding or static
    var: float    # variance of eta in 1/s^2

    @property
    def ttc_s(self) -> float | None:
        return 1.0 / self.eta if self.eta > 0 else None
