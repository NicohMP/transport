from dataclasses import dataclass

from transport.materials import Material1D


@dataclass(frozen=True)
class Reactor1D:
    half_width: float
    core_half_width: float
    speed: float
    core: Material1D
    reflector: Material1D

    def __post_init__(self) -> None:
        assert self.half_width >= self.core_half_width
        assert self.speed > 0
