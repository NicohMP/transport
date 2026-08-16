from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Material1D:
    # Inherent properties
    sigma_s: float
    sigma_c: float
    sigma_f: float
    nu: float
    source: float

    def __post_init__(self) -> None:
        values = (self.sigma_s, self.sigma_c, self.sigma_f, self.nu, self.source)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Material coefficients must be finite")

        if any(value < 0.0 for value in values):
            raise ValueError("Material coefficients must be non-negative")

        if self.sigma_t <= 0.0:
            raise ValueError("sigma_t must be strictly positive")

    # Computed properties
    @property
    def sigma_t(self) -> float:
        return self.sigma_f + self.sigma_c + self.sigma_s

    @property
    def sigma_a(self) -> float:
        return self.sigma_c + self.sigma_f

    @property
    def sigma_p(self) -> float:
        return self.sigma_c + self.nu * self.sigma_f

    @property
    def diffusion_coefficient(self) -> float:
        return 1.0 / (3.0 * self.sigma_t)
