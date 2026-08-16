from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransportStationaryResult:
    psi: np.ndarray
    phi: np.ndarray
    current: np.ndarray
    errors: np.ndarray
    iterations: int

    @property
    def is_positive(self) -> bool:
        return bool(np.all(self.psi >= 0.0) and np.all(self.phi >= 0.0))

    @property
    def is_finite(self) -> bool:
        return (
            bool(np.all(np.isfinite(self.psi)))
            and bool(np.all(np.isfinite(self.phi)))
            and bool(np.all(np.isfinite(self.current)))
            and bool(np.all(np.isfinite(self.errors)))
        )
