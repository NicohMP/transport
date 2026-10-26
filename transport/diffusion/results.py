from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CriticalityResult:
    """Diffusion criticality solution and convergence diagnostics."""

    k_eff: float
    phi: np.ndarray
    residual: np.ndarray
    normalized_residual: float
    iterations: int
    factorization_time: float
    iteration_time: float
    total_time: float

    @property
    def is_positive(self) -> bool:
        """Check that scalar flux is nonnegative."""
        return bool(np.all(self.phi >= 0.0))

    @property
    def is_finite(self) -> bool:
        """Check that solution values and diagnostics are finite."""
        return (
            bool(np.isfinite(self.k_eff))
            and bool(np.all(np.isfinite(self.phi)))
            and bool(np.all(np.isfinite(self.residual)))
            and bool(np.isfinite(self.normalized_residual))
            and bool(np.isfinite(self.factorization_time))
            and bool(np.isfinite(self.iteration_time))
            and bool(np.isfinite(self.total_time))
        )


@dataclass(frozen=True)
class DiffusionStationaryResult:
    """Stationary diffusion flux, current, and balance residual."""

    phi: np.ndarray
    current: np.ndarray
    balance_residual: np.ndarray

    @property
    def residual_norm_inf(self) -> float:
        """Return the maximum absolute balance residual."""
        return np.linalg.norm(self.balance_residual, ord=np.inf)

    @property
    def is_positive(self) -> bool:
        """Check that scalar flux is nonnegative."""
        return bool(np.all(self.phi >= 0.0))

    @property
    def is_finite(self) -> bool:
        """Check that flux, current, and residual are finite."""
        return (
            bool(np.all(np.isfinite(self.phi)))
            and bool(np.all(np.isfinite(self.current)))
            and bool(np.all(np.isfinite(self.balance_residual)))
        )


@dataclass(frozen=True)
class DiffusionTimeResult:
    """Diffusion flux and current histories."""

    phi: np.ndarray
    current: np.ndarray

    @property
    def is_positive(self) -> bool:
        """Check that scalar flux remains nonnegative."""
        return bool(np.all(self.phi >= 0.0))

    @property
    def is_finite(self) -> bool:
        """Check that flux and current histories are finite."""
        return bool(np.all(np.isfinite(self.phi))) and bool(
            np.all(np.isfinite(self.current))
        )
