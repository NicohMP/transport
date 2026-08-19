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

    def validate(
        self,
        *,
        n_directions: int,
        n_cells: int,
        phi_tol: float,
    ) -> None:
        """Validate solution invariants against the originating problem."""
        if not self.is_finite:
            raise ValueError("transport result contains NaN or infinity")
        if self.psi.shape != (n_directions, n_cells):
            raise ValueError("psi has invalid shape")
        if self.phi.shape != (n_cells,):
            raise ValueError("phi has invalid shape")
        if self.current.shape != (n_cells,):
            raise ValueError("current has invalid shape")
        if self.errors.ndim != 1 or self.errors.size != self.iterations:
            raise ValueError("errors size does not match iterations")
        if self.iterations < 1:
            raise ValueError("iterations must be strictly positive")
        if self.errors[-1] > phi_tol:
            raise ValueError(f"transport iteration error too large: {self.errors[-1]}")
        if not self.is_positive:
            raise ValueError("transport solution contains negative values")


@dataclass(frozen=True)
class EulerTransportResult:
    PSI: np.ndarray
    PHI: np.ndarray
    Current: np.ndarray
    stored_steps: np.ndarray
    errors: np.ndarray
    iterations: np.ndarray

    @property
    def is_positive(self) -> bool:
        return bool(np.all(self.PSI >= 0.0) and np.all(self.PHI >= 0.0))

    @property
    def is_finite(self) -> bool:
        return (
            bool(np.all(np.isfinite(self.PSI)))
            and bool(np.all(np.isfinite(self.PHI)))
            and bool(np.all(np.isfinite(self.Current)))
            and bool(np.all(np.isfinite(self.errors)))
            and bool(np.all(np.isfinite(self.iterations)))
        )

    def validate(
        self,
        *,
        n_steps: int,
        n_directions: int,
        n_cells: int,
        phi_tol: float,
    ) -> None:
        """Validate solution invariants against the originating problem."""
        if not self.is_finite:
            raise ValueError("transport result contains NaN or infinity")
        if self.stored_steps.ndim != 1 or self.stored_steps.size == 0:
            raise ValueError("stored_steps must be a non-empty vector")
        if self.PSI.shape != (self.stored_steps.size, n_directions, n_cells):
            raise ValueError("PSI has invalid shape")
        if self.PHI.shape != (n_steps + 1, n_cells):
            raise ValueError("phi has invalid shape")
        if self.Current.shape != (self.stored_steps.size, n_cells):
            raise ValueError("current has invalid shape")
        if self.errors.shape != (n_steps,):
            raise ValueError("errors has invalid shape")
        if self.iterations.shape != (n_steps,):
            raise ValueError("iterations has invalid shape")
        if self.stored_steps[0] != 0 or self.stored_steps[-1] != n_steps:
            raise ValueError("stored_steps must include initial and final states")
        if np.any(np.diff(self.stored_steps) <= 0):
            raise ValueError("stored_steps must be strictly increasing")
        if np.any(self.stored_steps < 0) or np.any(self.stored_steps > n_steps):
            raise ValueError("stored_steps contains an invalid time index")
        if np.any(self.iterations < 1):
            raise ValueError("iterations must be strictly positive")
        if np.any(self.errors >= phi_tol):
            raise ValueError(
                f"transport iteration error too large: {np.max(self.errors)}"
            )
        if not self.is_positive:
            raise ValueError("transport solution contains negative values")
