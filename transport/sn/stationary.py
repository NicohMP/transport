import numpy as np

from transport.mesh import UniformMesh1D
from transport.quadrature import SNQuadrature
from transport.sn._kernels import (
    criticality_residual_numba,
    negative_sweep_numba,
    positive_sweep_numba,
    solve_flux_only_numba,
)
from transport.sn.results import TransportStationaryResult


class StatTransportProblem:
    def __init__(
        self,
        mesh: UniformMesh1D,
        quadrature: SNQuadrature,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
        max_iterations: int = 10_000,
    ) -> None:
        self.mesh = mesh
        self.quadrature = quadrature
        self.max_iterations = max_iterations

        self.n_directions = quadrature.order
        self.n_half = self.n_directions // 2
        self.n_cells = mesh.n_cells

        # Separate nodes for positive and negative sweep
        self.mu = quadrature.nodes
        self.weights = quadrature.weights
        self.mu_pos = self.mu[self.n_half :]
        self.mu_neg_abs = self.mu_pos[::-1]

        # Precompute mesh properties
        self.h = mesh.h
        self.sigma_t = mesh.Sigma_t
        self.sigma_s = mesh.Sigma_s
        self.sigma_p = mesh.Sigma_p
        self.nu_sigma_f = mesh.nu * mesh.Sigma_f
        self.source = mesh.source

        # Default boundary condition to vacuum (Dirichlet)
        if left_boundary is None:
            left_boundary = np.zeros(self.n_half)
        if right_boundary is None:
            right_boundary = np.zeros(self.n_half)

        self.left_boundary = np.asarray(left_boundary, dtype=float)
        self.right_boundary = np.asarray(right_boundary, dtype=float)
        self._validate_boundaries()

    def scalar_flux(self, psi: np.ndarray) -> np.ndarray:
        return self.weights @ psi

    def current(self, psi: np.ndarray) -> np.ndarray:
        return (self.weights * self.mu) @ psi

    def criticality_residual(
        self, psi: np.ndarray, phi: np.ndarray, k_eff: float
    ) -> float:
        """Compute the normalized residual of the transport eigenproblem."""
        psi_values = np.asarray(psi, dtype=float)
        phi_values = np.asarray(phi, dtype=float)
        self._validate_psi(psi_values)
        self._validate_cell_array(phi_values, "phi")
        if not np.isfinite(k_eff) or k_eff <= 0.0:
            raise ValueError("k_eff must be finite and strictly positive")

        return float(
            criticality_residual_numba(
                psi_values,
                phi_values,
                k_eff,
                self.mu,
                self.sigma_t,
                self.sigma_s,
                self.nu_sigma_f,
                self.h,
                self.left_boundary,
                self.right_boundary,
            )
        )

    def solve(
        self,
        psi0: np.ndarray | None = None,
        fixed_source: np.ndarray | None = None,
        source_operator: np.ndarray | None = None,
        phi_tol: float = 1e-10,
    ) -> TransportStationaryResult:
        psi = self._initial_or_checked_psi(psi0)
        fixed_source = self._checked_fixed_source(fixed_source)
        source_operator = self._checked_source_operator(source_operator)

        errors: list[float] = []

        for iteration in range(1, self.max_iterations + 1):
            phi = self.scalar_flux(psi)
            source = 0.5 * source_operator * phi + fixed_source

            psi_new = self.sweep(source)

            phi_new = self.scalar_flux(psi_new)

            error = float(
                np.linalg.norm(phi_new - phi, ord=np.inf)
                / max(np.linalg.norm(phi, ord=np.inf), np.finfo(float).tiny)
            )
            errors.append(error)

            psi = psi_new
            if error < phi_tol:
                break
        else:
            raise RuntimeError(
                "SN source iteration did not converge "
                f"after {self.max_iterations} iterations; last error={errors[-1]}"
            )

        phi = self.scalar_flux(psi)
        current = self.current(psi)
        result = TransportStationaryResult(
            psi=psi,
            phi=phi,
            current=current,
            errors=np.asarray(errors),
            iterations=iteration,
        )
        result.validate(
            n_directions=self.n_directions,
            n_cells=self.n_cells,
            phi_tol=phi_tol,
        )
        return result

    def solve_flux_only(
        self,
        psi0: np.ndarray | None = None,
        fixed_source: np.ndarray | None = None,
        source_operator: np.ndarray | None = None,
        phi_tol: float = 1e-10,
        assume_valid: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Solve the fixed-source problem and return only psi, phi and error.

        Lightweight path for nested algorithms (transport criticality). Same source iteration and Numba sweeps as
        solve(), but skips current computation, result construction, full error history, and final result validation. ``assume_valid=True`` also skips input copies and validation and may reuse ``psi0`` as workspace.
        Intended for trusted nested loops that replace ``psi0`` with the returned array immediately.
        """
        if assume_valid:
            psi = (
                0.5 * np.ones((self.n_directions, self.n_cells))
                if psi0 is None
                else psi0
            )
            fixed_source = 0.5 * self.source if fixed_source is None else fixed_source
            source_operator = (
                self.sigma_p if source_operator is None else source_operator
            )
        else:
            psi = self._initial_or_checked_psi(psi0)
            fixed_source = self._checked_fixed_source(fixed_source)
            source_operator = self._checked_source_operator(source_operator)

        psi, phi, iteration, error = solve_flux_only_numba(
            psi,
            fixed_source,
            source_operator,
            self.weights,
            self.mu_pos,
            self.sigma_t,
            self.h,
            self.left_boundary,
            self.right_boundary,
            phi_tol,
            self.max_iterations,
        )

        if error < phi_tol:
            return psi, phi, iteration, error

        raise RuntimeError(
            "SN source iteration did not converge "
            f"after {self.max_iterations} iterations; last error={error}"
        )

    def sweep(self, source: np.ndarray) -> np.ndarray:
        psi_neg = negative_sweep_numba(
            source,
            self.mu_neg_abs,
            self.sigma_t,
            self.h,
            self.right_boundary,
        )
        psi_pos = positive_sweep_numba(
            source,
            self.mu_pos,
            self.sigma_t,
            self.h,
            self.left_boundary,
        )
        return np.concatenate((psi_neg, psi_pos), axis=0)

    def _validate_boundaries(self) -> None:
        if self.left_boundary.shape != (self.n_half,):
            raise ValueError("left_boundary must have size quadrature.order // 2")
        if self.right_boundary.shape != (self.n_half,):
            raise ValueError("right_boundary must have size quadrature.order // 2")
        if not np.all(np.isfinite(self.left_boundary)):
            raise ValueError("left_boundary contains NaN or infinity")
        if not np.all(np.isfinite(self.right_boundary)):
            raise ValueError("right_boundary contains NaN or infinity")

    def _validate_psi(self, psi: np.ndarray) -> None:
        if psi.shape != (self.n_directions, self.n_cells):
            raise ValueError("psi must have shape (quadrature.order, mesh.n_cells)")
        if not np.all(np.isfinite(psi)):
            raise ValueError("psi contains NaN or infinity")

    def _validate_cell_array(self, values: np.ndarray, name: str) -> None:
        if values.shape != (self.n_cells,):
            raise ValueError(f"{name} must have shape (mesh.n_cells,)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")

    def _initial_or_checked_psi(self, psi0: np.ndarray | None) -> np.ndarray:
        if psi0 is None:
            return 0.5 * np.ones((self.n_directions, self.n_cells))

        psi = np.array(psi0, dtype=float, copy=True)
        self._validate_psi(psi)
        return psi

    def _checked_fixed_source(self, fixed_source: np.ndarray | None) -> np.ndarray:
        if fixed_source is None:
            return 0.5 * self.source

        values = np.asarray(fixed_source, dtype=float)
        self._validate_cell_array(values, "fixed_source")
        return values

    def _checked_source_operator(
        self, source_operator: np.ndarray | None
    ) -> np.ndarray:
        if source_operator is None:
            return self.sigma_p

        values = np.asarray(source_operator, dtype=float)
        self._validate_cell_array(values, "source_operator")
        return values
