import numpy as np
from numba import njit

from transport.mesh import UniformMesh1D
from transport.quadrature import SNQuadrature
from transport.sn.results import TransportStationaryResult


@njit
def positive_sweep_numba(
    source: np.ndarray,
    mu_pos: np.ndarray,
    sigma_t: np.ndarray,
    h: float,
    left_boundary: np.ndarray,
) -> np.ndarray:
    n_half = mu_pos.size
    n_cells = sigma_t.size
    psi_pos = np.empty((n_half, n_cells))
    psi_in = left_boundary.copy()

    for i in range(n_cells):
        for m in range(n_half):
            psi_pos[m, i] = (psi_in[m] * mu_pos[m] + h * source[i]) / (
                mu_pos[m] + h * sigma_t[i]
            )
            psi_in[m] = psi_pos[m, i]

    return psi_pos


@njit
def negative_sweep_numba(
    source: np.ndarray,
    mu_neg_abs: np.ndarray,
    sigma_t: np.ndarray,
    h: float,
    right_boundary: np.ndarray,
) -> np.ndarray:
    n_half = mu_neg_abs.size
    n_cells = sigma_t.size
    psi_neg = np.empty((n_half, n_cells))
    psi_in = right_boundary.copy()

    for i in range(n_cells - 1, -1, -1):
        for m in range(n_half):
            psi_neg[m, i] = (psi_in[m] * mu_neg_abs[m] + h * source[i]) / (
                mu_neg_abs[m] + h * sigma_t[i]
            )
            psi_in[m] = psi_neg[m, i]

    return psi_neg


class StatTransportProblem:
    def __init__(
        self,
        mesh: UniformMesh1D,
        quadrature: SNQuadrature,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
        phi_tol: float = 1e-10,
        max_iterations: int = 10_000,
    ) -> None:
        self.mesh = mesh
        self.quadrature = quadrature
        self.phi_tol = phi_tol
        self.max_iterations = max_iterations

        self.n_directions = quadrature.order
        self.n_half = self.n_directions // 2
        self.n_cells = mesh.n_cells

        if self.n_directions % 2 != 0:
            raise ValueError("quadrature order must be even")

        # Separate nodes for positive and negative sweep
        self.mu = quadrature.nodes
        self.weights = quadrature.weights
        self.mu_pos = self.mu[self.n_half :]
        self.mu_neg_abs = self.mu_pos[::-1]

        # Precompute mesh properties
        self.h = mesh.h
        self.sigma_t = mesh.Sigma_t
        self.sigma_s = mesh.Sigma_s
        self.source = mesh.source

        # Default boundary condition to vacuum (Dirichlet)
        if left_boundary is None:
            left_boundary = np.zeros(self.n_half)
        if right_boundary is None:
            right_boundary = np.zeros(self.n_half)

        self.left_boundary = np.asarray(left_boundary, dtype=float)
        self.right_boundary = np.asarray(right_boundary, dtype=float)
        self._validate_inputs()

    def scalar_flux(self, psi: np.ndarray) -> np.ndarray:
        return self.weights @ psi

    def current(self, psi: np.ndarray) -> np.ndarray:
        return (self.weights * self.mu) @ psi

    def solve(
        self,
        psi0: np.ndarray | None = None,
        fixed_source: np.ndarray | None = None,
    ) -> TransportStationaryResult:
        # Start from an isotropic angular flux: psi = phi / 2, with phi = 1.
        if psi0 is None:
            psi = 0.5 * np.ones((self.n_directions, self.n_cells))
        else:
            psi = np.array(psi0, dtype=float, copy=True)
            self._validate_psi(psi)

        if fixed_source is None:
            fixed_source = 0.5 * self.source
        else:
            fixed_source = np.asarray(fixed_source, dtype=float)
            self._validate_cell_array(fixed_source, "fixed_source")

        errors: list[float] = []

        for iteration in range(1, self.max_iterations + 1):
            phi = self.scalar_flux(psi)
            source = 0.5 * self.sigma_s * phi + fixed_source

            psi_new = self.sweep(source)

            phi_new = self.scalar_flux(psi_new)

            error = float(
                np.linalg.norm(phi_new - phi, ord=np.inf)
                / max(np.linalg.norm(phi, ord=np.inf), np.finfo(float).tiny)
            )
            errors.append(error)

            psi = psi_new
            if error < self.phi_tol:
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
        self.validate_result(result)
        return result

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

    def _validate_inputs(self) -> None:
        if self.mu.shape != (self.n_directions,):
            raise ValueError("quadrature nodes have invalid shape")
        if self.weights.shape != (self.n_directions,):
            raise ValueError("quadrature weights have invalid shape")
        if not np.all(self.mu[: self.n_half] < 0.0):
            raise ValueError("first half of quadrature nodes must be negative")
        if not np.all(self.mu[self.n_half :] > 0.0):
            raise ValueError("second half of quadrature nodes must be positive")
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

    def validate_result(self, result: TransportStationaryResult) -> None:
        if not result.is_finite:
            raise ValueError("transport result contains NaN or infinity")
        if result.psi.shape != (self.n_directions, self.n_cells):
            raise ValueError("psi has invalid shape")
        if result.phi.shape != (self.n_cells,):
            raise ValueError("phi has invalid shape")
        if result.current.shape != (self.n_cells,):
            raise ValueError("current has invalid shape")
        if result.errors.size != result.iterations:
            raise ValueError("errors size does not match iterations")
        if result.errors[-1] > self.phi_tol:
            raise ValueError(
                f"transport iteration error too large: {result.errors[-1]}"
            )
        if not result.is_positive:
            raise ValueError("transport solution contains negative values")
