import time

import numpy as np
import scipy.sparse.linalg as spla

from transport.diffusion.operators import (
    assemble_diffusion_loss_matrix,
    assemble_fission_matrix,
    diffusion_current,
)
from transport.diffusion.results import CriticalityResult, DiffusionStationaryResult
from transport.diffusion.types import DiffusionConditions
from transport.mesh import UniformMesh1D


def criticity_factor(
    mesh: UniformMesh1D,
    ktol: float = 1e-12,
    residual_tol: float = 1e-10,
    max_iterations: int = 10_000,
    boundary: DiffusionConditions = "extrapolated",
) -> CriticalityResult:
    """Compute diffusion criticality by power iteration."""
    fission_mass = mesh.h * mesh.nu * mesh.Sigma_f
    if not np.any(fission_mass > 0.0):
        raise ValueError("Cannot compute criticality without fission production")

    total_start = time.perf_counter()
    A = assemble_diffusion_loss_matrix(mesh, boundary)

    factorization_start = time.perf_counter()
    LU = spla.splu(A)
    factorization_time = time.perf_counter() - factorization_start

    phi = np.ones(mesh.centers.size)
    phi /= np.sum(fission_mass * phi)

    k_eff = 1.0
    normalized_residual = np.inf
    iteration_start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        phi_unorm = LU.solve(fission_mass * phi)
        k_new = float(np.sum(fission_mass * phi_unorm))
        if k_new <= 0.0 or not np.isfinite(k_new):
            raise ValueError(f"Invalid criticality factor: {k_new}")

        phi_new = phi_unorm / k_new

        if abs(k_new - k_eff) / abs(k_new) < ktol:
            lhs = A @ phi_new
            rhs = (fission_mass * phi_new) / k_new
            residual = lhs - rhs
            residual_scale = max(
                np.linalg.norm(lhs, ord=np.inf),
                np.linalg.norm(rhs, ord=np.inf),
                np.finfo(float).tiny,
            )
            normalized_residual = float(
                np.linalg.norm(residual, ord=np.inf) / residual_scale
            )

            if normalized_residual < residual_tol:
                iteration_time = time.perf_counter() - iteration_start
                total_time = time.perf_counter() - total_start
                return CriticalityResult(
                    k_eff=k_new,
                    phi=phi_new,
                    residual=residual,
                    normalized_residual=normalized_residual,
                    iterations=iteration,
                    factorization_time=factorization_time,
                    iteration_time=iteration_time,
                    total_time=total_time,
                )

        phi = phi_new
        k_eff = k_new

    raise RuntimeError(
        "Criticality power iteration did not converge "
        f"after {max_iterations} iterations; "
        f"last k={k_eff}, normalized_residual={normalized_residual}"
    )


def solve_stationnary_source(mesh: UniformMesh1D, source: np.ndarray) -> np.ndarray:
    """Solve diffusion for a supplied source vector."""
    A = assemble_diffusion_loss_matrix(mesh)

    LU = spla.splu(A)
    phi = LU.solve(source)
    return phi


class StatDiffusionSolver:
    """Stationary diffusion solver with fission and a fixed source."""

    def __init__(
        self, boundary: DiffusionConditions = "extrapolated"
    ) -> None:
        """Set diffusion boundary conditions."""
        self.boundary: DiffusionConditions = boundary

    def solve(self, mesh: UniformMesh1D) -> DiffusionStationaryResult:
        """Compute stationary flux, current, and balance residual."""
        A = assemble_diffusion_loss_matrix(mesh, self.boundary)
        F = assemble_fission_matrix(mesh)
        b = mesh.h * mesh.source
        LU = spla.splu(A - F)
        phi = LU.solve(b)

        current = diffusion_current(
            mesh,
            phi,
            boundary=self.boundary,
        )

        residual = (
            (current[1:] - current[:-1]) / mesh.h
            + mesh.Sigma_a * phi
            - mesh.nu * mesh.Sigma_f * phi
            - mesh.source
        )

        result = DiffusionStationaryResult(
            phi=phi,
            current=current,
            balance_residual=residual,
        )

        self.validate_result(mesh, result)

        return result

    def validate_result(
        self,
        mesh: UniformMesh1D,
        result: DiffusionStationaryResult,
        residual_tol: float = 1e-10,
    ) -> None:
        """Validate solution arrays, positivity, and balance."""
        if not result.is_finite:
            raise ValueError("Diffusion result contains NaN or infinity")

        if result.phi.shape != (mesh.n_cells,):
            raise ValueError("phi has invalid shape")

        if result.current.shape != (mesh.n_cells + 1,):
            raise ValueError("current has invalid shape")

        if result.balance_residual.shape != (mesh.n_cells,):
            raise ValueError("balance_residual has invalid shape")

        if result.residual_norm_inf > residual_tol:
            raise ValueError(f"Balance residual too large: {result.residual_norm_inf}")

        if np.any(result.phi < -residual_tol):
            raise ValueError("Diffusion solution contains negative values")
