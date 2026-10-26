import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from transport.diffusion.operators import (
    assemble_diffusion_loss_matrix,
    assemble_fission_matrix,
    harmonic_face_diffusion,
)
from transport.diffusion.results import DiffusionTimeResult
from transport.diffusion.types import DiffusionConditions
from transport.mesh import UniformMesh1D


class EulerDiffusionFissionSolver:
    """Implicit Euler solver for diffusion with fission."""

    def __init__(
        self,
        mesh: UniformMesh1D,
        velocity: float,
        endtime: float,
        timestep: float,
        phi0: np.ndarray,
        boundary: DiffusionConditions = "extrapolated",
    ) -> None:
        """Initialize time stepping and factorize the diffusion system."""
        self.boundary: DiffusionConditions = boundary
        self.mesh = mesh
        self.dt = timestep
        self.Nt = int(endtime / timestep)
        self.phi0 = phi0
        self.Dh = harmonic_face_diffusion(self.mesh)
        self.A = assemble_diffusion_loss_matrix(mesh, boundary)
        self.F = assemble_fission_matrix(mesh=mesh)
        self.mass_diag = np.full(mesh.n_cells, mesh.h / (velocity * self.dt))
        self.LU = spla.splu(self.A + sp.diags(self.mass_diag, format="csc") - self.F)

    def solve(self):
        """Compute flux and current histories with implicit Euler."""
        b = self.mesh.h * self.mesh.source
        phi_t = self.phi0
        PHI = np.zeros([self.Nt + 1, self.mesh.n_cells])
        PHI[0, :] = phi_t
        for n in range(self.Nt):
            rhs = self.mass_diag * phi_t + b
            phi_t = self.LU.solve(rhs)
            PHI[n + 1, :] = phi_t

        current = np.zeros((self.Nt + 1, self.mesh.n_cells + 1))
        current[:, 1:-1] = -self.Dh[1:-1] * (PHI[:, 1:] - PHI[:, :-1]) / self.mesh.h

        result = DiffusionTimeResult(
            phi=PHI,
            current=current,
        )

        self.validate_result(result)

        return result

    def validate_result(self, result: DiffusionTimeResult, residual_tol=1e-10):
        """Validate solution shapes, finiteness, and positivity."""
        if not result.is_finite:
            raise ValueError("Diffusion result contains NaN or infinity")

        if result.phi.shape != (self.Nt + 1, self.mesh.n_cells):
            print(result.phi.shape)
            raise ValueError("phi has invalid shape")

        if result.current.shape != (self.Nt + 1, self.mesh.n_cells + 1):
            print(result.current.shape)
            raise ValueError("current has invalid shape")

        if np.any(result.phi < -residual_tol):
            raise ValueError("Diffusion solution contains negative values")
