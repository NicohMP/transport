import unittest

import numpy as np
import scipy.sparse.linalg as spla

from transport.diffusion import (
    StatDiffusionSolver,
    assemble_diffusion_loss_matrix,
    assemble_fission_matrix,
)
from transport.mesh import UniformMesh1D


def make_mesh(n_cells: int = 20) -> UniformMesh1D:
    h = 0.5
    faces = np.linspace(-0.5 * n_cells * h, 0.5 * n_cells * h, n_cells + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    return UniformMesh1D(
        n_cells=n_cells,
        h=h,
        faces=faces,
        centers=centers,
        Sigma_s=np.full(n_cells, 0.70),
        Sigma_c=np.full(n_cells, 0.06),
        Sigma_f=np.full(n_cells, 0.02),
        nu=np.full(n_cells, 2.4),
        source=np.full(n_cells, 0.03),
    )


class StatDiffusionSolverTests(unittest.TestCase):
    def test_stationary_solution_includes_fission_production(self) -> None:
        mesh = make_mesh()
        loss = assemble_diffusion_loss_matrix(mesh)
        fission = assemble_fission_matrix(mesh)
        source = mesh.h * mesh.source

        expected = spla.spsolve(loss - fission, source)
        solution_without_fission = spla.spsolve(loss, source)
        result = StatDiffusionSolver().solve(mesh)

        np.testing.assert_allclose(result.phi, expected, rtol=1e-13, atol=1e-13)
        self.assertGreater(
            np.linalg.norm(result.phi - solution_without_fission, ord=np.inf),
            1e-3,
        )
        self.assertLess(result.residual_norm_inf, 1e-12)
        self.assertLess(
            np.linalg.norm((loss - fission) @ result.phi - source, ord=np.inf),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
