import unittest

import numpy as np

from transport import Material1D, Reactor1D, SNQuadrature, UniformMesh1D
from transport.sn.transient import EulerTransportProblem


def make_problem(n_steps: int = 12) -> EulerTransportProblem:
    core = Material1D(0.70, 0.06, 0.04, 2.4, 0.03)
    reflector = Material1D(0.80, 0.10, 0.0, 0.0, 0.0)
    reactor = Reactor1D(10.0, 6.0, 1.0, core, reflector)
    mesh = UniformMesh1D.from_reactor(reactor, n_cells=40)
    quadrature = SNQuadrature.gauss_legendre(order=4)
    timestep = 0.01
    return EulerTransportProblem(
        mesh=mesh,
        quadrature=quadrature,
        velocity=2.0,
        endtime=n_steps * timestep,
        timestep=timestep,
    )


class EulerTransportProblemTests(unittest.TestCase):
    def test_solve_stores_all_phi_and_every_tenth_psi(self) -> None:
        problem = make_problem(n_steps=12)
        result = problem.solve(phi_tol=1e-8)

        np.testing.assert_array_equal(result.stored_steps, np.array([0, 10, 12]))
        self.assertEqual(result.PSI.shape, (3, problem.n_directions, problem.n_cells))
        self.assertEqual(result.PHI.shape, (13, problem.n_cells))
        self.assertEqual(result.Current.shape, (3, problem.n_cells))
        self.assertEqual(result.errors.shape, (12,))
        self.assertEqual(result.iterations.shape, (12,))
        self.assertTrue(np.issubdtype(result.iterations.dtype, np.integer))

        for stored_index, step in enumerate(result.stored_steps):
            np.testing.assert_allclose(
                problem.scalar_flux(result.PSI[stored_index]),
                result.PHI[step],
                rtol=0.0,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                problem.current(result.PSI[stored_index]),
                result.Current[stored_index],
                rtol=0.0,
                atol=1e-13,
            )

    def test_time_interval_must_match_fixed_timestep(self) -> None:
        core = Material1D(0.70, 0.06, 0.04, 2.4, 0.03)
        reflector = Material1D(0.80, 0.10, 0.0, 0.0, 0.0)
        reactor = Reactor1D(10.0, 6.0, 1.0, core, reflector)
        mesh = UniformMesh1D.from_reactor(reactor, n_cells=40)
        quadrature = SNQuadrature.gauss_legendre(order=4)

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            EulerTransportProblem(
                mesh=mesh,
                quadrature=quadrature,
                velocity=2.0,
                endtime=0.105,
                timestep=0.01,
            )

    def test_solve_rejects_invalid_tolerance(self) -> None:
        problem = make_problem(n_steps=2)
        with self.assertRaisesRegex(ValueError, "phi_tol"):
            problem.solve(phi_tol=0.0)


if __name__ == "__main__":
    unittest.main()
