import unittest

import numpy as np

from transport import Material1D, Reactor1D, SNQuadrature, UniformMesh1D
from transport.sn._kernels import (
    implicit_euler_transport_step_numba,
    instat_negative_sweep_numba,
    instat_positive_sweep_numba,
)
from transport.sn import StatTransportProblem, TransportStationaryResult


def make_problem(n_cells: int = 40, order: int = 8) -> StatTransportProblem:
    core = Material1D(0.70, 0.06, 0.04, 2.4, 0.03)
    reflector = Material1D(0.80, 0.10, 0.0, 0.0, 0.0)
    reactor = Reactor1D(10.0, 6.0, 1.0, core, reflector)
    mesh = UniformMesh1D.from_reactor(reactor, n_cells)
    quadrature = SNQuadrature.gauss_legendre(order)
    return StatTransportProblem(mesh, quadrature)


class StatTransportProblemTests(unittest.TestCase):
    def test_quadrature_validates_sweep_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            SNQuadrature(nodes=np.array([-0.5, 0.5]), weights=np.ones(3))

        with self.assertRaisesRegex(ValueError, "positive even"):
            SNQuadrature(nodes=np.array([0.0]), weights=np.ones(1))

        with self.assertRaisesRegex(ValueError, "symmetric about zero"):
            SNQuadrature(
                nodes=np.array([-0.7, -0.2, 0.1, 0.8]),
                weights=np.ones(4),
            )

    def test_result_validates_its_own_data(self) -> None:
        result = TransportStationaryResult(
            psi=np.ones((4, 3)),
            phi=np.ones(3),
            current=np.zeros(3),
            errors=np.array([1e-12]),
            iterations=1,
        )

        result.validate(n_directions=4, n_cells=3, phi_tol=1e-10)

        invalid_result = TransportStationaryResult(
            psi=result.psi,
            phi=result.phi,
            current=result.current,
            errors=np.array([1e-6]),
            iterations=1,
        )
        with self.assertRaisesRegex(ValueError, "error too large"):
            invalid_result.validate(n_directions=4, n_cells=3, phi_tol=1e-10)

    def test_flux_only_matches_full_solver(self) -> None:
        problem = make_problem()
        profile = np.linspace(0.1, 1.0, problem.n_cells)
        psi0 = np.broadcast_to(
            0.5 * profile, (problem.n_directions, problem.n_cells)
        )
        fixed_source = 0.5 * problem.source

        result = problem.solve(
            psi0=psi0,
            fixed_source=fixed_source,
            source_operator=problem.sigma_s,
            phi_tol=1e-10,
        )
        psi, phi, iterations, error = problem.solve_flux_only(
            psi0=psi0,
            fixed_source=fixed_source,
            source_operator=problem.sigma_s,
            phi_tol=1e-10,
        )
        psi_fast, phi_fast, fast_iterations, fast_error = problem.solve_flux_only(
            psi0=psi0.copy(),
            fixed_source=fixed_source,
            source_operator=problem.sigma_s,
            phi_tol=1e-10,
            assume_valid=True,
        )

        self.assertEqual(iterations, result.iterations)
        self.assertEqual(fast_iterations, result.iterations)
        self.assertAlmostEqual(error, result.errors[-1])
        self.assertAlmostEqual(fast_error, result.errors[-1])
        np.testing.assert_allclose(phi, result.phi, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(psi, result.psi, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(phi_fast, result.phi, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(psi_fast, result.psi, rtol=0.0, atol=1e-14)

    def test_criticality_residual_matches_explicit_operator(self) -> None:
        problem = make_problem(n_cells=20, order=4)
        rng = np.random.default_rng(42)
        psi = rng.random((problem.n_directions, problem.n_cells))
        phi = problem.scalar_flux(psi)
        k_eff = 0.97

        transport = np.empty_like(psi)
        for m in range(problem.n_half):
            mu_abs = -problem.mu[m]
            incoming = problem.right_boundary[m]
            for i in range(problem.n_cells - 1, -1, -1):
                transport[m, i] = (
                    (mu_abs + problem.h * problem.sigma_t[i]) * psi[m, i]
                    - mu_abs * incoming
                )
                incoming = psi[m, i]

        for m in range(problem.n_half, problem.n_directions):
            mu = problem.mu[m]
            incoming = problem.left_boundary[m - problem.n_half]
            for i in range(problem.n_cells):
                transport[m, i] = (
                    (mu + problem.h * problem.sigma_t[i]) * psi[m, i]
                    - mu * incoming
                )
                incoming = psi[m, i]

        loss = transport - 0.5 * problem.h * problem.sigma_s * phi
        fission = 0.5 * problem.h * problem.nu_sigma_f * phi / k_eff
        expected = np.linalg.norm(loss - fission) / (
            np.linalg.norm(loss)
            + np.sqrt(problem.n_directions) * np.linalg.norm(fission)
        )

        actual = problem.criticality_residual(psi, phi, k_eff)
        self.assertAlmostEqual(actual, expected, places=14)

    def test_time_kernel_keeps_previous_time_flux_constant(self) -> None:
        problem = make_problem(n_cells=20, order=4)
        rng = np.random.default_rng(7)
        psi_n = rng.random((problem.n_directions, problem.n_cells))
        psi_n_before = psi_n.copy()
        fixed_source = np.zeros(problem.n_cells)
        source_operator = np.zeros(problem.n_cells)
        velocity = 2.0
        dt = 0.1

        expected_neg = instat_negative_sweep_numba(
            psi_n,
            velocity,
            dt,
            fixed_source,
            problem.mu_neg_abs,
            problem.sigma_t,
            problem.h,
            problem.right_boundary,
        )
        expected_pos = instat_positive_sweep_numba(
            psi_n,
            velocity,
            dt,
            fixed_source,
            problem.mu_pos,
            problem.sigma_t,
            problem.h,
            problem.left_boundary,
        )

        psi, _, iterations, _ = implicit_euler_transport_step_numba(
            psi_n,
            fixed_source,
            source_operator,
            problem.weights,
            problem.mu_pos,
            problem.sigma_t,
            problem.h,
            problem.left_boundary,
            problem.right_boundary,
            velocity,
            dt,
            np.inf,
            1,
        )

        self.assertEqual(iterations, 1)
        np.testing.assert_allclose(psi_n, psi_n_before, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            psi,
            np.concatenate((expected_neg, expected_pos)),
            rtol=0.0,
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
