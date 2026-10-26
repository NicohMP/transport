import numpy as np

from transport.mesh import UniformMesh1D
from transport.quadrature import SNQuadrature
from transport.sn._kernels import implicit_euler_transport_step_numba
from transport.sn.results import EulerTransportResult


class EulerTransportProblem:
    """Transient SN transport problem using implicit Euler."""

    def __init__(
        self,
        mesh: UniformMesh1D,
        quadrature: SNQuadrature,
        velocity: float,
        endtime: float,
        timestep: float,
        left_boundary: np.ndarray | None = None,
        right_boundary: np.ndarray | None = None,
        max_iterations: int = 10_000,
    ) -> None:
        """Initialize transport data and time stepping."""
        self.mesh = mesh
        self.quadrature = quadrature
        self.max_iterations = max_iterations
        self.velocity = float(velocity)

        self.n_directions = quadrature.order
        self.n_half = self.n_directions // 2
        self.n_cells = mesh.n_cells

        # Time steps
        self.T = float(endtime)
        self.dt = float(timestep)
        self._validate_time_parameters()
        self.Nt = int(round(self.T / self.dt))
        self.psi_storage_stride = 10

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
        """Integrate angular flux over directions."""
        return self.weights @ psi

    def current(self, psi: np.ndarray) -> np.ndarray:
        """Compute the first angular flux moment."""
        return (self.weights * self.mu) @ psi

    def solve(
        self,
        psi0: np.ndarray | None = None,
        fixed_source: np.ndarray | None = None,
        source_operator: np.ndarray | None = None,
        phi_tol: float = 1e-10,
    ) -> EulerTransportResult:
        """Advance transport and record solution histories."""
        if not np.isfinite(phi_tol) or phi_tol <= 0.0:
            raise ValueError("phi_tol must be finite and strictly positive")

        psi = self._initial_or_checked_psi(psi0)
        fixed_source = self._checked_fixed_source(fixed_source)
        source_operator = self._checked_source_operator(source_operator)

        errors = np.empty(self.Nt)
        iterations = np.empty(self.Nt, dtype=np.int64)

        stored_steps = np.arange(
            0,
            self.Nt + 1,
            self.psi_storage_stride,
            dtype=np.int64,
        )
        if stored_steps[-1] != self.Nt:
            stored_steps = np.append(stored_steps, self.Nt)

        PSI = np.empty((stored_steps.size, self.n_directions, self.n_cells))
        PHI = np.empty((self.Nt + 1, self.n_cells))
        Current = np.empty((stored_steps.size, self.n_cells))

        PSI[0] = psi
        PHI[0] = self.scalar_flux(psi)
        Current[0] = self.current(psi)
        next_stored_index = 1

        for n in range(self.Nt):
            psi_next, phi_next, it, err = self._step(
                psi_n=psi,
                fixed_source=fixed_source,
                source_operator=source_operator,
                phi_tol=phi_tol,
            )
            if err >= phi_tol:
                raise RuntimeError(
                    "SN source iteration did not converge at time step "
                    f"{n + 1}/{self.Nt}; iterations={it}, last error={err}"
                )

            psi = psi_next
            PHI[n + 1] = phi_next
            errors[n] = err
            iterations[n] = it

            step = n + 1
            if step == stored_steps[next_stored_index]:
                PSI[next_stored_index] = psi_next
                Current[next_stored_index] = self.current(psi_next)
                next_stored_index += 1

        result = EulerTransportResult(
            PSI=PSI,
            PHI=PHI,
            Current=Current,
            stored_steps=stored_steps,
            errors=errors,
            iterations=iterations,
        )
        result.validate(
            n_steps=self.Nt,
            n_directions=self.n_directions,
            n_cells=self.n_cells,
            phi_tol=phi_tol,
        )
        return result

    def _step(
        self,
        psi_n: np.ndarray,
        fixed_source: np.ndarray,
        source_operator: np.ndarray,
        phi_tol: float,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Solve one implicit Euler transport step."""
        return implicit_euler_transport_step_numba(
            psi_n=psi_n,
            fixed_source=fixed_source,
            source_operator=source_operator,
            weights=self.weights,
            mu_pos=self.mu_pos,
            sigma_t=self.sigma_t,
            h=self.h,
            left_boundary=self.left_boundary,
            right_boundary=self.right_boundary,
            velocity=self.velocity,
            dt=self.dt,
            phi_tol=phi_tol,
            max_iterations=self.max_iterations,
        )

    def _validate_time_parameters(self) -> None:
        """Check velocity, time grid, and iteration limit."""
        values = (self.velocity, self.T, self.dt)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("velocity, endtime and timestep must be finite")
        if self.velocity <= 0.0:
            raise ValueError("velocity must be strictly positive")
        if self.T <= 0.0:
            raise ValueError("endtime must be strictly positive")
        if self.dt <= 0.0:
            raise ValueError("timestep must be strictly positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")

        n_steps = self.T / self.dt
        if not np.isclose(n_steps, round(n_steps), rtol=1e-12, atol=1e-12):
            raise ValueError("endtime must be an integer multiple of timestep")

    def _validate_boundaries(self) -> None:
        """Check boundary array shapes and finiteness."""
        if self.left_boundary.shape != (self.n_half,):
            raise ValueError("left_boundary must have size quadrature.order // 2")
        if self.right_boundary.shape != (self.n_half,):
            raise ValueError("right_boundary must have size quadrature.order // 2")
        if not np.all(np.isfinite(self.left_boundary)):
            raise ValueError("left_boundary contains NaN or infinity")
        if not np.all(np.isfinite(self.right_boundary)):
            raise ValueError("right_boundary contains NaN or infinity")

    def _validate_psi(self, psi: np.ndarray) -> None:
        """Check angular flux shape and finiteness."""
        if psi.shape != (self.n_directions, self.n_cells):
            raise ValueError("psi must have shape (quadrature.order, mesh.n_cells)")
        if not np.all(np.isfinite(psi)):
            raise ValueError("psi contains NaN or infinity")

    def _validate_cell_array(self, values: np.ndarray, name: str) -> None:
        """Check a cell array's shape and finiteness."""
        if values.shape != (self.n_cells,):
            raise ValueError(f"{name} must have shape (mesh.n_cells,)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")

    def _initial_or_checked_psi(self, psi0: np.ndarray | None) -> np.ndarray:
        """Initialize angular flux or validate and copy psi0."""
        if psi0 is None:
            return 0.5 * np.ones((self.n_directions, self.n_cells))

        psi = np.array(psi0, dtype=float, copy=True)
        self._validate_psi(psi)
        return psi

    def _checked_fixed_source(self, fixed_source: np.ndarray | None) -> np.ndarray:
        """Return the default or validated fixed source."""
        if fixed_source is None:
            return self.source

        values = np.asarray(fixed_source, dtype=float)
        self._validate_cell_array(values, "fixed_source")
        return values

    def _checked_source_operator(
        self, source_operator: np.ndarray | None
    ) -> np.ndarray:
        """Return the default or validated source operator."""
        if source_operator is None:
            return self.sigma_p

        values = np.asarray(source_operator, dtype=float)
        self._validate_cell_array(values, "source_operator")
        return values
