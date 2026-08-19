"""Compiled numerical kernels for the one-dimensional SN solver.

This module is deliberately private: its functions accept already validated,
contiguous numerical arrays and form the performance-critical implementation
behind :mod:`transport.sn.stationary`.
"""

import numpy as np
from numba import njit


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def solve_flux_only_numba(
    psi: np.ndarray,
    fixed_source: np.ndarray,
    source_operator: np.ndarray,
    weights: np.ndarray,
    mu_pos: np.ndarray,
    sigma_t: np.ndarray,
    h: float,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    phi_tol: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Run the complete source iteration without crossing into Python."""
    n_directions, n_cells = psi.shape
    n_half = n_directions // 2

    phi = np.empty(n_cells)
    phi_new = np.empty(n_cells)
    source = np.empty(n_cells)
    psi_new = np.empty_like(psi)
    incoming_positive = np.empty(n_half)
    incoming_negative = np.empty(n_half)

    for i in range(n_cells):
        value = 0.0
        for m in range(n_directions):
            value += weights[m] * psi[m, i]
        phi[i] = value

    error = np.inf
    tiny = np.finfo(np.float64).tiny

    for iteration in range(1, max_iterations + 1):
        for i in range(n_cells):
            source[i] = 0.5 * source_operator[i] * phi[i] + fixed_source[i]

        # Negative directions are stored in increasing quadrature-node order,
        # hence reversed positive-node index used for their absolute value.
        for m in range(n_half):
            incoming_negative[m] = right_boundary[m]
        for i in range(n_cells - 1, -1, -1):
            for m in range(n_half):
                mu_abs = mu_pos[n_half - 1 - m]
                value = (mu_abs * incoming_negative[m] + h * source[i]) / (
                    mu_abs + h * sigma_t[i]
                )
                psi_new[m, i] = value
                incoming_negative[m] = value

        for m in range(n_half):
            incoming_positive[m] = left_boundary[m]
        for i in range(n_cells):
            for m in range(n_half):
                mu = mu_pos[m]
                value = (mu * incoming_positive[m] + h * source[i]) / (
                    mu + h * sigma_t[i]
                )
                psi_new[n_half + m, i] = value
                incoming_positive[m] = value

        max_difference = 0.0
        max_phi = 0.0
        for i in range(n_cells):
            value = 0.0
            for m in range(n_directions):
                value += weights[m] * psi_new[m, i]
            phi_new[i] = value

            difference = abs(value - phi[i])
            if difference > max_difference:
                max_difference = difference

            phi_magnitude = abs(phi[i])
            if phi_magnitude > max_phi:
                max_phi = phi_magnitude

        error = max_difference / max(max_phi, tiny)

        psi, psi_new = psi_new, psi
        phi, phi_new = phi_new, phi

        if error < phi_tol:
            return psi, phi, iteration, error

    return psi, phi, max_iterations, error


@njit(cache=True)
def criticality_residual_numba(
    psi: np.ndarray,
    phi: np.ndarray,
    k_eff: float,
    mu: np.ndarray,
    sigma_t: np.ndarray,
    sigma_s: np.ndarray,
    nu_sigma_f: np.ndarray,
    h: float,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
) -> float:
    """Return ||(T-S)psi-F psi/k||_2 normalized as in the report."""
    n_directions, n_cells = psi.shape
    n_half = n_directions // 2
    residual_norm_squared = 0.0
    loss_norm_squared = 0.0
    fission_norm_squared = 0.0

    for m in range(n_half):
        mu_abs = -mu[m]
        incoming = right_boundary[m]
        for i in range(n_cells - 1, -1, -1):
            transport = (mu_abs + h * sigma_t[i]) * psi[m, i] - mu_abs * incoming
            loss = transport - 0.5 * h * sigma_s[i] * phi[i]
            fission = 0.5 * h * nu_sigma_f[i] * phi[i] / k_eff
            residual = loss - fission

            residual_norm_squared += residual * residual
            loss_norm_squared += loss * loss
            fission_norm_squared += fission * fission
            incoming = psi[m, i]

    for m in range(n_half, n_directions):
        mu_value = mu[m]
        incoming = left_boundary[m - n_half]
        for i in range(n_cells):
            transport = (mu_value + h * sigma_t[i]) * psi[m, i] - mu_value * incoming
            loss = transport - 0.5 * h * sigma_s[i] * phi[i]
            fission = 0.5 * h * nu_sigma_f[i] * phi[i] / k_eff
            residual = loss - fission

            residual_norm_squared += residual * residual
            loss_norm_squared += loss * loss
            fission_norm_squared += fission * fission
            incoming = psi[m, i]

    scale = np.sqrt(loss_norm_squared) + np.sqrt(fission_norm_squared)
    return np.sqrt(residual_norm_squared) / max(scale, np.finfo(np.float64).tiny)


@njit(cache=True)
def instat_positive_sweep_numba(
    psi_n: np.ndarray,
    velocity: float,
    dt: float,
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
    alpha = 1 / (velocity * dt)

    for i in range(n_cells):
        for m in range(n_half):
            psi_pos[m, i] = (
                psi_in[m] * mu_pos[m] + h * (source[i] + psi_n[n_half + m, i] * alpha)
            ) / (mu_pos[m] + h * (sigma_t[i] + alpha))
            psi_in[m] = psi_pos[m, i]

    return psi_pos


@njit(cache=True)
def instat_negative_sweep_numba(
    psi_n: np.ndarray,
    velocity: float,
    dt: float,
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
    alpha = 1 / (velocity * dt)

    for i in range(n_cells - 1, -1, -1):
        for m in range(n_half):
            psi_neg[m, i] = (
                psi_in[m] * mu_neg_abs[m] + h * (source[i] + psi_n[m, i] * alpha)
            ) / (mu_neg_abs[m] + h * (sigma_t[i] + alpha))
            psi_in[m] = psi_neg[m, i]

    return psi_neg


@njit(cache=True)
def implicit_euler_transport_step_numba(
    psi_n: np.ndarray,
    fixed_source: np.ndarray,
    source_operator: np.ndarray,
    weights: np.ndarray,
    mu_pos: np.ndarray,
    sigma_t: np.ndarray,
    h: float,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    velocity: float,
    dt: float,
    phi_tol: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Solve one implicit time step while keeping ``psi_n`` unchanged."""
    n_directions, n_cells = psi_n.shape
    n_half = n_directions // 2

    # psi_n is the state at t^n and must remain the fixed temporal source
    # throughout the source iteration. psi is the t^(n+1) iterate.
    psi = psi_n.copy()
    phi = np.empty(n_cells)
    phi_new = np.empty(n_cells)
    source = np.empty(n_cells)
    psi_new = np.empty_like(psi_n)
    incoming_positive = np.empty(n_half)
    incoming_negative = np.empty(n_half)
    alpha = 1.0 / (velocity * dt)

    for i in range(n_cells):
        value = 0.0
        for m in range(n_directions):
            value += weights[m] * psi[m, i]
        phi[i] = value

    error = np.inf
    tiny = np.finfo(np.float64).tiny

    for iteration in range(1, max_iterations + 1):
        for i in range(n_cells):
            source[i] = 0.5 * source_operator[i] * phi[i] + fixed_source[i]

        # Negative directions are stored in increasing quadrature-node order,
        # hence reversed positive-node index used for their absolute value.
        for m in range(n_half):
            incoming_negative[m] = right_boundary[m]
        for i in range(n_cells - 1, -1, -1):
            for m in range(n_half):
                mu_abs = mu_pos[n_half - 1 - m]
                value = (
                    mu_abs * incoming_negative[m]
                    + h * (source[i] + alpha * psi_n[m, i])
                ) / (mu_abs + h * (sigma_t[i] + alpha))
                psi_new[m, i] = value
                incoming_negative[m] = value

        for m in range(n_half):
            incoming_positive[m] = left_boundary[m]
        for i in range(n_cells):
            for m in range(n_half):
                mu = mu_pos[m]
                value = (
                    mu * incoming_positive[m]
                    + h * (source[i] + alpha * psi_n[n_half + m, i])
                ) / (mu + h * (sigma_t[i] + alpha))
                psi_new[n_half + m, i] = value
                incoming_positive[m] = value

        max_difference = 0.0
        max_phi = 0.0
        for i in range(n_cells):
            value = 0.0
            for m in range(n_directions):
                value += weights[m] * psi_new[m, i]
            phi_new[i] = value

            difference = abs(value - phi[i])
            if difference > max_difference:
                max_difference = difference

            phi_magnitude = abs(phi[i])
            if phi_magnitude > max_phi:
                max_phi = phi_magnitude

        error = max_difference / max(max_phi, tiny)

        psi, psi_new = psi_new, psi
        phi, phi_new = phi_new, phi

        if error < phi_tol:
            return psi, phi, iteration, error

    return psi, phi, max_iterations, error
