import numpy as np
import scipy.sparse as sp

from transport.diffusion.types import DiffusionConditions
from transport.mesh import UniformMesh1D


def harmonic_face_diffusion(mesh: UniformMesh1D) -> np.ndarray:
    """Compute harmonic diffusion coefficients at mesh faces."""
    harmonic_diffusion = np.zeros_like(mesh.faces)
    harmonic_diffusion[-1] = mesh.diffusion_coefficient[-1]
    harmonic_diffusion[0] = mesh.diffusion_coefficient[0]
    harmonic_diffusion[1:-1] = (
        2
        * mesh.diffusion_coefficient[1:]
        * mesh.diffusion_coefficient[:-1]
        / (mesh.diffusion_coefficient[1:] + mesh.diffusion_coefficient[:-1])
    )
    return harmonic_diffusion


def assemble_diffusion_loss_matrix(
    mesh: UniformMesh1D, boundary: DiffusionConditions = "extrapolated"
) -> sp.csc_matrix:
    """Assemble the sparse diffusion loss matrix."""
    # Compute harmonic diffusion coefficient at each face
    Dh = harmonic_face_diffusion(mesh)

    # Determine extremities flux from boundary conditions
    if boundary == "extrapolated":
        delta_l = 2.1312 * mesh.diffusion_coefficient[0]
        delta_r = 2.1312 * mesh.diffusion_coefficient[-1]
        left_leak = Dh[0] / (mesh.h / 2 + delta_l)
        right_leak = Dh[-1] / (mesh.h / 2 + delta_r)
    elif boundary == "dirichlet":
        left_leak = Dh[0] / (mesh.h)
        right_leak = Dh[-1] / (mesh.h / 2)
    else:
        raise ValueError("unknown boundary")

    # Assemble diffusion diagonal
    diag = np.empty(mesh.centers.size)
    diag[0] = left_leak + Dh[1] / mesh.h + mesh.h * mesh.Sigma_a[0]
    diag[-1] = Dh[-2] / mesh.h + right_leak + mesh.h * mesh.Sigma_a[-1]
    diag[1:-1] = (Dh[1:-2] + Dh[2:-1]) / mesh.h + mesh.h * mesh.Sigma_a[1:-1]
    # Assemble sub/super diagonals
    lower = -Dh[1:-1] / mesh.h
    upper = -Dh[1:-1] / mesh.h
    # Assemble diffusion matrix
    A = (
        sp.diags(lower, offsets=-1, format="csc")
        + sp.diags(diag, offsets=0, format="csc")
        + sp.diags(upper, offsets=1, format="csc")
    )
    return A


def diffusion_current(
    mesh: UniformMesh1D, phi: np.ndarray, boundary: DiffusionConditions = "extrapolated"
) -> np.ndarray:
    """Compute diffusion currents at mesh faces."""
    Dh = harmonic_face_diffusion(mesh)
    currents = np.zeros_like(Dh)
    if boundary == "extrapolated":
        delta_l = 2.1312 * mesh.diffusion_coefficient[0]
        delta_r = 2.1312 * mesh.diffusion_coefficient[-1]
        left_leak = Dh[0] / (mesh.h / 2 + delta_l)
        right_leak = Dh[-1] / (mesh.h / 2 + delta_r)
    elif boundary == "dirichlet":
        left_leak = Dh[0] / (mesh.h)
        right_leak = Dh[-1] / (mesh.h / 2)
    currents[1:-1] = -Dh[1:-1] * (phi[1:] - phi[:-1]) / mesh.h
    currents[0] = -left_leak * phi[0]
    currents[-1] = right_leak * phi[-1]
    return currents


def assemble_fission_matrix(mesh: UniformMesh1D) -> sp.dia_matrix:
    """Assemble the diagonal fission production matrix."""
    return sp.diags(mesh.h * mesh.nu * mesh.Sigma_f, format="csc")
