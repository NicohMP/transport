from dataclasses import dataclass
import numpy as np
from transport.reactor import Reactor1D


@dataclass(frozen=True)
class UniformMesh1D:
    """Uniform one-dimensional mesh with material coefficients."""

    n_cells: int
    h: float
    faces: np.ndarray
    centers: np.ndarray
    Sigma_s: np.ndarray
    Sigma_c: np.ndarray
    Sigma_f: np.ndarray
    nu: np.ndarray
    source: np.ndarray

    def __post_init__(self) -> None:
        """Validate mesh arrays and material coefficients."""
        values = (self.Sigma_s, self.Sigma_c, self.Sigma_f, self.nu, self.source)

        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Material coefficients must be finite")

        if any(value.shape != (self.n_cells,) for value in values):
            raise ValueError("source must have size n_cells")

        if any(np.any(value < 0.0) for value in values):
            raise ValueError("Material coefficients must be non-negative")

        if np.any(self.Sigma_t <= 0.0):
            raise ValueError("sigma_t must be strictly positive")

        if self.centers.size != self.n_cells:
            raise ValueError("centers must have size n_cells")

        if self.faces.size != self.n_cells + 1:
            raise ValueError("faces must have size n_cells + 1")

    @classmethod
    def from_reactor(cls, reactor: Reactor1D, n_cells: int):
        """Discretize a reactor with interfaces on mesh faces."""
        # Mesh spatial params
        n_cells = int(n_cells)
        h = 2 * reactor.half_width / n_cells
        # Material mesh arrays
        faces = np.linspace(-reactor.half_width, reactor.half_width, n_cells + 1)
        centers = 0.5 * (faces[:-1] + faces[1:])

        # Check interface is on a face
        distance = reactor.half_width + reactor.core_half_width
        if not np.isclose(distance / h, round(distance / h)):
            raise ValueError("Heart-reflector interface should lie on a mesh face")
        # Masque de coeur pour définir les materiaux
        core_mask = np.abs(centers) < reactor.core_half_width

        sigma_s = np.where(core_mask, reactor.core.sigma_s, reactor.reflector.sigma_s)
        sigma_f = np.where(core_mask, reactor.core.sigma_f, reactor.reflector.sigma_f)
        sigma_c = np.where(core_mask, reactor.core.sigma_c, reactor.reflector.sigma_c)
        nu = np.where(core_mask, reactor.core.nu, reactor.reflector.nu)
        source = np.where(core_mask, reactor.core.source, reactor.reflector.source)

        return cls(
            n_cells=n_cells,
            h=h,
            faces=faces,
            centers=centers,
            Sigma_s=sigma_s,
            Sigma_c=sigma_c,
            Sigma_f=sigma_f,
            nu=nu,
            source=source,
        )

    # Computed properties
    @property
    def Sigma_t(self) -> np.ndarray:
        """Return total cross sections per cell."""
        return self.Sigma_f + self.Sigma_c + self.Sigma_s

    @property
    def Sigma_a(self) -> np.ndarray:
        """Return absorption cross sections per cell."""
        return self.Sigma_c + self.Sigma_f

    @property
    def Sigma_p(self) -> np.ndarray:
        """Return scattering and fission production per cell."""
        return self.Sigma_s + self.nu * self.Sigma_f

    @property
    def diffusion_coefficient(self) -> np.ndarray:
        """Return diffusion coefficients per cell."""
        return 1.0 / (3.0 * self.Sigma_t)
