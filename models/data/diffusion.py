from dataclasses import dataclass
import numpy as np
from transport.mesh import UniformMesh1D


@dataclass(frozen=True)
class MaterialSample:
    """Sampled material fields, cell width, and optional velocity."""

    h: float
    sigma_s: np.ndarray
    sigma_c: np.ndarray
    sigma_f: np.ndarray
    nu: np.ndarray
    source: np.ndarray
    velocity: float | None = None

    def __post_init__(self) -> None:
        """Validate sampled fields, cell width, and velocity."""
        n_cells = self.sigma_s.size
        fields = (self.sigma_s, self.sigma_c, self.sigma_f, self.nu, self.source)

        if self.h <= 0.0:
            raise ValueError("h must be positive")

        if self.velocity is not None and self.velocity <= 0.0:
            raise ValueError("velocity must be positive")

        for field in fields:
            if field.shape != (n_cells,):
                raise ValueError("All fields must have shape (n_cells,)")

            if not np.all(np.isfinite(field)):
                raise ValueError("All fields must be finite")

            if np.any(field < 0.0):
                raise ValueError("Material/source fields must be non-negative")

        if np.any(self.sigma_s + self.sigma_c + self.sigma_f <= 0.0):
            raise ValueError("sigma_t must be strictly positive")

    @property
    def n_cells(self) -> int:
        """Return the number of sampled cells."""
        return self.sigma_s.size

    def to_uniform_mesh(self) -> UniformMesh1D:
        """Build a uniform mesh from the sampled fields."""
        faces = np.linspace(
            -0.5 * self.n_cells * self.h,
            0.5 * self.n_cells * self.h,
            self.n_cells + 1,
        )
        centers = 0.5 * (faces[:-1] + faces[1:])

        return UniformMesh1D(
            n_cells=self.n_cells,
            h=self.h,
            faces=faces,
            centers=centers,
            Sigma_s=self.sigma_s,
            Sigma_c=self.sigma_c,
            Sigma_f=self.sigma_f,
            nu=self.nu,
            source=self.source,
        )


@dataclass(frozen=True)
class GaussianParameter:
    """Gaussian means, deviations, and clipping bounds."""

    mean: np.ndarray
    std: np.ndarray
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        """Validate Gaussian arrays and clipping bounds."""
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape")

        if not np.all(np.isfinite(self.mean)):
            raise ValueError("mean must be finite")

        if not np.all(np.isfinite(self.std)):
            raise ValueError("std must be finite")

        if np.any(self.std < 0.0):
            raise ValueError("std must be non-negative")

        if self.min_value > self.max_value:
            raise ValueError("min_value must be <= max_value")

    @classmethod
    def from_scalar(
        cls,
        size: int,
        mean: float,
        min_value: float,
        max_value: float,
        std: float = 0,
    ) -> "GaussianParameter":
        """Broadcast scalar Gaussian settings to an array."""
        return cls(
            mean=mean * np.ones(size),
            std=std * np.ones(size),
            min_value=min_value,
            max_value=max_value,
        )


@dataclass(frozen=True)
class GaussianConfig:
    """Gaussian sampling parameters for a material sample."""

    h: GaussianParameter
    sigma_s: GaussianParameter
    sigma_c: GaussianParameter
    sigma_f: GaussianParameter
    nu: GaussianParameter
    source: GaussianParameter
    velocity: GaussianParameter | None = None

    def __post_init__(self) -> None:
        """Check scalar and material parameter shapes."""
        if self.h.mean.shape != (1,):
            raise ValueError("h must have shape (1,)")

        if self.velocity is not None and self.velocity.mean.shape != (1,):
            raise ValueError("velocity must have shape (1,)")

        n_cells = self.sigma_s.mean.size
        fields = (self.sigma_s, self.sigma_c, self.sigma_f, self.nu, self.source)

        for field in fields:
            if field.mean.shape != (n_cells,):
                raise ValueError(
                    "All material/source parameters must have shape (n_cells,)"
                )

    @property
    def n_cells(self) -> int:
        """Return the configured number of cells."""
        return self.sigma_s.mean.size


class GaussianSampler:
    """Generate material samples from clipped Gaussians."""

    def __init__(
        self,
        params: GaussianConfig,
        seed: int | None = None,
    ) -> None:
        """Set sampling parameters and initialize the random generator."""
        self.params = params
        self.rng = np.random.default_rng(seed)

    def _sample_parameter(self, param: GaussianParameter) -> np.ndarray:
        """Draw and clip a Gaussian parameter array."""
        values = self.rng.normal(loc=param.mean, scale=param.std)
        return np.clip(values, param.min_value, param.max_value)

    def _sample_scalar(self, param: GaussianParameter) -> float:
        """Draw a scalar from a one-element Gaussian parameter."""
        values = self._sample_parameter(param)

        if values.shape != (1,):
            raise ValueError("Expected scalar parameter with shape (1,)")

        return float(values[0])

    def sample(self) -> MaterialSample:
        """Draw one material sample."""
        velocity = None
        if self.params.velocity is not None:
            velocity = self._sample_scalar(self.params.velocity)

        return MaterialSample(
            h=self._sample_scalar(self.params.h),
            sigma_s=self._sample_parameter(self.params.sigma_s),
            sigma_c=self._sample_parameter(self.params.sigma_c),
            sigma_f=self._sample_parameter(self.params.sigma_f),
            nu=self._sample_parameter(self.params.nu),
            source=self._sample_parameter(self.params.source),
            velocity=velocity,
        )
