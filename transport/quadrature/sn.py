from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SNQuadrature:
    nodes: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        weights = np.asarray(self.weights, dtype=float)

        if nodes.ndim != 1 or weights.ndim != 1:
            raise ValueError("nodes and weights must be one-dimensional arrays")
        if nodes.shape != weights.shape:
            raise ValueError("nodes and weights must have the same shape")
        if nodes.size == 0 or nodes.size % 2 != 0:
            raise ValueError("quadrature order must be a positive even integer")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
            raise ValueError("nodes and weights must be finite")

        n_half = nodes.size // 2
        if not np.all(nodes[:n_half] < 0.0) or not np.all(nodes[n_half:] > 0.0):
            raise ValueError("quadrature nodes must be negative then positive")
        if not np.allclose(nodes[:n_half], -nodes[:n_half - 1 : -1]):
            raise ValueError("quadrature nodes must be symmetric about zero")
        if not np.allclose(weights[:n_half], weights[:n_half - 1 : -1]):
            raise ValueError("quadrature weights must be symmetric about zero")

        # Direct construction accepts array-like inputs and normalizes them
        # before they are consumed by the solvers.
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def order(self) -> int:
        return len(self.nodes)

    @classmethod
    def gauss_legendre(cls, order: int) -> "SNQuadrature":
        if not isinstance(order, int) or order <= 0 or order % 2 != 0:
            raise ValueError("order must be a positive even integer")

        nodes, weights = np.polynomial.legendre.leggauss(order)
        return cls(nodes=nodes, weights=weights)
