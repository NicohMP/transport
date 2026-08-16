from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SNQuadrature:
    nodes: np.ndarray
    weights: np.ndarray

    @property
    def order(self) -> int:
        return len(self.nodes)

    @classmethod
    def gauss_legendre(cls, order: int) -> "SNQuadrature":
        if not isinstance(order, int) or order <= 0 or order % 2 != 0:
            raise ValueError("order must be a positive even integer")

        nodes, weights = np.polynomial.legendre.leggauss(order)
        return cls(nodes=nodes, weights=weights)
