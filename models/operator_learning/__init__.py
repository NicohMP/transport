"""Neural operator and surrogate model architectures."""

from models.operator_learning.diffusion import (
    CNN1DDiffusionSurrogate,
    DilatedResidualBlock,
)

__all__ = [
    "CNN1DDiffusionSurrogate",
    "DilatedResidualBlock",
]
