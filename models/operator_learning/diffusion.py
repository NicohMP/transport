import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    """Apply two dilated convolutions with a residual connection."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel: int,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            padding=dilation,
            dilation=dilation,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class CNN1DDiffusionSurrogate(nn.Module):
    """Predict log flux and criticality from material fields and cell width.

    Input and target normalization is handled by the caller. The criticality
    branch uses only material and geometry features; the flux branch also
    receives the external source.
    """

    def __init__(
        self,
        n_cells: int = 100,
        width: int = 64,
        kernel: int = 3,
        k_hidden: int = 64,
        dilations: list[int] = [1, 2, 4, 8, 16],
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Material channels: sigma_s, sigma_c, sigma_f, nu, h, x_coordinate.
        self.material_stem = nn.Conv1d(
            in_channels=6,
            out_channels=width,
            kernel_size=5,
            padding=2,
        )
        self.material_blocks = nn.Sequential(
            *(DilatedResidualBlock(width, dilation, kernel) for dilation in dilations)
        )

        # The flux depends on the fixed source; k_eff does not.
        self.flux_stem = nn.Conv1d(
            in_channels=width + 1,
            out_channels=width,
            kernel_size=3,
            padding=1,
        )
        self.flux_blocks = nn.Sequential(
            *(DilatedResidualBlock(width, dilation, kernel) for dilation in dilations)
        )
        self.phi_head = nn.Conv1d(width, 1, kernel_size=1)

        # Concatenate global average and maximum material features.
        self.k_head = nn.Sequential(
            nn.Linear(2 * width, k_hidden),
            nn.SiLU(),
            nn.Linear(k_hidden, 1),
        )

        coordinate = torch.linspace(-1.0, 1.0, n_cells).view(1, 1, n_cells)
        self.register_buffer("coordinate", coordinate)

    def forward(
        self,
        cell: torch.Tensor,
        h: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map cell features (batch, 5, n_cells) and h to both predictions.

        Cell channels are sigma_s, sigma_c, sigma_f, nu, and source. Cell
        widths have shape (batch,) or (batch, 1). Outputs are log_phi with
        shape (batch, n_cells) and k_eff with shape (batch,).
        """
        if cell.ndim != 3 or cell.shape[1] != 5:
            raise ValueError("cell must have shape (batch, 5, n_cells)")

        if h.ndim == 1:
            h = h.unsqueeze(1)
        if h.ndim != 2 or h.shape[1] != 1:
            raise ValueError("h must have shape (batch, 1)")

        batch_size, _, n_cells = cell.shape
        if n_cells != self.coordinate.shape[-1]:  # type: ignore
            raise ValueError(f"Expected {self.coordinate.shape[-1]} cells, got {n_cells}")  # type: ignore

        material = cell[:, :4]  # sigma_s, sigma_c, sigma_f, nu
        source = cell[:, 4:5]

        h_channel = h.unsqueeze(-1).expand(batch_size, 1, n_cells)
        coordinate = self.coordinate.expand(batch_size, -1, -1)  # type: ignore

        material_input = torch.cat(
            (material, h_channel, coordinate),
            dim=1,
        )
        material_features = self.material_blocks(self.material_stem(material_input))

        flux_features = torch.cat((material_features, source), dim=1)
        flux_features = self.flux_blocks(self.flux_stem(flux_features))

        log_phi = self.phi_head(flux_features).squeeze(1)

        pooled_mean = material_features.mean(dim=-1)
        pooled_max = material_features.amax(dim=-1)
        k_eff = self.k_head(torch.cat((pooled_mean, pooled_max), dim=1)).squeeze(1)

        return {
            "log_phi": log_phi,
            "k_eff": k_eff,
        }
