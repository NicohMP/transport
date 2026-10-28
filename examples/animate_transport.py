"""Generate the README animation from the implicit Euler transport solver."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transport import Material1D, Reactor1D, SNQuadrature, UniformMesh1D
from transport.sn import EulerTransportProblem, StatTransportProblem


def simulate():
    """Compute startup and stationary fluxes for the notebook's reactor."""
    reactor = Reactor1D(
        half_width=10.0,
        core_half_width=6.0,
        speed=1.0,
        core=Material1D(0.70, 0.06, 0.02, 2.4, 0.03),
        reflector=Material1D(0.80, 0.04, 0.0, 0.0, 0.0),
    )
    mesh = UniformMesh1D.from_reactor(reactor, n_cells=120)
    quadrature = SNQuadrature.gauss_legendre(8)
    timestep, endtime = 2.0, 180.0
    source = 0.5 * mesh.source

    stationary = StatTransportProblem(mesh, quadrature).solve(
        fixed_source=source, source_operator=mesh.Sigma_p, phi_tol=1e-10
    )
    result = EulerTransportProblem(
        mesh, quadrature, reactor.speed, endtime, timestep
    ).solve(
        psi0=np.zeros((quadrature.order, mesh.n_cells)),
        fixed_source=source,
        source_operator=mesh.Sigma_p,
        phi_tol=1e-10,
    )
    times = np.arange(result.PHI.shape[0]) * timestep
    gap = np.max(np.abs(result.PHI - stationary.phi), axis=1) / stationary.phi.max()

    np.testing.assert_allclose(result.PHI[0], 0.0)
    np.testing.assert_allclose(result.PHI, result.PHI[:, ::-1], atol=1e-10)
    if gap[-1] >= 1e-3:
        raise RuntimeError("The demo has not approached its stationary reference.")
    print(f"Final relative gap to equilibrium: {gap[-1]:.3e}")
    print(f"Maximum source-iteration error: {result.errors.max():.3e}")
    return reactor, mesh, quadrature, stationary, result, times, gap


def render(output: Path) -> None:
    """Save a looping GIF and a stationary PNG preview."""
    reactor, mesh, _, stationary, result, times, _ = simulate()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(9.0, 4.5), dpi=100, facecolor="white")
    ax = fig.add_axes((0.10, 0.16, 0.86, 0.74), facecolor="white")
    L, a = reactor.half_width, reactor.core_half_width
    ax.axvspan(-a, a, color="0.95", zorder=0)
    for interface in (-a, a):
        ax.axvline(interface, color="0.65", lw=0.8, ls=":")
    for xpos, label in (
        (-(L + a) / 2, "reflector"),
        (0, "core"),
        ((L + a) / 2, "reflector"),
    ):
        ax.text(
            xpos, 1.035, label, transform=ax.get_xaxis_transform(),
            ha="center", color="0.4", size=10,
        )

    # One fixed normalization preserves the physical growth of the flux.
    reference = stationary.phi / stationary.phi.max()
    profiles = result.PHI / stationary.phi.max()
    ax.plot(
        mesh.centers, reference, color="0.4", lw=1.3,
        ls=(0, (4, 4)), label="Stationary SN",
    )
    line, = ax.plot(
        mesh.centers, profiles[0], color="#31688e", lw=2, label="Implicit Euler"
    )
    time_text = ax.text(
        0.97, 0.96, "", transform=ax.transAxes, ha="right", va="top", size=11
    )
    ax.set_xlim(-L, L)
    ax.set_ylim(-0.025, 1.12)
    ax.set_xticks((-10, -6, 0, 6, 10))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel(r"$x$ (a.u.)", labelpad=8)
    ax.set_ylabel(r"$\phi(x,t)\,/\,\max_x\phi_{\mathrm{stat}}$", labelpad=10)
    ax.tick_params(colors="0.3")
    ax.grid(axis="y", color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("0.6")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    def update(step):
        """Display a computed state without temporal interpolation."""
        line.set_ydata(profiles[step])
        time_text.set_text(f"$t = {times[step]:.0f}$ a.u.")
        return line, time_text

    fps = 12
    # Uniform simulation steps followed by a one-second hold at the final state.
    frames = [*range(len(times)), *([len(times) - 1] * fps)]
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
    update(len(times) - 1)
    preview = output.with_suffix(".png")
    fig.savefig(preview, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Animation: {output} ({output.stat().st_size / 1024:.0f} KiB)")
    print(f"Static preview: {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "assets" / "transport_euler.gif",
        help="Destination GIF; a PNG preview is saved beside it.",
    )
    render(parser.parse_args().output)
