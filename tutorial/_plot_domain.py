import matplotlib.pyplot as plt
import numpy as np

from parallel_heat.domain import DenseDomain


def plot_dense_domain(  # noqa: PLR0913
    domain: DenseDomain,
    include_ghost_cells: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    x_offset: float = 0,
    y_offset: float = 0,
):
    dx = domain.parameters.dx
    dy = domain.parameters.dy

    if include_ghost_cells:
        X_inds = np.arange(-1, domain.num_cells_x + 1)
        Y_inds = np.arange(-1, domain.num_cells_y + 1)
    else:
        X_inds = np.arange(0, domain.num_cells_x)
        Y_inds = np.arange(0, domain.num_cells_y)

    bd_xlow = (-0.5) * dx
    bd_xhigh = (domain.num_cells_x - 0.5) * dx

    bd_ylow = (-0.5) * dy
    bd_yhigh = (domain.num_cells_y - 0.5) * dy

    Y, X = np.meshgrid(Y_inds * dy, X_inds * dx)

    if vmin is None:
        vmin = np.min(domain.field)
    if vmax is None:
        vmax = np.max(domain.field)

    # values
    plt.scatter(
        X + x_offset,
        Y + y_offset,
        c=domain._field.copy() if include_ghost_cells else domain.field.copy(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    # boundary and grid
    plt.plot(
        np.array([bd_xlow, bd_xlow, bd_xhigh, bd_xhigh, bd_xlow]) + x_offset,
        np.array([bd_ylow, bd_yhigh, bd_yhigh, bd_ylow, bd_ylow]) + y_offset,
        "--k",
    )

    lines = np.empty((2, domain.num_cells_x - 1, 2))
    lines[0, :, 1] = bd_ylow
    lines[1, :, 1] = bd_yhigh
    lines[:, :, 0] = np.linspace(
        bd_xlow + dx, bd_xhigh - dx, domain.num_cells_x - 1
    )
    plt.plot(
        lines[..., 0] + x_offset,
        lines[..., 1] + y_offset,
        ":k",
    )
    lines = np.empty((2, domain.num_cells_y - 1, 2))
    lines[0, :, 0] = bd_xlow
    lines[1, :, 0] = bd_xhigh
    lines[:, :, 1] = np.linspace(
        bd_ylow + dy, bd_yhigh - dy, domain.num_cells_y - 1
    )
    plt.plot(
        lines[..., 0] + x_offset,
        lines[..., 1] + y_offset,
        ":k",
    )

    if include_ghost_cells:
        # left ghost
        plt.plot(
            np.array([bd_xlow - dx, bd_xlow - dx]) + x_offset,
            np.array([bd_ylow, bd_yhigh]) + y_offset,
            ":k",
            alpha=0.25,
        )
        plt.plot(
            np.broadcast_to(
                [[bd_xlow - dx], [bd_xlow]], (2, domain.num_cells_y + 1)
            )
            + x_offset,
            np.broadcast_to(
                np.linspace(bd_ylow, bd_yhigh, domain.num_cells_y + 1),
                (2, domain.num_cells_y + 1),
            )
            + y_offset,
            ":k",
            alpha=0.25,
        )

        # right ghost
        plt.plot(
            np.array([bd_xhigh + dx, bd_xhigh + dx]) + x_offset,
            np.array([bd_ylow, bd_yhigh]) + y_offset,
            ":k",
            alpha=0.25,
        )
        plt.plot(
            np.broadcast_to(
                [[bd_xhigh], [bd_xhigh + dx]], (2, domain.num_cells_y + 1)
            )
            + x_offset,
            np.broadcast_to(
                np.linspace(bd_ylow, bd_yhigh, domain.num_cells_y + 1),
                (2, domain.num_cells_y + 1),
            )
            + y_offset,
            ":k",
            alpha=0.25,
        )

        # bottom ghost
        plt.plot(
            np.array([bd_xlow, bd_xhigh]) + x_offset,
            np.array([bd_ylow - dy, bd_ylow - dy]) + y_offset,
            ":k",
            alpha=0.25,
        )
        plt.plot(
            np.broadcast_to(
                np.linspace(bd_xlow, bd_xhigh, domain.num_cells_x + 1),
                (2, domain.num_cells_x + 1),
            )
            + x_offset,
            np.broadcast_to(
                [[bd_ylow - dy], [bd_ylow]], (2, domain.num_cells_x + 1)
            )
            + y_offset,
            ":k",
            alpha=0.25,
        )

        # top ghost
        plt.plot(
            np.array([bd_xlow, bd_xhigh]) + x_offset,
            np.array([bd_yhigh + dy, bd_yhigh + dy]) + y_offset,
            ":k",
            alpha=0.25,
        )
        plt.plot(
            np.broadcast_to(
                np.linspace(bd_xlow, bd_xhigh, domain.num_cells_x + 1),
                (2, domain.num_cells_x + 1),
            )
            + x_offset,
            np.broadcast_to(
                [[bd_yhigh], [bd_yhigh + dy]], (2, domain.num_cells_x + 1)
            )
            + y_offset,
            ":k",
            alpha=0.25,
        )

    plt.gca().set_aspect(1)
