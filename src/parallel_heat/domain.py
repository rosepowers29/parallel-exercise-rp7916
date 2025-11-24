from dataclasses import dataclass
from enum import IntEnum

import numpy as np


@dataclass(frozen=True, eq=True, slots=True)
class ProblemParamaters:
    """Specifies the constants used in a given problem. An immutable dataclass
    is used to provide an easy way to ensure compatibility between domains.
    """

    dx: float
    dy: float
    dt: float
    diffusivity: float

    @property
    def stability_criterion(self):
        return (
            2 * self.diffusivity * self.dt * (1 / self.dx**2 + 1 / self.dy**2)
        )


class Side(IntEnum):
    BOTTOM = 0
    RIGHT = 1
    TOP = 2
    LEFT = 3


class DenseDomain:
    # private values to make these readonly

    _problem_parameters: ProblemParamaters

    _domain_width: int
    _domain_height: int
    
    # stores solution at current step, with a ring of ghost cells around it
    _field: np.ndarray

    # permanently allocated temporary storage of intermediate values
    #   (used by the update call)
    _fluxes_horiz: np.ndarray
    _fluxes_vert: np.ndarray

    @property
    def num_cells_x(self):
        return self._domain_width

    @property
    def num_cells_y(self):
        return self._domain_height

    @property
    def width(self):
        return (self.num_cells_x + 1) * self._problem_parameters.dx

    @property
    def height(self):
        return (self.num_cells_y + 1) * self._problem_parameters.dy

    @property
    def parameters(self):
        return self._problem_parameters

    def __init__(
        self,
        width: int,
        height: int,
        parameters: ProblemParamaters,
    ):
        """Initializes a domain to solve the heat equation on. The domain is a
        grid with spatial resolutions `dx` in x and `dy` in y, with the given
        width and height.

        Args:
            width (int): number of grid cells in the x-direction on
                the interior of the domain.
            height (int): number of grid cells in the y-direction on
                the interior of the domain.
            parameters (ProblemParameters): the parameters used
                in the problem that this domain will solve.
        """
        self._domain_width = width
        self._domain_height = height
        self._problem_parameters = parameters

        # we will add a ring of "ghost" cells around the domain so that we can
        # sample outside the grid when computing the Laplacian
        self._field = np.empty(
            (
                self.num_cells_x + 2,
                self.num_cells_y + 2,
            ),
        )

        # intermediate values
        self._fluxes_horiz = np.empty(
            (
                self.num_cells_x + 1,
                self.num_cells_y,
            ),
        )
        self._fluxes_vert = np.empty(
            (
                self.num_cells_x,
                self.num_cells_y + 1,
            ),
        )

        # set dirichlet boundary conditions
        self.ghost_cells_right[:] = 0
        self.ghost_cells_left[:] = 0
        self.ghost_cells_top[:] = 0
        self.ghost_cells_bottom[:] = 0

        # NaN out corners because they will never be used
        # (things will NaN if they do)
        self._field[0, 0] = np.nan
        self._field[0, -1] = np.nan
        self._field[-1, 0] = np.nan
        self._field[-1, -1] = np.nan


    @property
    def field(self):
        return self._field[1:-1, 1:-1]

    @property
    def ghost_cells_bottom(self):
        return self._field[1:-1, 0]

    @property
    def ghost_cells_top(self):
        # flip so we are counterclockwise
        return self._field[1:-1, -1][::-1]

    @property
    def ghost_cells_left(self):
        # flip so we are counterclockwise
        return self._field[0, 1:-1][::-1]

    @property
    def ghost_cells_right(self):
        return self._field[-1, 1:-1]

    def get_edge_interior(self, side: Side, clockwise: bool = False):
        """Retrieves a view of a domain edge (inside).

        Args:
            side (Side): the edge to retrieve from
            clockwise(bool, optional): if True, returns the values in the
                clockwise direction. Otherwise, returns them counterclockwise.
                Defaults to counterclockwise.
        """

        # first get ccw direction
        if side == Side.BOTTOM:
            view = self.field[:, 0]
        elif side == Side.RIGHT:
            view = self.field[-1, :]
        elif side == Side.TOP:
            # flip so we are counterclockwise
            view = self.field[::-1, -1]
        elif side == Side.LEFT:
            # flip so we are counterclockwise
            view = self.field[0, ::-1]
        else:
            raise ValueError
        if clockwise:
            return view[::-1]
        return view

    def get_edge_exterior(self, side: Side, clockwise: bool = False):
        """Retrieves a view of the ghost cells in a domain edge.

        Args:
            side (Side): the edge to retrieve from
            clockwise(bool, optional): if True, returns the values in the
                clockwise direction. Otherwise, returns them counterclockwise.
                Defaults to counterclockwise.
        """
        if side == Side.BOTTOM:
            view = self.ghost_cells_bottom
        elif side == Side.RIGHT:
            view = self.ghost_cells_right
        elif side == Side.TOP:
            view = self.ghost_cells_top
        elif side == Side.LEFT:
            view = self.ghost_cells_left
        else:
            raise ValueError
        if clockwise:
            return view[::-1]
        return view

    def step(self):
        _step_dense_domain(self)


def _step_dense_domain(domain: DenseDomain):
    """Updates the given domain using FTCS. The `field` array is populated
    by the field at the next step using the field values given in the
    `field` array.

    Args:
        domain (Domain): the domain to update
    """
    dx = domain.parameters.dx
    dy = domain.parameters.dy

    # (du/dx)_{i+0.5,j} = (u_{i+1,j} - u_{i,j}) / dx
    domain._fluxes_horiz[:, :] = (
        domain._field[1:, 1:-1] - domain._field[:-1, 1:-1]
    ) / dx

    # (du/dy)_{i,j+0.5} = (u_{i,j+1} - u_{i,j}) / dy
    domain._fluxes_vert[:, :] = (
        domain._field[1:-1, 1:] - domain._field[1:-1, :-1]
    ) / dy

    const_factor = domain.parameters.diffusivity * domain.parameters.dt

    domain._field[1:-1, 1:-1] += (-const_factor / dx) * (
        domain._fluxes_horiz[:-1, :]
    )
    domain._field[1:-1, 1:-1] += (-const_factor / dy) * (
        domain._fluxes_vert[:, :-1]
    )
    domain._field[1:-1, 1:-1] += (const_factor / dx) * (
        domain._fluxes_horiz[1:, :]
    )
    domain._field[1:-1, 1:-1] += (const_factor / dy) * (
        domain._fluxes_vert[:, 1:]
    )
