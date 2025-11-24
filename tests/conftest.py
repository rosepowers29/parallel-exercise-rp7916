import numpy as np
import pytest

from parallel_heat.chunk_domain import rectangular_chunk_domain
from parallel_heat.domain import ProblemParamaters


@pytest.fixture(
    scope="session",
    params=[
        ProblemParamaters(0.1, 0.2, 0.05, 2),
        ProblemParamaters(0.03, 0.02, 0.01, 0.25),
        ProblemParamaters(1, 1, 0.2, 1),
    ],
)
def problem_parameters(request):
    return request.param


# ==============================================================================
#  Decay Problem
# ==============================================================================
#
# This fixture sets up a test that validates against the analytical solution:
#
# u(x,t) = exp(-(kx**2 + ky**2) * t) * sin(kx * x) * sin(ky * y)
#
# The temporal error can be mostly killed by using the discrete decay rate
# (1 - dt * (kx**2 + ky**2) )^numsteps


# mesh setups: set parameters and domain sizes
@pytest.fixture(
    params=[
        (160, 160, 4, 4, ProblemParamaters(2, 4, 0.4, 2)),
        (170, 150, 8, 8, ProblemParamaters(1, 1, 1, 0.1)),
        (150, 170, 16, 16, ProblemParamaters(0.3, 0.3, 0.005, 1)),
    ],
    ids=[
        "160x160-different_dx_dy",
        "170x150-whole_steps",
        "150x170-same_dx_dy",
    ],
)
def decay_problem_setups(request):
    return request.param


@pytest.fixture
def decay_problem_parameters(decay_problem_setups):
    return decay_problem_setups[-1]


# modal solutions (grid with mesh setups)
@pytest.fixture(params=[(2, 3), (3, 1)], ids=["(2,3)", "(3,1)"])
def decay_problem_modes(request):
    return request.param


# constructed arguments for ChunkDomain
@pytest.fixture
def decay_problem_one_chunk_and_subdomain_indices(decay_problem_setups):
    ncellx, ncelly, subx, suby, parameters = decay_problem_setups
    return rectangular_chunk_domain(
        width=ncellx,
        height=ncelly,
        subdiv_size_x=subx,
        subdiv_size_y=suby,
        return_subdomain_index_grid=True,
    )


# constructed arguments for ChunkDomain
@pytest.fixture
def decay_problem_one_chunk(decay_problem_one_chunk_and_subdomain_indices):
    return decay_problem_one_chunk_and_subdomain_indices[:2]


# analytical solution
@pytest.fixture
def decay_problem_analytical_solution(
    decay_problem_setups, decay_problem_modes
):
    ncellx, ncelly, subx, suby, parameters = decay_problem_setups
    width = (ncellx + 1) * parameters.dx
    height = (ncelly + 1) * parameters.dy

    modex, modey = decay_problem_modes

    wavenumberx = np.pi * modex / width
    wavenumbery = np.pi * modey / height

    diffspeed = parameters.diffusivity * (wavenumberx**2 + wavenumbery**2)

    decay_rate = 1 - parameters.dt * diffspeed

    def solution(x_coords, y_coords, timestep):
        return (
            np.sin(wavenumberx * x_coords)
            * np.sin(wavenumbery * y_coords)
            * decay_rate**timestep
        )

    return solution


# recover true solution per subdomain
@pytest.fixture
def decay_problem_analytical_solution_per_subdomain(
    decay_problem_one_chunk_and_subdomain_indices,
    decay_problem_analytical_solution,
    decay_problem_setups,
):
    ncellx, ncelly, subx, suby, parameters = decay_problem_setups
    grid = decay_problem_one_chunk_and_subdomain_indices[-1]
    num_subdomains = len(decay_problem_one_chunk_and_subdomain_indices[0])
    subdomain_grids = [None for _ in range(num_subdomains)]
    for idomain, width, height in decay_problem_one_chunk_and_subdomain_indices[
        0
    ]:
        ix, iy = np.where(grid == idomain)
        subdomain_grids[idomain] = (
            width,
            height,
            subx * ix[0],
            suby * iy[0],
        )

    def solution(subdomain_index, timestep):
        w, h, xstart, ystart = subdomain_grids[subdomain_index]
        Y, X = np.meshgrid(
            (np.arange(h) + 1 + ystart) * parameters.dy,
            (np.arange(w) + 1 + xstart) * parameters.dx,
        )
        return decay_problem_analytical_solution(X, Y, timestep)

    return solution
