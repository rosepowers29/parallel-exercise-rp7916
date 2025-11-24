import numpy as np
import pytest

from parallel_heat.domain import DenseDomain


@pytest.fixture(
    scope="function",
    params=[
        (6, 6),
        (5, 3),
        (4, 5),
    ],
)
def dense_domain(request, problem_parameters):
    width, height = request.param
    return DenseDomain(width, height, problem_parameters)


def test_domain_init(dense_domain, problem_parameters):
    assert (
        dense_domain.num_cells_x + 2,
        dense_domain.num_cells_y + 2,
    ) == dense_domain._field.shape
    assert dense_domain.parameters == problem_parameters


def test_decay_dirichlet(
    decay_problem_setups, decay_problem_analytical_solution
):
    """Verifies the dense_domain with an analytically known homogeneous
    Dirichlet problem."""

    ncellx, ncelly, _subx, _suby, parameters = decay_problem_setups

    domain = DenseDomain(ncellx, ncelly, parameters)

    Y, X = np.meshgrid(
        (np.arange(domain.num_cells_y) + 1) * domain.parameters.dy,
        (np.arange(domain.num_cells_x) + 1) * domain.parameters.dx,
    )

    def truesol(step):
        return decay_problem_analytical_solution(X, Y, step)

    domain.field[...] = truesol(0)

    for i in range(1, 4):
        domain.step()
        np.testing.assert_array_almost_equal(
            domain.field,
            truesol(i),
            5,  # 5 sigfigs
            "Analytical solution does not decay as expected!",
        )
