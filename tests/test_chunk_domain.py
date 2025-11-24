import numpy as np
import pytest

from parallel_heat.chunk_domain import ChunkDomain, rectangular_chunk_domain
from parallel_heat.domain import ProblemParamaters


@pytest.fixture(
    params=[
        (16, 16, 4, 4),
        (17, 15, 8, 5),
        (15, 17, 6, 6),
    ],
    ids=[
        "16x16-(4x4)",
        "17x15-(8x5)",
        "15x17-(6x6)",
    ],
)
def gridded_chunk_domain(request):
    ncellx, ncelly, w, h = request.param
    subdomains, links, grid = rectangular_chunk_domain(
        width=ncellx,
        height=ncelly,
        subdiv_size_x=w,
        subdiv_size_y=h,
        return_subdomain_index_grid=True,
    )
    nsub = grid.size
    index_field = np.arange(ncellx * ncelly).reshape((ncellx, ncelly))
    index_field_per_subdomain = [None for _ in range(nsub)]

    for ix in range(grid.shape[0]):
        for iy in range(grid.shape[1]):
            domain_index = grid[ix, iy]
            _, local_w, local_h = subdomains[domain_index]

            index_field_per_subdomain[domain_index] = index_field[
                ix * w : (ix * w + local_w),
                iy * h : (iy * h + local_h),
            ]

    return (
        ChunkDomain(subdomains, links, ProblemParamaters(1, 1, 1, 1)),
        index_field,
        index_field_per_subdomain,
    )


def test_field_set_grid(gridded_chunk_domain):
    domain, index_field, index_field_per_subdomain = gridded_chunk_domain
    nsub = len(domain.subdomains)

    for idomain in range(nsub):
        expect_field = index_field_per_subdomain[idomain]
        domain.set_field(idomain, expect_field)

        obtained_field = domain.subdomains[idomain].field
        np.testing.assert_array_equal(
            obtained_field,
            expect_field,
        )


def test_field_get_grid(gridded_chunk_domain):
    domain, index_field, index_field_per_subdomain = gridded_chunk_domain
    nsub = len(domain.subdomains)

    for idomain in range(nsub):
        expect_field = index_field_per_subdomain[idomain]
        domain.subdomains[idomain].field[...] = expect_field

        obtained_field = domain.get_field(idomain)
        np.testing.assert_array_equal(
            obtained_field,
            expect_field,
        )


def test_decay_dirichlet(
    decay_problem_one_chunk,
    decay_problem_parameters,
    decay_problem_analytical_solution_per_subdomain,
):
    """Verifies the dense_domain with an analytically known homogeneous
    Dirichlet problem."""

    truesol = decay_problem_analytical_solution_per_subdomain

    domain = ChunkDomain(
        *decay_problem_one_chunk, parameters=decay_problem_parameters
    )
    num_domains = len(domain.subdomains)
    for isub in range(num_domains):
        domain.set_field(isub, truesol(isub, 0))

    for istep in range(1, 4):
        domain.step()
        for isub in range(num_domains):
            expect = truesol(isub, istep)
            got = domain.get_field(isub)
            try:
                np.testing.assert_array_almost_equal(
                    got,
                    expect,
                    5,  # 5 sigfigs
                    f"Step {istep}: Analytical solution does not"
                    " decay as expected!",
                )
            except AssertionError as e:
                errs = got - expect
                maxerr = np.max(np.abs(errs))
                print(f"Subdomain {isub} at step {istep}:")
                print(
                    "(y=)| \n"
                    + "\n".join(
                        f"{iy:4d}| "
                        + " ".join(
                            f"{errs[ix, iy] / maxerr:+4.0%}"
                            for ix in range(errs.shape[0])
                        )
                        for iy in range(errs.shape[1] - 1, -1, -1)
                    )
                    + f"\n     {('-' * (5 * errs.shape[0]))}\n"
                    + f"   x= {
                        ' '.join(
                            (f'%{4}d' % (ix)) for ix in range(errs.shape[0])
                        )
                    }"
                )
                raise e
