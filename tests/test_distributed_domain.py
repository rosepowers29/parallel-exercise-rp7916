import numpy as np
import pytest

from parallel_heat.chunk_domain import DomainLink
from parallel_heat.distributed_domain import DistributedDomain
from parallel_heat.domain import DenseDomain, ProblemParamaters, Side


@pytest.mark.parametrize(
    "subdomains",
    [
        [[(0, 4, 5)]],
        [[(0, 4, 5)], [(1, 3, 6)]],
        [[(2, 3, 3)], [(1, 2, 4), (0, 3, 3)], [(3, 2, 2)]],
    ],
    ids=["singleton", "1+1", "1+2+1"],
)
def test_get_and_set_field(subdomains):
    with DistributedDomain(
        subdomains, [], ProblemParamaters(1, 1, 1, 1)
    ) as domain:
        subdomain_fields = {}
        npoints = 0
        for proc in subdomains:
            for sub in proc:
                i, w, h = sub
                subdomain_fields[i] = np.arange(
                    npoints, npoints + w * h
                ).reshape((w, h))
                npoints += w * h

        # set, then immediate get
        for idomain, field_expect in subdomain_fields.items():
            domain.set_field(idomain, field_expect)

            field_got = domain.get_field(idomain)

            np.testing.assert_array_equal(
                field_got,
                field_expect,
                f"Setting and getting field on domain {idomain}: incorrect field",
            )

        # next solutions will be offset by +npoints

        # set fields
        for idomain, field in subdomain_fields.items():
            domain.set_field(idomain, field + npoints)

        # get fields
        for idomain, field in subdomain_fields.items():
            field_got = domain.get_field(idomain)
            field_expect = field + npoints

            np.testing.assert_array_equal(
                field_got,
                field_expect,
                f"Retrieving domain {idomain} after setting all fields -- "
                f"{idomain} is the first incorrect field.\n"
                "Retrieval immediately after set was correct... Was there any "
                "cross contamination?",
            )


def test_single_subdomain(problem_parameters):
    # with this, u_{i,j} <= u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}
    sidelen = 3
    init_cond = np.arange(sidelen * sidelen).reshape((sidelen, sidelen))

    ref_sol = DenseDomain(sidelen, sidelen, problem_parameters)
    ref_sol.field[...] = init_cond
    ref_sol.step()

    expected_sol = ref_sol.field

    with DistributedDomain(
        [[(0, sidelen, sidelen)]],
        [],
        problem_parameters,
    ) as domain:
        domain.set_field(0, init_cond)

        domain.step()

        got_sol = domain.get_field(0)
        np.testing.assert_array_almost_equal(
            got_sol,
            expected_sol,
            err_msg="DistributedDomain for a single subdomain does not "
            "match the solution for DenseDomain",
        )


@pytest.mark.parametrize(
    "same_process", [True, False], ids=["one_process", "separate_processes"]
)
@pytest.mark.parametrize("side_a", Side, ids=[s.name for s in Side])
@pytest.mark.parametrize("side_b", Side, ids=[s.name for s in Side])
def test_single_connections(side_a, side_b, same_process):
    # with two elements, see if one step changes values as expected across
    # the interface

    params = ProblemParamaters(1, 1, 1, 1)
    # with this, u_{i,j} <= u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}

    sidelen = 3

    def gen_interface_array(side: Side):
        # [ 1,  2,  3]
        array = np.zeros((sidelen, sidelen))
        edge_arr = np.arange(sidelen) + 1
        if side == Side.BOTTOM:
            array[:, 0] = edge_arr
        elif side == Side.RIGHT:
            array[-1, :] = edge_arr
        elif side == Side.TOP:
            array[:, -1] = np.flip(edge_arr)
        else:
            array[0, :] = np.flip(edge_arr)
        return array

    def gen_solution_array(side: Side):
        #   [-3, -2, -1]
        # 0 [ 1,  2,  3] 0
        # 0 [ 0,  0,  0] 0
        # 0 [ 0,  0,  0] 0
        #     0   0   0

        array = np.zeros((sidelen, sidelen))
        edge = np.array([-4, -4, -8])
        mid = np.array([1, 2, 3])

        if side == Side.BOTTOM:
            array[:, 0] = edge
            array[:, 1] = mid
        elif side == Side.RIGHT:
            array[-1, :] = edge
            array[-2, :] = mid
        elif side == Side.TOP:
            array[:, -1] = np.flip(edge)
            array[:, -2] = np.flip(mid)
        else:
            array[0, :] = np.flip(edge)
            array[1, :] = np.flip(mid)
        return array

    with DistributedDomain(
        [[(0, sidelen, sidelen), (1, sidelen, sidelen)]]
        if same_process
        else [[(0, sidelen, sidelen)], [(1, sidelen, sidelen)]],
        [DomainLink(domain_a=0, side_a=side_a, domain_b=1, side_b=side_b)],
        params,
    ) as domain:
        domain.set_field(0, -gen_interface_array(side_a))
        domain.set_field(1, gen_interface_array(side_b))

        domain.step()

        field_a = domain.get_field(0)
        expected = -gen_solution_array(side_a)
        np.testing.assert_array_almost_equal(
            field_a,
            expected,
            err_msg="There is an error with the domain[1] to domain[0] "
            f"coupling. side_a = {side_a}, side_b = {side_b}, "
            f"same_process = {same_process}",
        )

        field_b = domain.get_field(1)
        expected = gen_solution_array(side_b)
        np.testing.assert_array_almost_equal(field_b, expected)


def run_problem(chunks, links, decay_problem_parameters, truesol):
    domain = DistributedDomain(
        subdomain_dimensions_per_process=chunks,
        subdomain_links=links,
        parameters=decay_problem_parameters,
    )
    num_domains = max(max(dom[0] for dom in chunk) for chunk in chunks) + 1

    with domain:
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


def test_decay_dirichlet_singlechunk(
    decay_problem_one_chunk,
    decay_problem_parameters,
    decay_problem_analytical_solution_per_subdomain,
):
    domains, links = decay_problem_one_chunk
    run_problem(
        [domains],
        links,
        decay_problem_parameters,
        decay_problem_analytical_solution_per_subdomain,
    )


def test_decay_dirichlet_distributed(
    decay_problem_one_chunk_and_subdomain_indices,
    decay_problem_parameters,
    decay_problem_analytical_solution_per_subdomain,
):
    # test with 4 chunks
    domains, links, grid = decay_problem_one_chunk_and_subdomain_indices

    chunk_dist = np.zeros(grid.shape, dtype=np.uint64)
    chunk_dist[(grid.shape[0] // 2) :, :] += 1
    chunk_dist[:, (grid.shape[1] // 2) :] += 2

    domain_dict = {
        domain_index: (domain_index, width, height)
        for domain_index, width, height in domains
    }

    #  2 | 3
    #  -----
    #  0 | 1

    chunks = [[] for _ in range(4)]
    for ix in range(grid.shape[0]):
        for iy in range(grid.shape[1]):
            chunks[chunk_dist[ix, iy]].append(domain_dict[grid[ix, iy]])

    run_problem(
        chunks,
        links,
        decay_problem_parameters,
        decay_problem_analytical_solution_per_subdomain,
    )
