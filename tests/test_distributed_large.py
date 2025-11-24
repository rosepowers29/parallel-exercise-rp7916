import gc
from types import SimpleNamespace

import numpy as np
import psutil

from parallel_heat.chunk_domain import rectangular_chunk_domain
from parallel_heat.distributed_domain import DistributedDomain
from parallel_heat.domain import ProblemParamaters


def get_used_mb():
    return psutil.Process().memory_info().vms / (1024 * 1024)


MAX_ALLOWABLE_MB = 700


def test_large_distributed():
    # another decay problem, but with a very large domain
    params = ProblemParamaters(dx=0.01, dy=0.01, dt=0.0001, diffusivity=0.2)

    modex, modey = 3, 5
    ncellx, ncelly = 20000, 10000
    subx, suby = 2000, 2000

    subdomains, links, grid = rectangular_chunk_domain(
        width=ncellx,
        height=ncelly,
        subdiv_size_x=subx,
        subdiv_size_y=suby,
        return_subdomain_index_grid=True,
    )
    subdomains_dict = {sub[0]: sub for sub in subdomains}
    num_subdomains = len(subdomains)
    subdomain_gridparams = [SimpleNamespace() for _ in range(num_subdomains)]
    left_chunk = []
    right_chunk = []
    for ix in range(grid.shape[0]):
        for iy in range(grid.shape[1]):
            domain_ind = grid[ix, iy]
            _, width, height = subdomains_dict[domain_ind]
            gridparams = subdomain_gridparams[domain_ind]
            gridparams.xind_start = ix * subx
            gridparams.xind_end = gridparams.xind_start + width
            gridparams.yind_start = iy * suby
            gridparams.yind_end = gridparams.yind_start + height

            if ix < grid.shape[0] // 2:
                left_chunk.append((domain_ind, width, height))
            else:
                right_chunk.append((domain_ind, width, height))

    # analytical solution
    width = (ncellx + 1) * params.dx
    height = (ncelly + 1) * params.dy
    wavenumberx = np.pi * modex / width
    wavenumbery = np.pi * modey / height
    diffspeed = params.diffusivity * (wavenumberx**2 + wavenumbery**2)
    decay_rate = 1 - params.dt * diffspeed

    def truesol(domain_index, timestep):
        gridparams = subdomain_gridparams[domain_index]
        y_coords, x_coords = np.meshgrid(
            (np.arange(gridparams.yind_start, gridparams.yind_end) + 1)
            * params.dy,
            (np.arange(gridparams.xind_start, gridparams.xind_end) + 1)
            * params.dx,
        )
        return (
            np.sin(wavenumberx * x_coords)
            * np.sin(wavenumbery * y_coords)
            * decay_rate**timestep
        )

    memory_overhead = get_used_mb()

    with DistributedDomain(
        [left_chunk, right_chunk], links, params
    ) as distributed_domain:
        for isub in range(num_subdomains):
            distributed_domain.set_field(isub, truesol(isub, 0))

        for istep in range(3):
            if istep != 0:
                distributed_domain.step()

            gc.collect()
            total_mem = get_used_mb()
            memory_used = total_mem - memory_overhead
            if memory_used > MAX_ALLOWABLE_MB:
                e = RuntimeError(
                    f"Maximum memory budget of main process exceeded! "
                    f"({memory_used:.0f} MB > {MAX_ALLOWABLE_MB:.0f} MB)"
                )
                e.add_note(
                    "Measured the test's memory overhead to be "
                    f"{memory_overhead:.0f} MB.\n"
                    f"At step {istep}, the total "
                    f"memory is {total_mem:.0f} MB (overhead "
                    f"+ {memory_used:.0f} MB) on the main process."
                )
                full_fieldmem = ncellx * ncelly * 8 / (1024 * 1024)
                e.add_note(
                    f"Storing the entire field ({ncellx} x {ncelly}) with "
                    "doubles (8 B) takes"
                    f" {full_fieldmem:.0f} MB. The main process has taken "
                    f"{memory_used / full_fieldmem:.1%} of that."
                )
                raise e

            for isub in range(num_subdomains):
                expect = truesol(isub, istep)
                got = distributed_domain.get_field(isub)
                np.testing.assert_array_almost_equal(
                    got,
                    expect,
                    5,  # 5 sigfigs
                    f"Step {istep}, domain {isub}: Analytical solution does not"
                    " decay as expected!",
                )
                del expect
                del got
