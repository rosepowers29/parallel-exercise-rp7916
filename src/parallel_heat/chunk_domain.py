from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, overload

import numpy as np

from .domain import DenseDomain, ProblemParamaters, Side


@dataclass(frozen=True)
class DomainLink:
    """Represents a connection between two DenseDomains. The domains are
    assumed to be of compatible sizes (the size of side_a of domain_a must
    be equal to that of b), and the domains have the same orientation (along
    the shared edge, one is clockwise while the other is counterclockwise)
    """

    domain_a: int
    side_a: Side
    domain_b: int
    side_b: Side

    def direct_transfer(self, domain_a: DenseDomain, domain_b: DenseDomain):
        """Syncs the two adjacent dense domains by setting the edge ghost cell
        values to the opposite side's edge values.

        Args:
            domain_a (DenseDomain): the domain referenced by self.domain_a
            domain_b (DenseDomain): the domain referenced by self.domain_b
        """
        domain_a.get_edge_exterior(self.side_a)[:] = domain_b.get_edge_interior(
            self.side_b, clockwise=True
        )
        domain_b.get_edge_exterior(self.side_b)[:] = domain_a.get_edge_interior(
            self.side_a, clockwise=True
        )


class ChunkDomain:
    problem_parameters: ProblemParamaters
    subdomains: dict[int, DenseDomain]

    interior_links: list[DomainLink]

    communications: SimpleNamespace

    def __init__(
        self,
        subdomain_dimensions: list[tuple[int, int, int]],
        subdomain_links: list[DomainLink],
        parameters: ProblemParamaters,
    ):
        """Creates a ChunkDomain instance to solve on a coupled set of domains.

        Args:
            subdomain_dimensions (list[tuple[int, int, int]]): a list of tuples
                `(domain_index, width, height)` representing each subdomain and
                the width and height it should be.
            subdomain_links (list[DomainLink]): adjacency information between
                subdomains.
            parameters (ProblemParamaters): the parameters of the problem
        """
        self.problem_parameters = parameters
        self.subdomains = {
            domain_id: DenseDomain(width, height, parameters)
            for domain_id, width, height in subdomain_dimensions
        }

        # ignore all links not between two subdomains inside this chunk
        self.interior_links = []
        for link in subdomain_links:
            if (
                link.domain_a in self.subdomains
                and link.domain_b in self.subdomains
            ):
                self.interior_links.append(link)

    def step(self):
        for link in self.interior_links:
            link.direct_transfer(
                self.subdomains[link.domain_a], self.subdomains[link.domain_b]
            )

        for domain in self.subdomains.values():
            domain.step()

    def get_field(self, domain_index: int) -> np.ndarray:
        """Returns the field stored in a given domain.

        Args:
            domain_index (int): the index of the domain to retrieve from
        """
        return self.subdomains[domain_index].field

    def set_field(self, domain_index: int, field_values: np.ndarray):
        """Sets the field stored in a given domain.

        Args:
            domain_index (int): the index of the domain to set
        """
        self.subdomains[domain_index].field[...] = field_values


@overload
def rectangular_chunk_domain(
    width: int,
    height: int,
    subdiv_size_x: int,
    subdiv_size_y: int,
    return_subdomain_index_grid: Literal[True],
) -> tuple[list[tuple[int, int, int]], list[DomainLink], np.ndarray]: ...


@overload
def rectangular_chunk_domain(
    width: int,
    height: int,
    subdiv_size_x: int,
    subdiv_size_y: int,
    return_subdomain_index_grid: Literal[False] = False,
) -> tuple[list[tuple[int, int, int]], list[DomainLink]]: ...


def rectangular_chunk_domain(
    width: int,
    height: int,
    subdiv_size_x: int,
    subdiv_size_y: int,
    return_subdomain_index_grid: bool = False,
) -> (
    tuple[list[tuple[int, int, int]], list[DomainLink]]
    | tuple[list[tuple[int, int, int]], list[DomainLink], np.ndarray]
):
    """Returns the dimensions and adjacencies of a chunk corresponding to a
    width x height grid subdivided into DenseDomains of size
    subdiv_size_x x subdiv_size_y. If width or height are not divisible, the
    last domain (greatest x or y) is set to a smaller size.

    Args:
        width (int): number of cells total in the x-direction
        height (int): number of cells total in the y-direction
        subdiv_size_x (int): number of cells in a subdivision along x
        subdiv_size_y (int): number of cells in a subdivision along y

    Returns:
        tuple[list[tuple[int,int,int]], list[DomainLink]]: the first two
            arguments for the ChunkDomain constructor.
    """

    nsub_x = width // subdiv_size_x
    if width % subdiv_size_x != 0:
        nsub_x += 1
    nsub_y = height // subdiv_size_y
    if height % subdiv_size_y != 0:
        nsub_y += 1

    def position_to_index(ix, iy):
        if ix < 0 or iy < 0 or ix >= nsub_x or iy >= nsub_y:
            return -1
        return (ix * nsub_y) + iy

    domains = []
    links = []
    for ix in range(nsub_x):
        xlow = ix * subdiv_size_x
        w = min(width - xlow, subdiv_size_x)
        for iy in range(nsub_y):
            index = position_to_index(ix, iy)
            assert index == len(domains)
            ylow = iy * subdiv_size_y
            h = min(height - ylow, subdiv_size_y)
            domains.append((index, w, h))

            adj = position_to_index(ix, iy - 1)
            if adj > index:  # only one polarity is needed; no self-adj
                links.append(DomainLink(index, Side.BOTTOM, adj, Side.TOP))
            adj = position_to_index(ix + 1, iy)
            if adj > index:  # only one polarity is needed; no self-adj
                links.append(DomainLink(index, Side.RIGHT, adj, Side.LEFT))
            adj = position_to_index(ix, iy + 1)
            if adj > index:  # only one polarity is needed; no self-adj
                links.append(DomainLink(index, Side.TOP, adj, Side.BOTTOM))
            adj = position_to_index(ix - 1, iy)
            if adj > index:  # only one polarity is needed; no self-adj
                links.append(DomainLink(index, Side.LEFT, adj, Side.RIGHT))

    if return_subdomain_index_grid:
        index_grid = np.empty((nsub_x, nsub_y), dtype=np.uint64)
        for ix in range(nsub_x):
            for iy in range(nsub_y):
                index_grid[ix, iy] = position_to_index(ix, iy)
        return domains, links, index_grid

    return domains, links
