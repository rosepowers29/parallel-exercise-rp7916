import numpy as np

from .chunk_domain import DomainLink
from .domain import ProblemParamaters


class DistributedDomain:
    # ===================================================================
    #   feel free to add/remove attributes/properties/methods as needed
    # ===================================================================

    num_processes: int

    # =================
    #    populate me
    # =================

    # =================
    #  end populate me
    # =================

    def __init__(
        self,
        subdomain_dimensions_per_process: list[list[tuple[int, int, int]]],
        subdomain_links: list[DomainLink],
        parameters: ProblemParamaters,
    ):
        self.num_processes = len(subdomain_dimensions_per_process)
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================

    def step(self):
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================

    # context manager
    def __enter__(self):
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================
        return self

    def __exit__(self, type, value, traceback):
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================

    def get_field(self, domain_index: int) -> np.ndarray:
        """Returns the field stored in a given domain.

        Args:
            domain_index (int): the index of the domain to retrieve from
        """
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================

    def set_field(self, domain_index: int, field_values: np.ndarray):
        """Sets the field stored in a given domain.

        Args:
            domain_index (int): the index of the domain to set
        """
        # =================
        #    populate me
        # =================
        raise NotImplementedError
        # =================
        #  end populate me
        # =================
