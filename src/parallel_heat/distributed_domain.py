from __future__ import annotations

import numpy as np
import multiprocessing as mp
from multiprocessing import connection

from dataclasses import dataclass

from parallel_heat.chunk_domain import DomainLink
from parallel_heat.domain import ProblemParamaters, DenseDomain, Side




class DistributedDomain:
    """Parallel solver for the heat equation using multiprocessing."""
    
    def __init__(
        self,
        subdomain_dimensions_per_process: list[list[tuple[int, int, int]]],
        subdomain_links: list[DomainLink],
        parameters: ProblemParamaters,
    ):
        """Initialize the distributed domain solver.
        
        Args:
            subdomain_dimensions_per_process: List of subdomains for each process,
                where each subdomain is (index, width, height)
            subdomain_links: Connections between subdomains
            parameters: Problem parameters
        """
        self.num_processes = len(subdomain_dimensions_per_process)
        self.problem_parameters = parameters
        self.subdomain_links = subdomain_links
        
        # sort domains by their process
        self._dom_proc_dict = {}
        # iterate through the list of lists, each list proc_domains is a list of tuples of dimensions for a domain
        for process_index, process in enumerate(subdomain_dimensions_per_process):
            for domain_tuple in process:
                domain_index = domain_tuple[0]
                self._dom_proc_dict[domain_index] = process_index

        self._create_processes(subdomain_dimensions_per_process, parameters)
    
    #-----------------------------CREATION FUNCTIONS------------------------------#
    
    def _create_pipes(self):
        """Create pipes for cross-process communication."""
        self._pipes = [[] for _ in range(self.num_processes)]
        
        for link in self.subdomain_links:
            proc_a = self._dom_proc_dict[link.domain_a]
            proc_b = self._dom_proc_dict[link.domain_b]
            
            if proc_a == proc_b:
                continue
            
            conn_a, conn_b = mp.Pipe()

            self._pipes[proc_a].append(PipeInfo(link, True, conn_a, proc_b))
            self._pipes[proc_b].append(PipeInfo(link, False, conn_b, proc_a))


    
    def _create_links(self, process_index):
        """
        Gets all the links form all processes.
        Arguments: process_index (int): the process index
        """
        links = []
        for link in self.subdomain_links:
            proc_a = self._dom_proc_dict[link.domain_a]
            proc_b = self._dom_proc_dict[link.domain_b]
            
            # grabs any links to the process
            if process_index in [proc_a, proc_b]:
                links.append(link)
        return links
    
    def _create_processes(self, subdomain_dimensions_per_process, parameters):
        """
        Function for starting processes, to make __init__ cleaner
        Handles initializing queues, creating pipes and initializing the processes

        """
        # initialize a command queue for each process
        self._command_queues = [mp.Queue() for _ in range(self.num_processes)]
        # initialize a single result queue
        self._result_queue = mp.Queue()
        # make the pipes between processes
        self._create_pipes()
        
        # Start worker processes
        self._processes = []
        for process_idx in range(self.num_processes):
            proc_args = (process_idx, subdomain_dimensions_per_process[process_idx], self._create_links(process_idx), parameters,
                         self._command_queues[process_idx], self._result_queue, self._pipes[process_idx])
            process = mp.Process(target=_worker_process, args=proc_args)
            process.start()
            self._processes.append(process)


    #-----------------------------------CONTEXT MANAGERS------------------------------#
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for queue in self._command_queues:
            # the exit signal in the queue
            queue.put({'command': 'exit'})
        
        # join all processes once done
        for process in self._processes:
            process.join()
        
        return False

#------------------------------SOLVER CONTROLS---------------------------#
    def step(self):
        """
        Step the solver forward.
        """
        for queue in self._command_queues:
            queue.put({'command': 'step'})
        
        for _ in range(self.num_processes):
            result = self._result_queue.get()
            if result['status'] != 'step_complete':
                raise RuntimeError(f"Process {result['proc_rank']} failed")

    def get_field(self, domain_index: int) -> np.ndarray:
        """Get field values from a subdomain."""
        proc_rank = self._dom_proc_dict[domain_index]
        
        self._command_queues[proc_rank].put({
            'command': 'get_field',
            'domain_index': domain_index
        })
        
        result = self._result_queue.get()
        if result['status'] != 'field_data':
            raise RuntimeError(f"Failed to get field for domain {domain_index}")
        
        return result['field']

    def set_field(self, domain_index: int, field_values: np.ndarray):
        """Set field values for a subdomain."""
        proc_rank = self._dom_proc_dict[domain_index]
        
        self._command_queues[proc_rank].put({
            'command': 'set_field',
            'domain_index': domain_index,
            'field_values': field_values
        })
        
        result = self._result_queue.get()
        if result['status'] != 'set_complete':
            raise RuntimeError(f"Failed to set field for domain {domain_index}")


#--------------------------------WORKER FUNCTIONS----------------------------#

def _link_helper(subdomain_links: list[DomainLink], subdomains: dict):
    """
    Sort links based on whether they are in the subdomain or not
    
    :param subdomain_links: all links
    :type subdomain_links: list[DomainLink]
    :param subdomains: dict of subdomains
    :type subdomains: dict
    """
    # sort links based on whether they are in the subdomain or not
    links_in = []
    links_out = []

    for link in subdomain_links:
        if link.domain_a in subdomains and link.domain_b in subdomains:
            links_in.append(link)
        elif link.domain_a in subdomains or link.domain_b in subdomains:
            links_out.append(link)

    return links_in, links_out


def _worker_process(
    proc_rank: int,
    subdomain_dimensions: list[tuple[int, int, int]],
    subdomain_links: list[DomainLink],
    parameters: ProblemParamaters,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    pipe_connections: list[PipeInfo],
):
    """
    Docstring for _worker_process
    
    :param proc_rank: The process index
    :type proc_rank: int
    :param subdomain_dimensions: The list of dimensions for a given subdomain
    :type subdomain_dimensions: list[tuple[int, int, int]]
    :param subdomain_links: List of links between domains
    :type subdomain_links: list[DomainLink]
    :param parameters: the problem parameters
    :type parameters: ProblemParamaters
    :param command_queue: the command queue
    :type command_queue: mp.Queue
    :param result_queue: the result queue
    :type result_queue: mp.Queue
    :param pipe_connections: the list of PipeInfo objects containing the communication pipes, links, etc
    :type pipe_connections: list[PipeInfo]
    """
    
    # make a DenseDomain object for each subdomain in the process
    subdomains = {}
    for domain_index, width, height in subdomain_dimensions:
        subdomains[domain_index] = DenseDomain(width, height, parameters)
    
    # sort links based on whether they are within the subdomain or not
    links_in, links_out = _link_helper(subdomain_links, subdomains)
    
    # dict[DomainLink, PipeInfo]
    link_to_pipe_dict = {}
    for pipe_info in pipe_connections:
        link_to_pipe_dict[pipe_info.link] = pipe_info   
    # execution logic
    while True:
        command = command_queue.get()
        
        if command['command'] == 'exit':
            break
        
        elif command['command'] == 'get_field':
            domain_index = command['domain_index']
            field = subdomains[domain_index].field.copy()
            result_queue.put({
                'status': 'field_data',
                'proc_rank': proc_rank,
                'field': field
            })
        
        elif command['command'] == 'set_field':
            domain_index = command['domain_index']
            field_values = command['field_values']
            subdomains[domain_index].field[...] = field_values
            result_queue.put({
                'status': 'set_complete',
                'proc_rank': proc_rank
            })
        
        elif command['command'] == 'step':
            # sync DenseDomains outside of subdomain
            _sync_domains(subdomains, links_out, link_to_pipe_dict)
            
            # sync DenseDomains within subdomain
            for link in links_in:
                link.direct_transfer(
                    subdomains[link.domain_a],
                    subdomains[link.domain_b]
                )
            
            # Update all subdomains
            for subdomain in subdomains.values():
                subdomain.step()
            
            result_queue.put({
                'status': 'step_complete',
                'proc_rank': proc_rank
            })


def _sync_domains(
    subdomains: dict[int, DenseDomain],
    external_links: list[DomainLink],
    link_to_pipe: dict[DomainLink, PipeInfo]
):
    """
    function to sync domains not in the same subdomain (across processes).
    Calls the indirect_transfer method
    
    subdomains ( dict[int, DenseDomain]): the dictionary of subdomains
    external_links (list[DomainLink]): list of links between subdomain and outer domains
    link_to_pipe (dict[DomainLink, PipeInfo]): dictionary connecting the domainlinks to the communication pipes
    """
    
    for link in external_links:
        if link not in link_to_pipe:
            continue
        
        conn_info = link_to_pipe[link]
        pipe = conn_info.connection
        is_side_a = conn_info.side_a
        
        if is_side_a:
            domain = subdomains.get(link.domain_a)
            if domain is None:
                continue
            my_side = link.side_a
        else:
            domain = subdomains.get(link.domain_b)
            if domain is None:
                continue
            my_side = link.side_b
        indirect_transfer(domain, link, pipe, my_side, is_side_a)


def indirect_transfer(domain: DenseDomain, link: DomainLink, pipe: connection.Connection, side, is_a: bool):
    """
    Version of direct_transfer for the case where we have to transfer across processes.
    
    domain (DenseDomain): the domain within the subdomain
    link (DomainLink): the link between domains
    pipe (connection.Connection) the pipe object between domains
    """
    if is_a:
        subdomain_interior= domain.get_edge_interior(side, clockwise=True).copy()
        pipe.send(subdomain_interior)
        outside_interior = pipe.recv()
        domain.get_edge_exterior(side)[:] = outside_interior

    else:
        outside_interior = pipe.recv()
        domain.get_edge_exterior(side)[:] = outside_interior
        subdomain_interior= domain.get_edge_interior(side, clockwise=True).copy()
        pipe.send(subdomain_interior)


@dataclass 
class PipeInfo:
    link: DomainLink
    side_a: bool
    connection: connection.Connection
    remote_process: int
