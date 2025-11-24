# APC524 Assignment 6

This assignment covers parallelism, using the heat equation, which we discretize as

$$u^{(k+1)}_{i,j} =u^{(k)}_{i,j} + D(\Delta t)\left(\frac{u^{(k)}_{i-1,j} - 2u^{(k)}_{i,j} + u^{(k)}_{i+1,j}}{\Delta x^2} + \frac{u^{(k)}_{i,j-1} - 2u^{(k)}_{i,j} + u^{(k)}_{i,j+1}}{\Delta y^2}\right) =: u^{(k)}_{i,j} + \Delta t~ F(u^{(k)}, i,j; D, \Delta x, \Delta y) ~~~~~~(\dagger)$$

$D$, $\Delta t$, $\Delta x$, and $\Delta y$ are constants on the grid $(i \Delta x,j \Delta y)\in \Omega_{\text{int}} \subsetneq \Omega$, where $\Omega_{\text{int}}$ is the *discretized interior* of the domain $\Omega\subseteq \mathbb R^2$, on which we solve the heat equation. For any given time step $k$, we can expect $u^{(k)}_{i,j}$ to be known for all $i,j$, leaving a single unknown on the LHS.

In this problem, we will use homogeneous dirichlet boundary conditions ($u|_{\partial\Omega} = 0$), as they are the simplest. Essentially, we will expand the grid by one cell in each direction and set them to zero. We will call these cells "ghost cells". These ghost cells will not be subject to the evolution equation $(\dagger)$.

An explanation of this equation has been provided below, but it is not necessary to know for the sake of this assignment.

## Background

### The Heat Equation

The heat (or diffusion) equation is a parabolic PDE, describing the diffusion of some quantity $u$ across a domain $\Omega \subseteq \mathbb R^2$ over time.

$$\frac{\partial u}{\partial t} = D\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

Here, $D > 0$ is the diffusion constant, which we assume to be uniform across $\Omega$ for the purpose of this assignment. The dimensions of $D$ are $[L]^2/[T]$, with a higher number representing a medium that diffuses $u$ quicker. For instance, the *thermal diffusivity* of copper is approximately $1.1 \times 10^{-4}\, \text{m}^2/\text{s}$, while the *thermal diffusivity* of air is approximately $1.9 \times 10^{-5}\, \text{m}^2/\text{s}$ [[cit]](https://www.engineersedge.com/heat_transfer/thermal_diffusivity_table_13953.htm).

We can describe this on a grid using finite differences:

$$\frac{u^{(k+1)}_{i,j} - u^{(k)}_{i,j}}{\Delta t} = D\left(\frac{u^{(k)}_{i-1,j} - 2u^{(k)}_{i,j} + u^{(k)}_{i+1,j}}{\Delta x^2} + \frac{u^{(k)}_{i,j-1} - 2u^{(k)}_{i,j} + u^{(k)}_{i,j+1}}{\Delta y^2}\right) ~~~~(*)$$

where $u^{(k)}_{i,j}$ approximates $u(i\Delta x, j\Delta y, k\Delta t)$.
This is known as the explicit [forward-time, central-space (FTCS) scheme](https://en.wikipedia.org/wiki/FTCS_scheme), which you may have already encountered or will encounter in APC523. Here, "forward-time" refers to the "forward difference" approximation

$$\frac{\partial u}{\partial t} = \frac{u^{(k+1)}_{i,j} - u^{(k)}_{i,j}}{\Delta t} + O(\Delta t)$$

of the time derivative, while "central-space" refers to the "central difference" scheme for the Laplacian

$$\nabla^2 u = \nabla \cdot \nabla u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = \frac{u^{(k)}_{i-1,j} - 2u^{(k)}_{i,j} + u^{(k)}_{i+1,j}}{\Delta x^2} + \frac{u^{(k)}_{i,j-1} - 2u^{(k)}_{i,j} + u^{(k)}_{i,j+1}}{\Delta y^2} + O(\Delta x^2 + \Delta y^2)$$

This scheme is "stable" if we can guarantee $2D(\Delta t)\left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right) \le 1$. To obtain $(\dagger)$, we utilize the same algebraic manipulation one would employ when deriving Euler's method (which is effectively what the FTCS scheme is).

### Parallelism and Domain Decomposition of the Explicit Heat Equation

Domain decomposition involves splitting the problem across multiple subdomains $\Omega_1 \sqcup \dots \sqcup \Omega_n = \Omega$, and solving it on each subdomain individually, using some way of coupling each problem together where their boundaries intersect. Such a technique is designed for parallel architectures.

Since we are solving the heat equation with an *explicit* scheme (exclusively arithmetic to get from the known $u^{(k)}$ to an unknown $u^{(k+1)}$ -- there is no system of equations to solve for), the coupling is very simple.
If $u^{(k)} _ {i,j}$ is handled by the $\Omega _ 1$ solver and $u^{(k)} _ {i+1,j}$ is handled by the $\Omega _ 2$ solver, we simply need those values to be communicated between the solvers before computing $u^{(k+1)}$. The simplest way of dealing with these is to utilize the ghost cells we also used for boundary conditions. In theory a $u^{(k)} _ {i+1,j}$ in the boundary is referenced as an out-of-domain value, the same as a ghost cell. Thus, we can employ the same mechanism for either.

If instead, we had an implicit scheme, such as the [backward-time, central-space (BTCS)](https://en.wikipedia.org/wiki/Finite_difference_method#Implicit_method) scheme:

$$u^{(k+1)}_{i,j} - D(\Delta t)\left(\frac{u^{(k+1)}_{i-1,j} - 2u^{(k+1)}_{i,j} + u^{(k+1)}_{i+1,j}}{\Delta x^2} + \frac{u^{(k+1)}_{i,j-1} - 2u^{(k+1)}_{i,j} + u^{(k+1)}_{i,j+1}}{\Delta y^2}\right) = u^{(k)}_{i,j}~~,$$

one can imagine needing to employ a parallel solver for the above system of equations, which is far less trivial.

## Your Task

Inside the `parallel_heat` project, you have been provided a serial implementation of the FTCS heat solver that works on a single process / thread. An unsubdivded domain (and perhaps the individual subdomains of a decomposed domain) is managed by `src/parallel_heat/domain.py::DenseDomain`.

A notebook describing how it works is provided in `tutorial/dense_domain.ipynb`.

Additionally, in `src/parallel_heat/chunk_domain.py::ChunkDomain`, we have a solver that takes a collection of `DenseDomain`s and how they line up with each other, and solves the heat equation on the larger domain, handling the couplings between each subdomain. All couplings between `DenseDomain`s will line up perfectly along one edge:

- The length of an edge is the number of cells on the interior along that edge (`num_cells_x` for `Side.TOP` and `Side.BOTTOM` and `num_cells_y` for `Side.LEFT` and `Side.RIGHT`). If the edge of one domain is coupled with another, you can assume that they have the same length.
- You can assume the coupling has no offsets. If `corner_CW` represents the corner cell on the clockwise side of the edge and `corner_CCW` represents the corner cell on the counterclockwise side of the edge, then `corner_CW` on one side will line up with `corner_CCW` of the other.

A notebook describing how it works is provided in `tutorial/chunk_domain.ipynb`.

The tests in `tests/test_dense_domain.py` and `tests/test_chunk_domain.py` exist to validate `DenseDomain` and `ChunkDomain`, respectively. Your task is to write a parallel solver (fill the skeleton in `src/parallel_heat/distributed_domain.py`) whose behavior is the same as `CoupledDomain`, except for the constructor:

- The first argument in `ChunkDomain.__init__` takes in a list of `DenseDomain` parameters. Namely,
  - a list of 3-tuples `(subdomain_index, subdomain_width, subdomain_height)`, where `subdomain_index` is an integer ID used by `subdomain_links`. `subdomain_index` may not be sequential, and it may skip numbers. However, it will never repeat. `subdomain_width` and `subdomain_height` are the number of cells in x and y, respectively, for that subdomain. These values are passed directly into the `DenseDomain` constructor.

- The first argument in `DistributedDomain.__init__`, namely `subdomain_dimensions_per_process`, instead takes a list of these 3-tuple lists.
  - The length of this larger list represents the number of `multiprocessing` `Process`es that should be used.
  - Each process should handle the subdomains given by its corresponding element of `subdomain_dimensions_per_process`. While you do not need to do this, it would simplify your workload to give an element of `subdomain_dimensions_per_process` to each process, where that process passes its element as the first argument of `ChunkDomain`. That way, you only need to worry about the inter-process communication.

For full credit, all the tests in `test` should pass. `tests/test_distributed_domain.py` covers the basic features of the `DistributedDomain`, while `tests/test_distributed_large.py` is a ~1.5 GB simulation that takes much longer to perform. This test will fail if the memory footprint of your solver exceeds 700 MB on the main process, which means that you cannot have a local (often called **Host**) copy of the field values.

A workflow has been provided so that the tests can be run on a GitHub runner. We will be using this to check the correctness of your code, so (while it probably will not be an issue) make sure that the tests pass on `ubuntu-latest` within 10 minutes, as in `.github/workflows/ci.yml`.

To help you, a guide in `tutorial/python_multiprocessing.py` has been provided that can help supplement the course material. You may be able to adapt the example code for your implementation of `DistributedDomain`.
