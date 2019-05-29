import numpy as np
import sympy.vector as vec
from . import solvers
import inspect
from .utilities.validation import validate_input_numeric
import copy
from tqdm import tqdm


def sweep(xs, points, update_func, solver, solver_funcs, verbose=False):
    """
    This functions generalizes parameter sweeping to extract responses from a hologram.

    Given an ndarray ``xs`` of shape ``(M, N)``. ``M`` is the number of parameter sets to sweep. ``N`` is the number of parameters
    in each set.

    The update function ``update_func`` takes ``xs[i]`` and a ``solver`` and it updates the system according to
    the new parameter set ``xs[i]``, and returns an updated ``solver``. That is, the ``update_func`` should have a structure like

    .. code-block:: python

        def update_func(xs, solver):
            # xs is a set of parameters needed to update a solver
            # update solver using xs

            return solver

    For each parameter set ``xs[i]``, the ``sweep`` function update the ``solver`` using ``update_func`` , runs each
    ``solver_func`` for all ``points``, and collects the result in ``outputs``.

    :param xs: ``(M, N)`` ndarray of parameters to sweep. M is the number of parameter sets to sweep. Each set has N parameters.
    :param points: a list of sympy.vector.Point instances representing points of interest on a hologram. ``Np=len(points)``.
    :param update_func: a function that updates a ``solver`` according to a single parameter sweep.
    :param solver: an instance of solvers.Response.
    :param solver_funcs: a list of functions/methods from a solvers.Response class (not instance). ``Nf=len(solver_funcs)``.
    :param verbose: [Default = False] True or False to print out progress.
    :return outputs: ``(Np, Nf, M)`` ndarray
    """

    if inspect.isfunction(solver_funcs):
        solver_funcs = [solver_funcs]
    assert all([hasattr(solver, solver_fun.__name__) and callable(solver_fun) for solver_fun in solver_funcs]),\
        'solver_fun must be a method from class {}'.format(solvers.Response)
    assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of {} instances.'.format(vec.Point)
    assert isinstance(solver, solvers.Response), 'solver must be an instance of {}.'.format(solvers.Response)
    assert inspect.isfunction(update_func), 'update_func must be a function.'
    isvalid, msg = validate_input_numeric(xs, shape=(None, None))
    if not isvalid or xs.size == 0:
        raise Exception(msg + "It can not be empty.")

    Np = len(points)  # number of points on HOE
    M = xs.shape[0]  # number of scan parameter sets
    Nf = len(solver_funcs)  # number of solver functions

    outputs = np.ndarray(shape=(Np, Nf, M))

    solver = copy.deepcopy(solver)

    for i in tqdm(range(M), position=1, disable=not verbose):

        # update a solver, i.e. HOE and playback etc
        solver = update_func(xs[i], solver)

        # loop to consider each solver functions
        for j, func in enumerate(solver_funcs):
            result = func(solver, points, order=solver.order,
                          verbose=False)  # result will be a list of things
            for k in range(Np):
                outputs[k, j, i] = result[k]

    return outputs
