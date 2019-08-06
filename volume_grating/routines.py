import numpy as np
import sympy.vector as vec
from . import solvers
import inspect
from .utilities.validation import validate_input_numeric
import copy
from tqdm import tqdm
from functools import partial
from .systems import GCS
import scipy.optimize as optimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
            print(result)
            for k in range(Np):
                outputs[k, j, i] = result[k]

    return outputs


def get_de(solver, squeeze=True, **kwargs):
    """
    This function returns diffraction efficients of a given ``solver`` object and ``**kwargs`` for it. It is a convenient functional representation for other purposes such as optimization.

    :param solver: an instance of ``solvers.Responses``.
    :param squeeze: True or False whether to remove singleton dimension of the diffraction efficiency result.
    :param kwargs: ``**kwargs`` that works for ``solvers.Response.get_efficiency_at_points()``.
    :return: diffraction efficiencies in numpy array.
    """
    results = solver.get_efficiency_at_points(**kwargs)
    return np.abs(results[0].squeeze()) if squeeze else results[0]


def objective(params, truth, solver, update_solver_func, get_de_func):
    """
    This is an objective function (or a loss function) for ``fit_de_spectral`` to minimize mean-squared-error (MSE) loss.

    ``params`` is used to update ``solver`` via ``updated_solver_function``. Then, the predicted diffraction efficiency is calculated, via ``get_de_func``. The MSE is calculated between the ground truth DE and the predicted DE.

    The ``update_solver_func(params, solver)`` must be a callable with two arguments, in order, ``params`` and ``solver``. These arguments are the same as for ``objective`` function.

    ``get_de_func(solver)`` should only take one argument. It should return a numpy array for diffraction efficiencies.

    :param params: a 1D numpy array. **The last element of ``params`` is reserved to a DC offset.**
    :param truth: a numpy array with the same dimension as the output from ``get_de_func``.
    :param solver: an instance of ``solvers.Response`` class.
    :param update_solver_func: a callable function with two arguments ``params`` and ``solver``, see explanation above.
    :param get_de_func: a callable function that has only one argument, being ``solver``.
    :return: MSE value
    """
    update_solver_func(params, solver)
    de_pred = get_de_func(solver) + params[-1]
    return mean_squared_error(truth, de_pred)


def fit_de_spectral(wavelengths, de_data, solver, update_solver, initial_guess, bounds, plot=True):
    """
    This function fits a data of diffraction efficiency with prediction calculated via ``solvers.Response`` object.

    The ``update_solver_func(params, solver)`` must be a callable with two arguments, in order, ``params`` and ``solver``. These arguments are the same as for ``objective`` function.

    For example, the following code will fit data with prediction by varying ``dn`` and ``thickness`` of a hologram.
    .. code-block:: python

        # wavelengths are defined in nm
        # de are 1D numpy array from experimental result.

        def update_solver(params, solver):
            dn = params[0]
            t = params[1]
            solver.hologram.dn = dn
            solver.hologram.thickness = t

        x0 = np.array([0.035, 1.5e-6, 0.0])  # x0 has 3 elements because the last element is used to DC-offset.

        result = fit_de_spectral(wavelengths*1e-9, de, solver, update_solver, x0, bounds=((0.01, 0.08), (1e-6, 2e-6), (-0.2, 0.2)))

    :param wavelengths: numpy array for wavelengths in meter.
    :param de_data: numpy array for ground-truth or measured diffraction efficiency
    :param solver: an instance of ``solvers.Response``
    :param update_solver: a callable function with two arguments ``params`` and ``solver``, see explanation above.
    :param initial_guess: the initial guess for ``params`` in ``update_solver``.
    :param bounds: a tuple where each item is of the form (min, max), indicating min and max values for each parameter in ``params``
    :param plot: True or False to plot the fitting result.
    :return: fitting result (from scipy.optimize.minimize)
    """
    get_de_specific = partial(get_de, points=[GCS.origin], wavelengths=wavelengths, verbose=False)
    objective_partial = partial(objective, truth=de_data, solver=solver, update_solver_func=update_solver,
                            get_de_func=get_de_specific)
    result = optimize.minimize(objective_partial, initial_guess, bounds=bounds)
    if not result.success:
        print("***** Fail to fit *****")
    if plot and result.success:
        plt.plot(wavelengths, de_data)
        plt.plot(wavelengths, get_de_specific(solver))
        plt.title([ "{:0.2e}".format(x) for x in result.x])
    return result

