import numpy as np
import holograms
import sympy.vector as vec
import illumination
from utilities.validation import validate_input_numeric
import inspect
import engines
from engines import get_k_diff_at_points, get_dephase_at_points, get_k_hologram_at_points, \
    get_k_out_off_hologram_at_point
import materials
from tqdm import tqdm


class Response(object):
    """
    ``Response`` is a base instanceable class to compute responses of a hologram. ``playback`` is needed when any diffraction
    responses are queried. ``engine`` is needed when diffraction efficiency is queried.

    :param hologram: an instance of HologramBase subclasses only.
    :param playback: [default = None] an instance of illumination.Playback.
    :param engine: [default = None] an instance of Engine subclasses that defines physics used to calculate diffraction efficiency.
    :param order: an integer to specify a diffraction order to compute.
    """

    def __init__(self, hologram, playback=None, engine=None, order=1):
        self.hologram = hologram
        self.playback = playback
        self.engine = engine
        self.order = order

    @property
    def hologram(self):
        return self._hologram

    @hologram.setter
    def hologram(self, value):
        assert isinstance(value, holograms.HologramBase), 'hologram must be an instance of {} or its subclasses.'.format(holograms.HologramBase)
        self._hologram = value

    @property
    def playback(self):
        return self._playback

    @playback.setter
    def playback(self, value):
        assert value is None or isinstance(value, illumination.Playback), 'playback must be an instance of {}.'.format(illumination.Playback)
        self._playback = value

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, value):
        assert value is None or isinstance(value, engines.Engine), 'engine must be an instance of {} or its subclasses'.format(engines.Engine)
        self._engine = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        assert isinstance(value, int), 'order must be an integer.'
        self._order = value

    def get_k_hologram_at_points(self, points):
        """
        Return a list of grating vectors of a hologram at the specified points.

        :param points: a list of sympy.vector.Point instances.
        :return [k_holo]: a list of sympy.vector.Vector instances.
        """

        assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of {}'.format(vec.Point)

        return get_k_hologram_at_points(self.hologram, points=points)

    def get_k_diff_at_points(self, points, order=None, materail_outside = None):
        """
        Return a list of diffracted k vector against a list of Point instances for a specified order. If the order is
        left None, the order value would be that of the solver instance. The environment outside a hologram is defined
        by a Material instance in ``material_outside``. If ``material_outside`` is ``None``, the returned diffracted
        k vector is inside the hologram. Otherwise, Snell's law will be applied to return diffracted k vectors that are
        already refracted into the environment.

        :param points: a list of sympy.vector.Point instances.
        :param order: [optional, default=None] a diffraction order, must be an integer. If None, orders will be set to order of the solver instance.
        :param material_outside: [default=None] an instance of Material subclasses, representing an environmental material.
        :return [k_diff]: a list of sympy.vector.Vector instances.
        """
        assert materail_outside is None or isinstance(materail_outside, materials.Material), \
            'material_outside must be an instance of {}.'.format(materials.Material)
        if order is None:
            order = self.order

        k_diffs = get_k_diff_at_points(hologram=self.hologram, points=points, playback=self.playback,
                                    order=order)
        if materail_outside is None:
            return k_diffs
        else:
            new_k_diffs = []
            for i in range(len(k_diffs)):
                new_k = get_k_out_off_hologram_at_point(k_diffs[i], self.hologram, points[i], materail_outside)
                new_k_diffs.append(new_k)
            return new_k_diffs

    def get_dephase_at_points(self, points, order=None):
        """
        Returns a list of dephase quantity at a specified point for a specified list of diffraction order.

        :param points: a list of sympy.vector.Point instances
        :param order: [optional, default=None] a diffraction order, must be an integer. If None, orders will be set to order of the solver instance.
        :return [dephase]: a list of dephase (numeric values).
        """
        assert isinstance(order, int) or order is None, 'order must be integer or None. If being None, ' \
                                                        'self.order will be used.'
        if order is None:
            order = self.order
        return get_dephase_at_points(hologram=self.hologram, playback=self.playback, points=points, order=order)

    def get_efficiency_at_points(self, points, order=None, **kwargs):
        """
        Return a ndarray of efficiency against a list of Point instances

        :param points: a list of Point instance
        :param order: [optional, default=None] a diffraction order, must be an integer. If None, orders will be set to order of the solver instance.

        :return [efficiency]: a ndarray of diffraction efficiencies

        ``**kwargs`` parameters:
        - ``verbose``: True or False indicating whether to show progress in calculation.
        - ``wavelengths``: a number, a list, or a ndarray of wavelengths to calculate diffraction efficiency using the given playback geometry.
        """

        assert isinstance(order, int) or order is None, 'order must be integer or None. If being None, ' \
                                                        'self.order will be used.'
        assert self.engine is not None, 'Response object''s engine cannot be None.'
        if order is None:
            order = self.order

        assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of sympy.vector.Point instances.'

        eng = self.engine

        # config tqdm
        verbose = True
        if 'verbose' in list(kwargs.keys()):
            verbose = bool(kwargs['verbose'])

        wavelengths = np.array([self.playback.source.wavelength])
        if "wavelengths" in list(kwargs.keys()):
            wavelengths = np.array(kwargs["wavelengths"])

        efficiency = np.ndarray(shape=(len(points), wavelengths.size), dtype=np.float)

        for i in tqdm(range(len(points)), disable=not verbose, leave=True):
            p = points[i]
            param = eng.extract(hologram=self.hologram, playback=self.playback, point=p, order=self.order,
                                wavelengths=wavelengths)
            eff, _, _ = eng.solve(param)
            efficiency[i] = eff
        return efficiency


class Designer(object):
    """
    ``Designer`` is a base instanceable class to design a recording geometry from a target hologram.

    ``update_func`` must be of a following structure:

    .. code-block:: python

        def update_func(xs, hologram):
            # :param xs: a (1, M) ndarray of a set of parameters to search
            # :param hologram: an instance of HologramBase subclasses.

            # codes that use xs to update a hologram, most likely its recording,
            # which is an instance of illumination.Record

            return hologram # now updated to a parameter set xs
            
    ``loss_func`` must be of a following structure:
    
    .. code-block:: python
        
        def loss_func(xs, update_func, response, points, target, weights):
            # required:

            # :param xs: a (1,M) ndarray of a set of parameters to search
            # :param update_func: a update_func to update a hologram with xs
            # :param response: an instance of solvers.Response
            # :param points: a list of sympy.vector.Point instances, representing points on a hologram
            # :param target: a list of target response, such as grating vectors or diffraction efficiency, with the same length as that of points argument.

            # optionals:
            # :param weights: weight for each item in points. This optional argument must be named exactly ``weights``.

            hoe = response.hologram
            hoe = update_func(xs, hoe)

            # custom codes starts here

            # calculate loss between the target and the candidate

            return loss_value
            

    :param response: an instance of solvers.Response solver.
    :param update_func: a function that will update a hologram against a list of parameters.
    :param loss_func: a function that calculate error/loss metrics between the target hologram and a candidate hologram.
    :param optim_method: a method that ``scipy.optimize.minimize`` recognizes.
    """

    def __init__(self, response, update_func, loss_func, optim_method='BFGS'):
        self.response = response
        self.update_func = update_func
        self.loss_func = loss_func
        self.optim_method = optim_method

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, value):
        assert isinstance(value, Response), 'response must be an instance of {}.'.format(Response)
        self._response = value

    @property
    def update_func(self):
        return self._update_func

    @update_func.setter
    def update_func(self, value):
        assert inspect.isfunction(value), 'update_func must be a function.'
        self._update_func = value

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, value):
        assert inspect.isfunction(value), 'loss_func must be a function.'
        self._loss_func = value

    def run(self, xs_init, points, target, points_weight=np.array([1.0]), method=None):
        """
        Runs the candidate search by minimizing the loss/error value.

        :param xs_init: (N,) ndarray values specifying the initial guess of the parameters to optimize
        :param points: a list of points on the hologram to optimize, must be a list of sympy.vector.Point.
        :param target: a list of target response, such as grating vectors or diffraction efficiency, with the same length as that of points argument.
        :param points_weight: an (N,) ndarry of weight for each hologram points. If a scalar number if provided, the same weight is applied to all points. Default is one.
        :param method: [optional, Default=None] a method that scipy.optimize.minimize uses. If None, self.optim_method will be used.

        :return: a return from ``scipy.optimize.minimize()``
        """
        assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of {} instances.'.format(vec.Point)
        assert isinstance(xs_init, np.ndarray), 'xs_init must be 1D numpy array.'
        if method is None:
            method = self.optim_method

        if isinstance(points_weight, (int, float)):
            points_weight = np.array([points_weight], dtype=np.float)
        isvalid, msg = validate_input_numeric(points_weight, shape=(None,))
        if not isvalid:
            raise ValueError('points_weight must be a number or a list/ndarray of numbers. '+msg)
        if points_weight.shape[0]!=1 and points_weight.shape[0]!=len(points):
            raise ValueError('points_weight must be a number or a list/ndarray of numbers.')

        from scipy import optimize as optim

        if 'weights' in self.loss_func.__code__.co_varnames:
            args = (self.update_func, self.response, points, target, points_weight)
        else:
            args = (self.update_func, self.response, points, target)
        result = optim.minimize(self.loss_func, xs_init,
                                args=args,
                                method=method)

        return result


