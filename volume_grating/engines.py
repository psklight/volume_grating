# TODO KogelNikThreeWave

import numpy as np
import sympy.vector as vec
from . import holograms
from . import illumination
from .systems import GCS
from .utilities.validation import validate_input_numeric
from .utilities.optics import snell_law_k_space
from .utilities.geometry import vector_to_ndarray


class Engine(object):
    """
    ``Engine`` is a base class for all functional computation engines.

    """
    def __init__(self):
        pass

    @staticmethod
    def extract(hologram, playback, point, order=1, **kwargs):
        """
        **Abstract method**. Subclasses must override this method with the exact same input arguments. It must return
        a dict containing relevant parameters needed to calculate diffraction efficiency using ``solve()`` method.

        :param hologram: an instance of class or subclass of Holograms
        :param point: an instance of class sympy.vector.Point
        :param playback: an instance of a Playback class.
        :param order:  an integer specifying what order to diffract into
        :return: a dict of parameters
        """
        raise Exception('This method needs override implementation for {} subclasses.'.format(Engine))

    @staticmethod
    def solve(param, **kwargs):
        """
        **Abstract method**. Subclasses must override this method. The first required input argument should be ``param``,
        which should be an output from ``extract()`` method. More input arguments can be added, but they must have default
        values. This is because, in this release, ``Response.get_efficiency_at_points()`` only call ``.solver(param)`` without
        other additional arguments.

        :param param: a dict generated from ``extract()`` method.
        :return: a dict of parameters
        """
        raise Exception('This method needs override implementation for {} subclasses.'.format(Engine))


class KogelnikAnalytical(Engine):
    """
    ``KogelnikAnalytical`` assume a **lossless hologram.** Based on the paper Kogelnik, 1969, Coupled Wave Theory for Thick Hologram Gratings, Bell System
    Technical Journal, Vol 48, No 9. It uses ananlytical equations to solver for diffraction efficiency.
    This assumes that the surface of the hologram is perpendicular to the z axis,
    as defined in the paper. **The recording wave vectors of the holograms must NOT have a y-axis component**.

    """

    @staticmethod
    def solve(param):
        """
        Solve for diffraction efficiency of a set of parameters which must be extracted from the class' ``extract`` method.

        :param param: a dict containing important parameters extracted from a set of hologram and playback.
        :return efficiency: a numeric value of diffraction efficiency
        :return mode: a mode of a hologram, i.e. being transmissive or reflective
        :return caches: a dict of relevant calculated parameters for a particular engine. It is useful to debug.
        """

        mode = param['mode']
        k_r = param['k_r']
        k_d = param['k_d']
        K = param['K']
        thickness = param['thickness']
        n0 = param['n0']
        dn = param['dn']
        wavelength = param['wavelength']

        # from approximation of (n0+dn)^2 ~ n0^2 + 2*n0*dn, so ep0 = n0^2, ep1~n0*dn
        ep0 = n0 ** 2
        ep1 = 2 * dn * n0
        k0 = 2 * np.pi / wavelength  # free space wavenumber
        beta = k0 * n0
        kappa = 0.25 * (k0 * ep1 / np.sqrt(ep0))  # Eq. 6

        k_d_size = np.sqrt(np.sum(k_d ** 2, axis=-1))
        dephase = (beta ** 2 - k_d_size ** 2) / 2 / beta  # Eq. 17

        cr = k_r[:, -1] / beta
        cs = k_d[:, -1] / beta

        # gamma1 and gamma2, Eq. 30
        gamma1 = -0.5 * (1j * dephase / cs) + 0.5 * np.sqrt((-1j * dephase / cs) ** 2 - 4 * kappa ** 2 / cr / cs)
        gamma2 = -0.5 * (1j * dephase / cs) - 0.5 * np.sqrt((-1j * dephase / cs) ** 2 - 4 * kappa ** 2 / cr / cs)

        if mode is "transmission":  # transmission
            # Eq. 34
            s = 1j * kappa / cs / (gamma1 - gamma2) * (np.exp(gamma2 * thickness) - np.exp(gamma1 * thickness))
        if mode is "reflection":
            # Eq. 39
            denum = 1j * dephase + cs * (gamma1 * np.exp(gamma2 * thickness) - gamma2 * np.exp(gamma1 * thickness)) / (
                        np.exp(gamma2 * thickness) - np.exp(gamma1 * thickness))
            s = -1j * kappa / denum

        efficiency = np.abs(cs) / cr * np.abs(s) ** 2
        caches = {'kappa': kappa,
                  'dephase': dephase,
                  'cr': cr,
                  'cs': cs,
                  'S': s}
        return efficiency, mode, caches

    @staticmethod
    def extract(hologram, playback, point, wavelengths=None, order=1, **kwargs):
        """
        Extract relevant parameters from a set of hologram and playback for a single point on a hologram and at a
        specified diffraction order. If ``wavelengths`` is None, it will use a wavelength of a source defined in the
        playback.

        :param hologram: an instance of class or subclass of Holograms
        :param point: an instance of class sympy.vector.Point
        :param playback: an instance of a Playback class.
        :param wavelengths: a number, or a list of numbers, or a ndarray of number for wavelengths.
        :param order:  an integer specifying what order to diffract into
        """
        assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.' \
            .format(holograms.Hologram)
        assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(
            illumination.Playback)
        assert isinstance(point, vec.Point), 'point must be an instance of {}.'.format(vec.Point)

        assert isinstance(order, int), 'order must be an integer.'

        if wavelengths is None:
            wavelengths = np.array([playback.source.wavelength])
        if isinstance(wavelengths, (int, float, list)):
            wavelengths = np.array([wavelengths])
        isvalid, msg = validate_input_numeric(wavelengths)
        if not isvalid:
            raise Exception('wavelengths is invalid. ' + msg)
        wavelengths = np.squeeze(wavelengths)
        if wavelengths.ndim == 0:
            wavelengths = np.expand_dims(wavelengths, axis=-1)

        mode = hologram.mode
        K = get_k_hologram_at_points(hologram=hologram, points=[point])[0]
        K = np.expand_dims(vector_to_ndarray(K, GCS), axis=0)
        n0 = hologram.n0(wavelengths)
        n0 = np.squeeze(n0)
        if n0.ndim == 0:
            n0 = np.expand_dims(n0, axis=-1)
        dn = hologram.dn
        thickness = hologram.thickness
        k_r = get_source_k_into_hologram_at_point(point=point, hologram=hologram, source=playback.source)
        k_r = np.repeat(np.expand_dims(vector_to_ndarray(k_r.normalize(), GCS), axis=0), wavelengths.size, axis=0)

        for i, val in enumerate(wavelengths):
            k_r[i] = k_r[i] * 2 * np.pi / val * n0[i]

        k_d = k_r - order * K
        param = {'mode': mode,
                 'thickness': thickness,
                 'n0': n0,
                 'dn': dn,
                 'k_r': k_r,
                 'k_d': k_d,
                 'K': K,
                 'order': order,
                 'wavelength': wavelengths}

        return param


# This is old. It only works with a single wavelength specified by the playback's source object.
# class KogelnikAnalytical(Engine):
#     """
#     ``KogelnikAnalytical`` assume a **lossless hologram.** Based on the paper Kogelnik, 1969, Coupled Wave Theory for Thick Hologram Gratings, Bell System
#     Technical Journal, Vol 48, No 9. It uses ananlytical equations to solver for diffraction efficiency.
#     This assumes that the surface of the hologram is perpendicular to the z axis,
#     as defined in the paper. **The recording wave vectors of the holograms must NOT have a y-axis component**.
#
#     """
#
#     @staticmethod
#     def solve(param, **kwargs):
#         """
#         Solve for diffraction efficiency of a set of parameters which must be extracted from the class' ``extract`` method.
#
#         :param param: a dict containing important parameters extracted from a set of hologram and playback.
#         :return efficiency: a numeric value of diffraction efficiency
#         :return mode: a mode of a hologram, i.e. being transmissive or reflective
#         :return caches: a dict of relevant calculated parameters for a particular engine. It is useful to debug.
#         """
#
#         mode = param['mode']
#         k_r = param['k_r']
#         k_d = param['k_d']
#         K = param['K']
#         thickness = param['thickness']
#         n0 = param['n0']
#         dn = param['dn']
#         wavelength = param['wavelength']
#
#         # from approximation of (n0+dn)^2 ~ n0^2 + 2*n0*dn, so ep0 = n0^2, ep1~n0*dn
#         ep0 = n0**2
#         ep1 = 2*dn*n0
#         k0 = 2*np.pi/wavelength  # free space wavenumber
#         beta = k0*n0
#         kappa = 0.25*(k0*ep1/np.sqrt(ep0))  # Eq. 6
#         dephase = complex( (beta**2-float(k_d.magnitude())**2) / 2 / beta)  # Eq. 17
#         cr = float(k_r.dot(GCS.k))/beta  # Eq. 23
#         cs = float(k_d.dot(GCS.k))/beta  # Eq. 23
#
#         # gamma1 and gamma2, Eq. 30
#         gamma1 = -0.5 * (1j * dephase / cs) + 0.5 * np.sqrt((-1j * dephase / cs) ** 2 - 4 * kappa ** 2 / cr / cs)
#         gamma2 = -0.5 * (1j * dephase / cs) - 0.5 * np.sqrt((-1j * dephase / cs) ** 2 - 4 * kappa ** 2 / cr / cs)
#
#         if mode is "transmission":  # transmission
#             # Eq. 34
#             s = 1j*kappa/cs/(gamma1-gamma2)*(np.exp(gamma2*thickness)-np.exp(gamma1*thickness))
#         if mode is "reflection":
#             # Eq. 39
#             denum = 1j*dephase + cs*(gamma1*np.exp(gamma2*thickness)-gamma2*np.exp(gamma1*thickness))/(np.exp(gamma2*thickness)-np.exp(gamma1*thickness))
#             s = -1j*kappa/denum
#
#         efficiency = np.abs(cs)/cr*np.abs(s)**2
#         caches = {'kappa': kappa,
#                  'dephase': dephase,
#                  'cr': cr,
#                  'cs': cs,
#                  'S': s}
#         return efficiency, mode, caches
#
#     @staticmethod
#     def extract(hologram, playback, point, order=1):
#         """
#         Extract relevant parameters from a set of hologram and playback for a single point on a hologram and at
#         a specified diffraction order.
#
#         :param hologram: an instance of class or subclass of Holograms
#         :param point: an instance of class sympy.vector.Point
#         :param playback: an instance of a Playback class.
#         :param order:  an integer specifying what order to diffract into
#         """
#         assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.' \
#             .format(holograms.Hologram)
#         assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(
#             illumination.Playback)
#         assert isinstance(point, vec.Point), 'point must be an instance of {}.'.format(vec.Point)
#
#         assert isinstance(order, int), 'order must be an integer.'
#
#         mode = hologram.mode
#         K = get_k_hologram_at_points(hologram=hologram, points=[point])[0]
#         n0 = hologram.n0([playback.source.wavelength])[0]
#         dn = hologram.dn
#         thickness = hologram.thickness
#         k_r = get_source_k_into_hologram_at_point(point=point, hologram=hologram, source=playback.source)
#         k_d = get_k_diff_at_points(hologram, playback, [point], order)
#         k_d = k_d[0]
#         param = {'mode': mode,
#                  'thickness': thickness,
#                  'n0': n0,
#                  'dn': dn,
#                  'k_r': k_r,
#                  'k_d': k_d,
#                  'K': K,
#                  'order': order,
#                  'wavelength': playback.source.wavelength}
#         return param


class KogelnikTwoWave(Engine):
    """
    KogelnikTwoWave assumes a **lossless hologram**. It is based on a vectorial version of the coupled wave theory and only consider two-wave interaction.
    This allows a polarization effect and the y-component of the wave vector. The engine relies on scipy.integrate.solve_bvp
    to solve a boundary condition problem for ordinary differential equation (ODE) systems.

    Let's denote the electric-field complex amplitudes of the reference field and the diffracted field as Ar and Ad.
    Assume that the hologram's surface starts at z=0 and lies on the xy plane. The hologram thickness extends in z-axis.
    The field state can be represented by a matrix [Ar, Ad], i.e. a (2,) ndarray. This field state evolves as the fields propagate in z-direction.

    The evolution is a system of ODE that relate [Ar, Ad] to [dAr/dz, dAd/dz], and it is typically via a 2x2 matrix.
    The element of the matrix depends on many parameters including interacting wave vectors, grating vectors, index modulation, etc.
    """

    # TODO add effects of polarization

    @staticmethod
    def solve(param, bc_guess=None, tol=0.01, mesh_number=2, max_nodes=10000, verbose=1):
        """
        Solve for diffraction efficiency of a set of parameters which must be extracted from the class' ``extract`` method.

        Generally, only ``param`` is needed. However, if the ``scipy.integrate.solve_bvp`` fails to find a solution. It is still
        possible to locate a solution by adjusting ``solve_bvp``s arguments, including ``bc_guess``, ``mess_number``,
        ``max_nodes``, and ``tol``. Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html#scipy.integrate.solve_bvp for more information.

        ``bc_guess`` specifies an initial guess on the value of the field states at z=0 and z=thickness. It is a (2,2) ndarray.
        ``bc_guess[0] = np.array([1.0, 0.99], dtype=np.complex)`` guesses that at z=0, Ar = 1.0 and at z=thickness, Ar=0.99.
        ``bc_guess[1] = np.array([0.05, 0], dtype=np.complex)`` guesses that at z=0, Ad=0.05 and at z=thickness, Ad=0.0, which is
        a reasonble initial guess for a reflection hologram.

        :param param: a set of parameters needed to calculate efficiencies. Extract them from .extract(...)
        :param bc_guess: an initial guess for the boundary condition.
        :param tol: a tolerance value for ``solve_bvp``. Default = 0.01.
        :param mesh_number: The number of mesh points in the independent axis to start with. It must be an integer larger than one.
        :param max_nodes: the maximum number of nodes used in attempt to solve the problem. Solver will terminate when the number of mesh nodes is exceeded.
        :param verbose: verbose option for scipy.integrate.solve_bvp. 0 for silent, 1 for result only, 2 for detail.

        :return eff: efficiency of the hologram. If ``solve_bvp`` fails to converge, None is returned.
        :return mode: hologram mode, i.e. reflection or transmission
        :return caches: a dictionary. 'solver detail' for result from ``solve_bvp`` function and ``matrix`` for the interaction matrix.
        """

        # handle boolean verbose sent from solver.Response
        if isinstance(verbose, bool):
            verbose = int(verbose)

        assert bc_guess is None or bc_guess.shape == (2,2), 'bc_guess must be None or a (2,2) ndarray of type np.complex.'
        assert isinstance(mesh_number, int) and mesh_number>1, 'mesh_number must be an integer larger than 1.'
        assert verbose in (0,1,2), 'verbose must be either 0, 1, or 2.'
        assert isinstance(max_nodes, int) and max_nodes>0, 'max_nodes must be positive integer.'

        mode = param['mode']
        k_r = param['k_r']/1e6  # reference wave vector, now in 1/um unit
        k_d = param['k_d']/1e6  # diffracted wave vector, now in 1/um unit
        K = param['K']/1e6  # grating vector, now in 1/um unit
        thickness = param['thickness']/1e-6  # now in um
        n0 = param['n0']
        dn = param['dn']
        wavelength = param['wavelength']/1e-6  # now in um

        # from approximation of (n0+dn)^2 ~ n0^2 + 2*n0*dn, so ep0 = n0^2, ep1~n0*dn
        ep0 = n0 ** 2
        ep1 = 2*dn * n0
        k0 = 2 * np.pi / wavelength  # free space wavenumber
        beta = k0 * n0

        gamma0 = k0**2*ep0
        gamma = k0**2* ep1 / 2
        kz_r = float(k_r.dot(GCS.k))
        kz_d = float(k_d.dot(GCS.k))

        global matrix
        matrix = np.ndarray(shape=(2, 2), dtype=np.complex)
        matrix[0, 0] = -(gamma0 - beta**2) / 2 / 1j / kz_r
        matrix[0, 1] = -np.conj(gamma)/2/1j/kz_r
        matrix[1, 0] = -gamma/2/1j/kz_d
        matrix[1, 1] = -(gamma0-(float(k_d.magnitude()))**2)/2/1j/kz_d
        matrix = np.reshape(matrix, newshape=(matrix.size,))

        z = np.linspace(0.0, thickness, mesh_number)

        if mode == "transmission":
            y_guess = np.ndarray(shape=(2, z.size), dtype=np.complex)
            y_guess[0] = np.linspace(1.0, 0.95, z.size, dtype=np.complex)
            y_guess[1] = np.linspace(0.0, 0.05, z.size, dtype=np.complex)
            boundary_func = KogelnikTwoWave._boundary_for_transmission

        if mode == "reflection":
            y_guess = np.ndarray(shape=(2, z.size), dtype=np.complex)
            y_guess[0] = np.linspace(1.0, 0.95, z.size, dtype=np.complex)
            y_guess[1] = np.linspace(0.05, 0.0, z.size, dtype=np.complex)
            boundary_func = KogelnikTwoWave._boundary_for_reflection

        if bc_guess is not None:
            y_guess = bc_guess

        from scipy.integrate import solve_bvp

        result_detail = solve_bvp(fun=KogelnikTwoWave._rhs,
                                  bc=boundary_func, x=z, y=y_guess, verbose=verbose, max_nodes=max_nodes,
                                  tol=tol)

        if result_detail.success:
            if mode == "transmission":
                efficiency = result_detail.y[1, -1]
            else:
                efficiency = result_detail.y[1, 0]
            efficiency = np.abs(efficiency)**2
            efficiency *= kz_d/kz_r
        else:
            efficiency = None

        caches = {'solver detail': result_detail,
                  'matrix': matrix}

        return efficiency, mode, caches

    @staticmethod
    def extract(hologram, playback, point, order=1, **kwargs):
        """
        Extract relevant parameters from a set of hologram and playback for a single point on a hologram and at
        a specified diffraction order.

        :param hologram: an instance of class or subclass of Holograms
        :param point: an instance of class sympy.vector.Point
        :param playback: an instance of a Playback class.
        :param order:  an integer specifying what order to diffract into
        """
        assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.' \
            .format(holograms.Hologram)
        assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(
            illumination.Playback)
        assert isinstance(point, vec.Point), 'point must be an instance of {}.'.format(vec.Point)

        assert isinstance(order, int), 'order must be an integer.'

        mode = hologram.mode
        K = get_k_hologram_at_point(hologram=hologram, point=point)
        n0 = hologram.n0([playback.source.wavelength])[0]
        dn = hologram.dn
        thickness = hologram.thickness
        k_r = get_source_k_into_hologram_at_point(point=point, hologram=hologram, source=playback.source)
        k_d = get_k_diff_at_points(hologram, playback, [point], order)
        k_d = k_d[0]
        param = {'mode': mode,
                 'thickness': thickness,
                 'n0': n0,
                 'dn': dn,
                 'k_r': k_r,
                 'k_d': k_d,
                 'K': K,
                 'order': order,
                 'wavelength': playback.source.wavelength}
        return param

    @staticmethod
    def _rhs(x, y):
        """
        A function to compute a state [dAr/dz, dAd/dz].

        :param x: indendent variable of the problem, i.e. z. But it is not needed for uniform gratings.
        :param y: current state, i.e. [Ar, Ad]
        :return [dAr/dz, dAd/dz]: (2,) ndarray
        """
        matrix_h = np.reshape(matrix, newshape=(2, 2))
        dy = np.matmul(matrix_h, y)
        return dy

    @staticmethod
    def _boundary_for_reflection(ya, yb):
        return np.array([ya[0] - 1, yb[1]])

    @staticmethod
    def _boundary_for_transmission(ya, yb):
        return np.array([ya[0] - 1, ya[1]])


def loss_rms_k(k_targ, k_cand, weights=np.array([1.]), norm_fac=1e6):
    """
    Return a root-mean-square-error loss between the target K vectors and the candidate K vectors. The distance dK = Ktarg -
    Kcand is a Euclidean norm. Since dK is typically large, it will make optimization very difficult. Therefore, it is
    useful to normalize dK before squaring. Default normalization is 1e6.

    :param k_targ: a list of sympy.vector.Vector instances, representing a target K vector
    :param k_cand: a list of sympy.vector.Vector instances, representing a candidate K vector
    :param weights: a scalar number, a list of number defined a weight for each K point.
    :param norm_fac: Normalization factor applied to dK before squaring them. Default = 1e6.
    :return: loss value
    """
    assert all(isinstance(K, vec.Vector) for K in k_targ), 'k_targ must be a list of {} instance.'.format(vec.Vector)
    assert all(isinstance(K, vec.Vector) for K in k_cand), 'k_cand must be a list of {} instance.'.format(vec.Vector)
    assert isinstance(norm_fac, (int, float)), 'norm_fac must be a number.'
    assert len(k_targ) == len(k_cand), 'k_targ and k_cand lists must have the same length.'

    if isinstance(weights, (int, float)):
        weights = np.array([weights], dtype=np.float)
    isvalid, msg = validate_input_numeric(weights, shape=(None,))
    if not isvalid:
        raise Exception('weights is invalid. ' + msg)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float)
    if weights.shape[0]!=1 and weights.shape[0]!=len(k_targ):
        raise ValueError('weights must be a scalar number or a list/ndarray of number with a length of len(k_targ).')

    if isinstance(norm_fac, int):
        norm_fac = float(norm_fac)

    dks = np.ndarray(shape=(len(k_targ),), dtype=np.float)
    for i in range(len(k_targ)):
        k1 = k_targ[i]
        k2 = k_cand[i]
        dk = (k1-k2)
        dks[i] = float(dk.magnitude())/norm_fac

    return np.sqrt(np.mean(np.array(weights) * np.power(dks, 2)))


def loss_rms_de(de_targ, de_cand, weights=np.array([1.])):
    """
    Return a root-mean-square-error loss between target DE and candidate DE.

    :param de_targ: a list or (N,) ndarray of target DE values
    :param de_cand: a list or (N,) ndarray of candidate DE values
    :param weights: a scalar number, a list of number defined a weight for each DE point.
    :return: loss value
    """
    isvalid, msg = validate_input_numeric(de_targ, shape=(None,))
    if not isvalid:
        raise Exception('de_targ must be a list or (N,) ndarray of numbers.')
    isvalid, msg = validate_input_numeric(de_cand, shape=(None,))
    if not isvalid:
        raise Exception('de_cand must be a list or (N,) ndarray of numbers.')
    if weights is None:
        weights = np.array([1.])
    if isinstance(weights, (int, float)):
        weights = np.array([weights], dtype=np.float)
    isvalid, msg = validate_input_numeric(weights, shape=(None,))
    if not isvalid:
        raise Exception('weights is invalid. ' + msg)
    if isinstance(weights, list):
        weights = np.array(weights, dtype=np.float)
    if weights.shape[0] != 1 and weights.shape[0] != len(de_targ):
        raise ValueError('weights must be a scalar number or a list/ndarray of number with a length of len(de_targ).')

    if isinstance(de_targ, list):
        de_targ = np.array(de_targ)
    if isinstance(de_cand, list):
        de_cand = np.array(de_cand)

    return np.sqrt(np.mean(weights * np.power(de_targ-de_cand, 2)))


def get_k_diff_at_points(hologram, playback, points, order=1):
    """
    Return a list of diffracted wave vectors for specified points on a hologram.

    :param hologram: an instance of class or subclass of Holograms
    :param points: an instance of a list of instances of class sympy.vector.Point
    :param playback: an instance of a Playback class.
    :param order:  an non-zero integer specifying what order to diffract into.
    :return: [k_diff]: a list of diffracted k calculated by momentum matching (instances of sympy.vector.Vector)
    """
    assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.'\
        .format(holograms.Hologram)
    assert isinstance(order, int), 'order must be an integer.'
    assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(illumination.Playback)
    if not isinstance(points, list):  # help converting to list when entering only one point
        points = [points]
    assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.' \
        .format(vec.Point)

    k_diffs = []  # a list of k vector of diffracted light
    for p in points:
        k_diffs.append(get_k_diff_at_point(hologram, playback, p, order=order))
    return k_diffs


def get_k_diff_at_point(hologram, playback, point, order=1, **kwargs):
    """
    Returns a k_diff vector (instance of sympy.vector.Vector) at a given point on the hologram

    :param hologram: an instance of holograms.HologramBase subclasses.
    :param playback: an instance of illumination.Playback
    :param point: an instance of sympy.vector.Point specifying a point on a hologram
    :param order: an integer specifying a diffraction order
    :return k_diff: an instance of sympy.vector.Vector
    """
    assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.'.\
        format(holograms.Hologram)
    assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.\
        format(illumination.Playback)
    assert isinstance(point, vec.Point), 'point must be an instance of {}.'.format(vec.Point)
    assert isinstance(order, int), 'order must be a non-zero integer.'

    k1_inside = get_source_k_into_hologram_at_point(point=point, hologram=hologram, source=playback.source)

    K = get_k_hologram_at_point(hologram=hologram, point=point)

    k_diff = k1_inside - order * K

    return k_diff


def get_dephase_at_points(hologram, playback, points, order=1):
    """
    Return a list of dephase values for specified points on a hologram. Dephase is defined as (k^2-k_diff^2)/2/k where
    k is a magnitude of wave vector (i.e. 2*pi*n/lambda) and k_diff is the magnitude of the diffracted wave vector
    (i.e. k_diff = k_in - order*K).

    :param hologram: an instance of class or subclass of Holograms
    :param points: an instance of a list of instances of class sympy.vector.Point
    :param playback: an instance of a Playback class.
    :param order:  an non-zero integer specifying what order to diffract into.
    :return dephase: a list of dephase values
    """
    assert isinstance(hologram, holograms.Hologram), 'hologram must be an instance of {} or its subclasses.'\
        .format(holograms.Hologram)
    assert isinstance(order, int), 'order must be an integer.'
    assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(illumination.Playback)
    if not isinstance(points, list):  # help converting to list when entering only one point
        points = [points]
    assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.' \
        .format(vec.Point)
    dephases = []  # a list of (k^2-k_diff^2)/2/k where, here, k is the magnitude of the wave vector.
    for p in points:
        dephases.append(get_dephase_at_point(hologram, playback, p, order=order))
    return dephases


def get_dephase_at_point(hologram, playback, point, order=1):
    """
    Return a dephase value for a specified point on a hologram. Dephase is defined as (k^2-k_diff^2)/2/k where
    k is a magnitude of wave vector (i.e. 2*pi*n/lambda) and k_diff is the magnitude of the diffracted wave vector
    (i.e. k_diff = k_in - order*K).

    :param hologram: an instance of holograms.HologramBase subclasses.
    :param playback: an instance of illumination.Playback
    :param point: an instance of sympy.vector.Point specifying a point on a hologram
    :param order: an integer specifying a diffraction order
    :return dephase: a number
    """

    assert isinstance(hologram, holograms.HologramBase), 'hologram must be an instance of {} or its subclasses.'\
        .format(holograms.HologramBase)
    assert isinstance(point, vec.Point), 'point must be an instance of {}.'.format(vec.Point)
    assert isinstance(playback, illumination.Playback), 'playback must be an instance of {}.'.format(illumination.Playback)
    assert isinstance(order, int), 'order must be an integer.'

    k1 = get_source_k_into_hologram_at_point(point=point, hologram=hologram, source=playback.source)
    k_diff = get_k_diff_at_point(hologram=hologram, playback=playback, point=point, order=order)

    beta = float(k1.magnitude())
    dephase = (float(k_diff.magnitude())**2 - beta**2)/2/beta
    return dephase
  
  
def get_k_hologram_at_point(hologram, point):
    """
    Return a grating vector as defined by k1-k2 from two recording sources.

    :param hologram: an instance of holograms.HologramBase subclasses
    :param point: an instance of sympy.vector.Point
    :return k_hologram: an instance of sympy.vector.Vector
    """
    assert isinstance(hologram, holograms.HologramBase), 'hologram must be an instance of {} subclasses.'.format(holograms.HologramBase)
    assert isinstance(point, vec.Point), 'point must be an instance of {}'.format(vec.Point)

    if not hologram.is_ready():
        raise Exception('hologram is not fully configured.')

    k1_inside = get_source_k_into_hologram_at_point(point=point, hologram=hologram,
                                                    source=hologram.recording.source1)

    k2_inside = get_source_k_into_hologram_at_point(point=point, hologram=hologram,
                                                    source=hologram.recording.source2)

    return k1_inside-k2_inside


def get_k_hologram_at_points(hologram, points):
    """
    Return a list of grating vectors for specified points.

    :param hologram: an instance of holograms.HologramBase subclasses
    :param points: an instance of a list of instances of class sympy.vector.Point
    :return [k_holo]: a list of sympy.vector.Vector instances.
    """

    assert isinstance(hologram, holograms.HologramBase), 'hologram must be an instance of {} subclasses.'.format(
        holograms.HologramBase)
    assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of {} instances.'.format(vec.Point)

    K = []

    for i, p in enumerate(points):
        K.append( get_k_hologram_at_point(hologram, p) )

    return K


def get_source_k_into_hologram_at_point(point, hologram, source):
    """
    Return a k vector inside a hologram given a k vector from a source's side.

    :param point: an instance of sympy.vector.Point
    :param source: an instance of Source subclasses.
    :param hologram: an instance of hologram.HologramBase subclasses
    :return k_pp: an instance of sympy.vector.Vector
    """

    src = source

    k1 = src.get_k_at_points(points=[point])[0]
    n1 = src.material.n(src.wavelength)[0]

    norm_vec = hologram.get_norm_vec_at_points([point])[0]
    n_pp = hologram.material.n(src.wavelength)[0]

    k_pp = snell_law_k_space(k1, norm_vec=norm_vec, n_in=n1, n_out=n_pp, notify_tir=True)

    return k_pp


def get_k_out_off_hologram_at_point(k_pp, hologram, point, material_out):
    """
    Return a k vector that refracts out off a hologram. k_pp should be a k vector at the specified point.

    :param k_pp: a k vector inside a hologram, instance of sympy.vector.Vector
    :param hologram: an instance of holograms.HologramBase subclasses
    :param point: an instance of sympy.vector.Point
    :param material_out: an instance of materials.Material subclasses. Must be able to return a refractive index using ``.n()``.
    :return:
    """

    norm_vec = hologram.get_norm_vec_at_points(points=[point])[0]
    wavelength = hologram.recording.source1.wavelength
    n_pp = hologram.material.n(wavelength)[0]
    n_out = material_out.n(wavelength)[0]
    k_out = snell_law_k_space(k_pp, norm_vec, n_pp, n_out)

    return k_out
