import numpy as np
import sympy.vector as vec
from .validation import validate_input_numeric


def ndarray_to_vector(array, O):
    """
    Return a sympy.vector.Vector instance from a (3,) ndarray

    :param array: a (3,) array of numeric
    :param O: Coordinate system from sympy, Cartesian
    :return: a sympy.vector.Vector instance from a (3,) ndarray
    """
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'
    isvalid, msg = validate_input_numeric(array, shape=(3,))
    if isvalid:
        v = O.i*array[0] + O.j*array[1] + O.k*array[2]
        return v
    else:
        raise Exception(msg)


def vector_to_ndarray(v, O):
    """
    Return a (3,) ndarray of vector components from an input sympy.vector.Vector instance.

    :param v: an instance of sympy.vector.Vector
    :param O: Coordinate system from sympy, Cartesian
    :return: Return a (3,) ndarray of vector components
    """
    assert isinstance(v, vec.Vector), 'v must be an instance of sympy.vector.Vector.'
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'
    m = v.to_matrix(O)
    array = np.ndarray(shape=(3,), dtype=np.float)
    for i in range(array.size):
        array[i] = float(m[i])
    return array


def vectors_to_ndarray(vs, O):
    """
    Return (N,3) ndarray of vector components from a list of sympy.vector.Vector instances.

    :param vs: a list of sympy.vector.Vector instances. ``N=len(vs)``.
    :param O: Coordinate system from sympy, Cartesian
    :return: (N,3) ndarray of vector components
    """
    assert all(isinstance(v, vec.Vector) for v in vs), 'vs is a list of sympy.vector.Vector instances.'
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'
    array = np.ndarray(shape=(len(vs), 3), dtype=np.float)
    for i, v in enumerate(vs):
        array[i] = vector_to_ndarray(v, O)
    return array


def ndarray_to_point(array, O, name=None):
    """
    Return an instance of sympy.vector.Point for a (3,) coordinate.

    :param array: a (3,) ndarray containing x,y,z coordinate value
    :param name: [default=None] a name of the point, in string. If None, the name will be (x,y,z).
    :param O: Coordinate system in Cartesian
    :return: an instance of sympy.vector.Point
    """
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'
    assert isinstance(name, str) or name is None, 'name must be a string instance.'
    isvalid, msg = validate_input_numeric(array, shape=(3,))
    if isvalid:

        if name is None:
            name = "({},{},{})".format(array[0], array[1], array[2])

        v = O.origin.locate_new(name, O.i*array[0]+O.j*array[1]+O.k*array[2])
        return v
    else:
        raise Exception("At array. "+msg)


def point_to_ndarray(point, O):
    """
    Return a (3,) ndarry for x, y, z coordinate from a given sympy.vector.Point

    :param point: an instance of sympy.vector.Point
    :param O: an instance of sympy.vector.CoordSys3D
    :return: a (3,) ndarry for x, y, z coordinate
    """
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'
    assert isinstance(point, vec.Point), 'point must be an instance of sympy.vector.Point.'

    array = np.ndarray(shape=(3,), dtype=np.float)
    m = point.express_coordinates(O)  # m is now a tuple.
    for i in range(len(m)):
        array[i] = m[i]
    return array


def points_to_ndarray(points, O):
    """
    Return (N,3) ndarray of (x,y,z) coordinate from a list of sympy.vector.Point instances.

    :param points: a list of sympy.vector.Point instances. ``N=len(points)``.
    :param O: Coordinate system from sympy, Cartesian
    :return: (N,3) ndarray of (x,y,z) coordinate
    """
    assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of {} instance.'.format(vec.Point)
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of sympy.vector.CoordSys3D. Only works with Cartesian.'

    array = np.ndarray(shape=(len(points), 3), dtype=np.float)

    for i, p in enumerate(points):
        array[i] = point_to_ndarray(p, O)

    return array


def cartersian_to_spherical(arrays, decimals=None):
    """
    Return (N,3)-ndarray spherical coordinates of given (N,3)-ndarray Cartesian coordinates. For each i in N of spherical coordinates,
    1 value = the length or radius from origin
    2 value = the angle with respect to the x axis of the projection on the to xy plane
    3 value = the angle of projection on to the z axis

    :param array: a (N, 3) ndarray of (x,y,z) Cartesian coordinates. N is the number of points.
    :param decimals: [optional, default to None]. This resolves rounding error that might cause the argument to ``arccos`` to be larger than one.
    :return: a (N,3) ndarray of spherical coordinates

    """
    expanded = False
    if arrays.shape == (3,):
        arrays = np.expand_dims(arrays, axis=0)
        expanded = True

    sph = np.ndarray(shape=arrays.shape, dtype=np.float)

    for i in range(arrays.shape[0]):
        v = arrays[i]
        r = np.sqrt(np.sum(v**2))
        tilt = np.rad2deg(np.arccos(v[2]/r))
        theta = np.rad2deg(np.arctan2(v[1], v[0]))
        sph[i, 0] = r
        sph[i, 1] = theta
        sph[i, 2] = tilt

    if expanded:
        sph = sph.squeeze()
    return sph


def spherical_to_cartesian(arrays, decimals=None):
    """
    Return (N,3)-ndarray Cartesian coordinates from give (N,3)-ndarray spherical coordinates. For each i in N of spherical coordinates,
    1 value = the length or radius from origin
    2 value = the angle with respect to the x axis of the projection on the to xy plane
    3 value = the angle of projection on to the z axis

    :param array: a (N, 3) ndarray of (r, theta, tilt) coordinates. N is the number of points.
    :return: a (N,3) ndarray of Cartesian coordinates
    """
    expanded = False
    if arrays.shape == (3,):
        arrays = np.expand_dims(arrays, axis=0)
        expanded = True

    cart = np.ndarray(shape=arrays.shape, dtype=np.float)
    for i in range(arrays.shape[0]):
        v = arrays[i]
        z = v[0]*np.cos(np.deg2rad(v[2]))
        x = v[0]*np.sin(np.deg2rad(v[2]))*np.cos(np.deg2rad(v[1]))
        y = v[0]*np.sin(np.deg2rad(v[2]))*np.sin(np.deg2rad(v[1]))
        # cart[i, 0] = x
        # cart[i, 1] = y
        # cart[in, 2] = z
        if decimals is None:
            cart[i] = np.array([x, y, z], dtype=np.float)
        else:
            cart[i] = np.round(np.array([x,y,z], dtype=np.float), decimals=decimals)

    if expanded:
        cart = cart.squeeze()
    return cart


