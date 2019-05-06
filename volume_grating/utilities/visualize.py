import matplotlib.pyplot as plt
from utilities.geometry import vectors_to_ndarray, points_to_ndarray
import sympy.vector as vec
from systems import GCS
import numpy as np


def visualize_vector_transverse(vectors, points, axes=None, O=GCS):
    """
    Plot transvere components (v_xy = v-v_z) in a transverse plane (xy) for all vectors in ``vectors``, each at a point in ``points``.
    The number of vectors and points must equal.

    :param vectors: a list of sympy.vector.Vector instances
    :param points: a list of sympy.vector.Point instances
    :param axes: a matplot.pyplot.Axes instance, an axes to plot to.

    :return Q: an output from matplotlib.pyplot.quiver.
    :return axes: a matplotlib.pyplot.Axes instance to which the function plots.
    """
    assert all(isinstance(v, vec.Vector) for v in vectors), 'ks must be a list of {} instances.'.format(vec.Vector)
    assert all(isinstance(p, vec.Point) for p in points), 'ks must be a list of {} instances.'.format(vec.Point)
    assert len(vectors) == len(points), 'len(ks) must equal to len(points)'
    assert axes is None or isinstance(axes, plt.Axes), 'axes must be None or an instance of {}.'.format(plt.Axes)
    assert isinstance(O, vec.CoordSys3D), 'O must be an instance of {}'.format(vec.CoordSys3D)

    vecs_np = vectors_to_ndarray(vectors, O)
    points_np = points_to_ndarray(points, O)

    if axes is None:
        fig, axes = plt.subplots()
    plt.sca(axes)

    x = points_np[:, 0]
    y = points_np[:,1]
    u = vecs_np[:,0]
    v = vecs_np[:,1]

    axes.set_xlim(np.min(x)-1, np.max(x)+1)
    axes.set_ylim(np.min(y)-1, np.max(y)+1)

    Q = plt.quiver(x, y, u, v)

    plt.show()

    return Q, axes
