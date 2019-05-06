import numpy as np
import sympy.vector as vec
import warnings


def snell_law_k_space(k_in, norm_vec, n_in, n_out, notify_tir=True):
    """
    Return a wave vector k_out due to refraction.

    :param k_in: a sympy.vector.Vector for incoming k vector (with magnitude 2*pi*n/lambda)
    :param norm_vec: a normalized normal vector pointing into an incoming medium
    :param n_in: (real-part) of refractive index of the incoming medium
    :param n_out: (real-part) of a refractive index of the outgoing medium
    :param notify_tir: [default=True] True to catch TIR by printing out and return None instead.
    :return k_out: an instance of sympy.vector.Vector
    """
    assert isinstance(k_in, vec.Vector), 'k_in must be an instance of {}.'.format(k_in)
    assert isinstance(norm_vec, vec.Vector), 'norm_vec must be an instance of {}.'.format(vec.Vector)
    assert isinstance(n_in, (int, float)), 'n_in must be a number.'
    assert isinstance(n_out, (int, float)), 'n_out must be a number.'
    assert isinstance(notify_tir, bool), 'notify_tir must be either True or False.'

    n_in = float(n_in)
    n_out = float(n_out)

    norm_vec = norm_vec.normalize()
    t1_vec = k_in.cross(norm_vec).normalize()
    t2_vec = norm_vec.cross(t1_vec).normalize()

    k0 = k_in.magnitude()/n_in  # vacuum wave number, in 1e6 unit

    # tangential component, i.e. along t2_vec
    kt_in = k_in.dot(t2_vec)
    kt_out = kt_in  # boundary condition

    # normal component
    sign = np.sign(float(k_in.dot(norm_vec)))
    kn_out = sign * np.sqrt(complex((k0*n_out)**2 - kt_out**2))

    is_tir = np.iscomplex(kn_out)

    # compose k_out
    k_out = kt_out * t2_vec + kn_out * norm_vec

    if notify_tir and is_tir:
        k_out = None
        warnings.warn('TIR encountered.')


    return k_out
