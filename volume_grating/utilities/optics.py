import numpy as np
import sympy.vector as vec
import warnings
import matplotlib.pyplot as plt
import scipy.signal as signal


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


def get_bandwidth(y, x=None, mode='fwhm', width_points=0, plot=False):
    """
    Calculate a width of a largest peak (based on prominance (see scipy.signal.peak_widths)).

    :param y: a signal with peaks in a 1D-ndarray.
    :param x: a quantity where a signal rests upon, representing a peak width space. If None, the width is calculated based in data point space.
    :param mode: Chooses the relative height at which the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at least 0. See notes for further explanation. Also accepts "fwhm" for a value of 0.5 and "full" for a value of 1.
    :param width_points: a threshold in data-point space to detect peak. Peaks with narrower widths in data-point space will be ignored.
    :param plot: True or False to visually plot
    :return w: a peak width.
    """
    assert mode in ['fwhm', 'full'] or isinstance(mode, (float, int)), \
        KeyError('mode must be either {}, {}, a float from 0 to 1.'.format("fwhm", "full"))

    if mode == "fwhm":
        mode = 0.5
    if mode == "full":
        mode = 1.0
    rel_height = float(mode)
    if not 0.0 <= rel_height <= 1.0:
        KeyError('mode must be either {}, {}, a float from 0 to 1.'.format("fwhm", "full"))

    assert isinstance(width_points, int) and width_points >= 0, KeyError(
        'width_points must be zero or positive integer.')

    peaks, props = signal.find_peaks(y, width=width_points, rel_height=rel_height)

    max_peak_id = int(np.argmax(props['prominences']))
    width_result = signal.peak_widths(y, [peaks[max_peak_id]], rel_height=rel_height)

    if x is not None:
        x_right = ips_to_x(x, width_result[3])
        x_left = ips_to_x(x, width_result[2])
        w = x_right - x_left
    else:
        w = width_result[3] - width_result[2]

    if plot:
        if x is None:
            plt.plot(y)
            plt.plot(peaks, y[peaks], 'x')
            plt.hlines(*width_result[1:], color="C3")
        else:
            plt.plot(x, y)
            plt.plot(x[peaks], y[peaks], 'x')
            plt.hlines(width_result[1], x_left, x_right, color="C3")

        plt.title('width = {:.4}'.format(w[0]))
        plt.show()

    return w


def ips_to_x(x, ips):
    """
    A map from interpolated data point to a real quantity space. The map is based on a linear interpolation.

    :param x: an ndarray representing a quantity to interpolate for bandwidth.
    :param ips: a interpolated position from scipy.signal.find_peaks
    """
    id = int(ips)
    delta = ips - id
    range = x[id + 1] - x[id]
    x = x[id] + delta * range

    return x