import numpy as np


def surface_poly_2nd(coeffs, xs):
    """
    Return z = f(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2. It is a second-order 2D polynomial function.

    :param xs: (N, 2) ndarray for coordiates (x,y).
    :param coeffs: a (6,) ndarray or a list of six numbers for coefficients.
    :return: (N,) ndarray of z vales.
    """
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs, dtype=np.float)
    z = coeffs[0]
    z += coeffs[1] * xs[:, 0]
    z += coeffs[2] * xs[:, 1]
    z += coeffs[3] * xs[:, 0] ** 2
    z += coeffs[4] * xs[:, 0] * xs[:, 1]
    z += coeffs[5] * xs[:, 1] ** 2
    return z


def surface_gradients(coeffs, xs):
    """
    Return a tuple of (dz/dx, dz/dy) when z = f(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2.

    :param xs: (N, 2) ndarray for coordiates (x,y).
    :param coeffs: a (6,) ndarray or a list of six numbers for coefficients.
    :return dz/dx: (N,) ndarray of dz/dx values.
    :return dz/dy: (N,) ndarray of dz/dy values.
    """
    grad1 = coeffs[1] + 2.0 * coeffs[3] * xs[:, 0] + coeffs[4] * xs[:, 1]
    grad2 = coeffs[2] + coeffs[4] * xs[:, 0] + 2 * coeffs[5] * xs[:, 0]
    return grad1, grad2


def surface_poly_2nd_loss_func(coeffs, xs, z_truth, regularization=0):
    """
    Return a mean-squared error (MSE) of predicted z values and ground-trough z values. Predicted z values are calculated using
    ``surface_poly_2nd`` function.

    :param coeffs: a (6,) ndarray of coordinate or a list of six numbers.
    :param xs: a (N, 2) ndarray of N (x,y) coordinates.
    :param z_truth: (N,) ndarray of ground-trugh z values.
    :param regularization: [default=0] a level of reqularization.
    :return: a MSE value.
    """
    z_pred = surface_poly_2nd(coeffs=coeffs, xs=xs)
    return np.mean((z_pred - z_truth) ** 2) + regularization * (np.sum(coeffs ** 2) - coeffs[0] ** 2) / (
                coeffs.size - 1)


def prepare_surface_poly_2nd_fit(x, y, z):
    """
    Prepare ranges in x (shape of (Nx,)) and y (shape of (Ny,)) and gridded ground-truth z (shape of (Nx, Ny)) for optimization
    2D polynomial fit.

    :param x: (Nx,) ndarray of point ranges in x axis
    :param y: (Ny,) ndarray of point ranges in y axis
    :param z: (Nx, Ny) ndarray of ground-truth z values

    :return xs: (Nx*Ny, 2) grid-meshed (x_norm, y_norm) points that are also normalized.
    :return z_norm: (Nx*Ny,) ground-truth values that are normalized
    :return norm_factors: a dictionary of normalization factors
    """
    x_mean = np.mean(x)
    x_range = np.max(x) - np.min(x)
    x_norm = (x - x_mean) / x_range

    y_mean = np.mean(y)
    y_range = np.max(y) - np.min(y)
    y_norm = (y - y_mean) / y_range

    z_reshaped = np.reshape(z, newshape=(z.size,))
    z_mean = np.mean(z_reshaped)
    z_range = np.max(z_reshaped) - np.min(z_reshaped)
    z_norm = (z - z_mean) / z_range
    z_norm = np.reshape(z_norm, newshape=(z_norm.size,))

    norm_factors = {'x mean': x_mean,
                    'x range': x_range,
                    'y mean': y_mean,
                    'y range': y_range,
                    'z mean': z_mean,
                    'z range': z_range}

    x_norm, y_norm = np.meshgrid(x_norm, y_norm)

    x_norm = np.reshape(x_norm, newshape=(x_norm.size, 1))
    y_norm = np.reshape(y_norm, newshape=(y_norm.size, 1))
    xs = np.concatenate((x_norm, y_norm), axis=-1)

    xs = to_xs(x=x_norm, y=y_norm)

    return xs, z_norm, norm_factors


def scale_surface(norm_factors, x=None, y=None, z=None):
    """
    Apply normalization factors to input ``x``, ``y``, ``z`` data points.

    :param norm_factors: a dictionary of normalization factors. Must have keys of '... mean' and '... range' where ... is in {x, y, z}.
    :param x: ndarray of x values. Can be None.
    :param y: ndarray of y values. Can be None.
    :param z: ndarray of z values. Can be None.
    :return x_norm: normalized x values. None if input ``x`` is None.
    :return y_norm: normalized x values. None if input ``y`` is None.
    :return z_norm: normalized x values. None if input ``z`` is None.
    """

    x_norm = None
    y_norm = None
    z_norm = None

    if x:
        x_mean = norm_factors['x mean']
        x_range = norm_factors['x range']
        x_norm = (x - x_mean) / x_range

    if y:
        y_mean = norm_factors['y mean']
        y_range = norm_factors['y range']
        y_norm = (y - y_mean) / y_range

    if z:
        z_mean = norm_factors['z mean']
        z_range = norm_factors['z range']
        z_norm = (z - z_mean) / z_range

    return x_norm, y_norm, z_norm


def reverse_scale_surface(norm_factors, x=None, y=None, z=None):
    """
    Reverse the normalization step for input ``x``, ``y``, ``z`` data points.

    :param norm_factors: a dictionary of normalization factors. Must have keys of '... mean' and '... range' where ... is in {x, y, z}.
    :param x: ndarray of input normalized x values
    :param y: ndarray of input normalized y values
    :param z: ndarray of input normalized z values
    :return x: unnormalized x values. None if ``x`` is None.
    :return y: unnormalized x values. None if ``y`` is None.
    :return z: unnormalized x values. None if ``z`` is None.
    """
    if x:
        x_mean = norm_factors['x mean']
        x_range = norm_factors['x range']
        x = x * x_range + x_mean

    if y:
        y_mean = norm_factors['y mean']
        y_range = norm_factors['y range']
        y = x * y_range + y_mean

    if z:
        z_mean = norm_factors['z mean']
        z_range = norm_factors['z range']
        z = z * z_range + z_mean

    return x, y, z


def to_xs(x, y):
    """
    Return a (N,2) of (x,y) pairs from two (N,) ndarrays of x and y values. ``x`` and ``y`` must have the same shape.

    :param x: ndarray of x values. Must have the same shape as ``y``.
    :param y: ndarray of y values. Must have the same shape as ``x``.
    :return: (N,2) of (x,y) pairs
    """

    if isinstance(x, (int, float)):
        x = np.array(x)
    if isinstance(y, (int, float)):
        y = np.array(y)
    assert x.shape == y.shape, 'x and y must have the same shape.'
    x = np.reshape(x, newshape=(x.size, 1))
    y = np.reshape(y, newshape=(y.size, 1))
    xs = np.concatenate((x, y), axis=-1)
    return xs
