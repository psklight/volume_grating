import numpy as np
from utilities.validation import validate_input_numeric


class Material(object):
    """
    Material is a base class to model optical material.
    """
    def __init__(self):
        pass

    def n(self, wavelengths):
        """
        **Abstract method**. Subclasses must re-define ``n`` method that returns a (real-part) of refractive indices at
        specified wavelengths.

        :param wavelengths: a list or ndarray of wavelengths in meter.
        :return n: an ndarray or real-part refractive indices.
        """
        raise Exception('Need implementation for {}.'.format(type(self)))


class NKconstant(Material):
    """
    NKconstant models a material with a constant complex index of N + i*K.

    :param N: the refractive index value
    :param K: the absorption/gain value
    """

    def __init__(self, N, K=0):

        super(NKconstant, self).__init__()
        self.N = N
        self.K = K

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        assert isinstance(value, (int, float)), 'n must be numeric.'
        self._N = float(value)

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        assert isinstance(value, (int,float)), 'k must be numeric.'
        self._K = float(value)

    def n(self, wavelength):
        """
        Return the real part, n, of the complex refractive index (n+i*k)

        :param wavelength: a list or ndarray of wavelengths in meter. Since the model is constant, recommend a single value input.
        :return: refractive index n (real part only at the moment)
        """
        if not isinstance(wavelength, list):  # help converting to list if entering one value
            wavelength = [wavelength]
        if validate_input_numeric(wavelength):
            wavelength = np.array(wavelength, dtype=np.float)
            return np.ones(shape=wavelength.shape)*self.N
        else:
            raise Exception('wavelength must be a list or ndarray of numeric.')

    def k(self, wavelength):
        """
        Return the imaginary part, k, of the complex refractive index (n+i*k)

        :param wavelength: a list or ndarray of wavelengths in meter. Since the model is constant, recommend a single value input.
        :return: refractive index n (real part only at the moment)
        """
        if not isinstance(wavelength, list):  # help converting to list if entering one value
            wavelength = [wavelength]
        if validate_input_numeric(wavelength):
            wavelength = np.array(wavelength, dtype=np.float)
            return np.ones(shape=wavelength)*self.K
        else:
            raise Exception('wavelength must be a list or ndarray of numeric.')

    def get_complex_index(self, wavelength):
        """
        Return the whole complex refractive index, i.e. n+i*k, which is constant

        :param wavelength: a list or ndarray of wavelengths in meter. Since the model is constant, recommend a single value input.
        :return:
        """
        if not isinstance(wavelength, list):  # help converting to list if entering one value
            wavelength = [wavelength]
        if validate_input_numeric(wavelength):
            wavelength = np.array(wavelength, dtype=np.float)
            return np.ones(shape=wavelength)*self.N + np.ones(shape=wavelength)*self.K * 1j
        else:
            raise Exception('wavelength must be a list or ndarray of numeric.')

    def __str__(self):
        return "n: {}, k: {}".format(self.N, self.K)


class Cauchy_2coeff(Material):
    """
    Cauchy_2coeff models a material with two Cauchy coefficients, B and C.
    See https://en.wikipedia.org/wiki/Cauchy%27s_equation for Cauchy model.
    Unit of C is micron**2.

    :param coeff: (2,) ndarry of numbers. B = coeff[0] and C = coeff[1].
    """

    def __init__(self, coeff):
        """Constructor for Cauchy_2coeff"""
        super(Cauchy_2coeff, self).__init__()
        self.coeff = coeff
        
    @property
    def coeff(self):
        return self._coeff
    
    @coeff.setter
    def coeff(self, value):
        if validate_input_numeric(value, shape=(2,)):
            self._coeff = value
        else:
            raise Exception('coeff must be a list of 2 numeric values, such as [1.0, 2.0].')

    def n(self, wavelength):
        """
        Returns a (real-part) of refractive indices at specified wavelengths.

        :param wavelength: a list or ndarray of wavelengths in meter.
        :return: refractive index n
        """
        if not isinstance(wavelength, list):  # help converting to list if entering one value
            wavelength = [wavelength]
        if validate_input_numeric(wavelength):
            wavelength = np.array(wavelength, dtype=np.float)/1e-6
            return self.coeff[0] + self.coeff[1]/wavelength**2
        else:
            raise Exception('wavelength must be a list or ndarray of numeric.')

    def __str__(self):
        return "Cauchy, B: {}, C: {}".format(self.coeff[0], self.coeff[1])

    @property
    def k(self):
        return 0.0


#: Dictionary of Cauchy coefficients for some materials.
cauchy_dict = {'bk7'    : [1.5046, 4.2e-3],
               'lens'   : [1.5321, 7.43e-3],
               'pc'     : [1.5551, 9.72e-3],
               'pp_original'    : [1.4824, 6.93e-3],
               'pp_processed'   : [1.4900, 6.96e-3]}

#: NKconstant instance of air or vacuum.
air = NKconstant(N=1, K=0)
