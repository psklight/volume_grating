import numpy as np
import sympy.vector as vec
from utilities.validation import validate_input_numeric
from utilities.geometry import ndarray_to_vector
import materials
from systems import GCS


class Source(object):
    """
    Source is a base class for all optical source classes.

    :param material: an instance of Material class or its subclasses, specifying which material the source is embedded.
    :param wavelength: a numeric value of wavelength in meter. Default = 660 nm.
    """

    def __init__(self, material, wavelength=660e-9):
        self.material = material
        self.wavelength = wavelength

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        assert isinstance(value, (int, float)), 'wavelength must be numeric.'
        self._wavelength = value

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        assert isinstance(value, materials.Material), 'material must be an instance of {} class or its subclasses.'.\
            format(materials.Material)
        self._material = value

    def __str__(self):
        return str(self.__dict__)

    def get_k_at_points(self, points):
        """
        **Abstract method**. Subclasses must override this method. Returns a list of wave vectors for a set of
        specified points.

        :param points: a list of sympy.vector.Point instances.
        :return: a list of sympy.vector.Vector instances corresponding to the wave vectors at those points.
        """
        raise Exception('For {}, this method needs override implementation.'.format(type(self)))


class Planewave(Source):
    """
    Planewave class defines a planewave source, whose wave vector is the same every where.

    :param material: an instance of Material or its subclasses, specifying which material the source is embedded.
    :param direction: an (3,) ndarray specifying x, y, z components of direction vector. It does not need to be normalized.
    :param wavelength: a numerical value of wavelength in meter. Default = 660 nm.
    """

    def __init__(self, material, direction, wavelength=660e-9):
        super(Planewave, self).__init__(material=material, wavelength=wavelength)
        self.direction = direction

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        assert isinstance(value, (int, float)), 'wavelength must be numeric.'
        self._wavelength = float(value)
        
    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, value):
        if validate_input_numeric(value, shape=(3,)):
            value = np.array(value)
            self._direction = value
            self._direction_ = ndarray_to_vector(value, GCS).normalize()
        else:
            raise Exception('direction must be a list of 3 numeric or a (3,) ndarray of numeric.')
        
    @property
    def direction_(self):
        return self._direction_
    
    @direction_.setter
    def direction_(self, value):
        self._direction_ = value

    def get_k_at_points(self, points):
        """
        Returns a list of wave vectors for a set of specified points.

        :param points: a list of sympy.vector.Point instances.
        :return: a list of sympy.vector.Vector instances.
        """
        return [self._direction_*2*np.pi/self.wavelength*self.material.n(self.wavelength)]*len(points)


class PointSource(Source):
    """
    PointSource class defines a source that light either diverges or focuses to a point.

    :param point: a (3,) ndarray or a list of 3 numeric values.
    :param material: an instance of Material or its subclasses, specifying which material the source is embedded.
    :param direction: a string, either "diverge or "focus".
    :param wavelength: a numerical value of wavelength in meter. Default = 660 nm.
    """

    def __init__(self, point, material, direction='diverge', wavelength=660e-9):

        super(PointSource, self).__init__(material=material, wavelength=wavelength)
        self.point = point
        self.wavelength = wavelength
        self.direction = direction

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        assert value in ('diverge', 'focus'), 'direction must be either ""diverge"" or ""focus"".'
        self._direction = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        value = float(value)
        assert isinstance(value, float), 'wavelength must be numeric.'
        self._wavelength = value

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value):
        isvalid, _ = validate_input_numeric(value, shape=(3,))
        if not isvalid:
            raise Exception('Components must be a list of 3 numeric or a (3,) ndarray of numeric.')
        value = np.array(value).astype(np.float)
        self._point = value
        # update Point
        P = GCS.origin.locate_new('source origin', value[0]*GCS.i + value[1]*GCS.j + value[2]*GCS.k)
        self.point_ = P

    @property
    def point_(self):
        return self._point_

    @point_.setter
    def point_(self, value):
        self._point_ = value

    def __str__(self):
        return str(self.__dict__)

    def get_k_at_points(self, points):
        """
        Returns a list of wave vectors for a set of specified points.

        :param points: a list of sympy.vector.Point instances.
        :return: a list of sympy.vector.Vector instances.
        """
        if not isinstance(points, list):  # help converting to list when entering only one point
            points = [points]
        assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.'\
            .format(vec.Point)
        k = []
        dir = 1.0 if self.direction=="diverge" else -1.0
        for p in points:
            k.append(dir*p.position_wrt(self.point_).normalize()*2*np.pi/self.wavelength*self.material.n(self.wavelength))
        return k
