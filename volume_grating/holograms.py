from systems import GCS
from materials import Cauchy_2coeff
import sympy.vector as vec
import illumination
from utilities.validation import validate_input_numeric
import numpy as np


class HologramBase(object):
    """
    HologramBase is a base class of other hologram class definitions. It needs some basic physical parameters to
    initialize. However, without being fully initialized (e.g. recording=None), the hologram cannot be asked for its
    responses.

    :param thickness: a thickness in meter; must be numeric.
    :param material: an instance of Material class or its subclasses. It must be able to be called for refractive index. Default to None.
    :param dn: the index modulation of the hologram; must be numeric.
    :param recording: an instance of illumination.Record class. Default to None.
    """

    def __init__(self, thickness=10e-6, material=None, dn=0.045, recording=None):
        self.thickness = thickness
        self.material = material
        self.dn = dn
        self.origin = GCS.origin
        self.recording = recording

    @property
    def dn(self):
        return self._dn

    @dn.setter
    def dn(self, value):
        value = float(value)
        assert isinstance(value, float), 'dn must be numeric.'
        self._dn = value

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        assert isinstance(value, Cauchy_2coeff) or value is None, 'New value must be a type of {}.' \
            .format(type(Cauchy_2coeff([0, 0])))
        self._material = value

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        value = float(value)
        assert isinstance(value, float), 'Thickness must be numeric.'
        self._thickness = value

    @property
    def recording(self):
        return self._recording

    @recording.setter
    def recording(self, value):
        assert isinstance(value, illumination.Record) or value is None, 'Recording must be an instance of {} class' \
            .format(illumination.Record)
        self._recording = value

    @property
    def mode(self):
        k1 = self.recording.source1.get_k_at_points([GCS.origin])[0]
        k2 = self.recording.source2.get_k_at_points([GCS.origin])[0]
        kz1 = float(k1.dot(GCS.k))
        kz2 = float(k2.dot(GCS.k))
        if kz1 * kz2 >= 0:
            mode = "transmission"
        else:
            mode = "reflection"
        return mode

    def __str__(self):
        return str(self.__dict__)

    def is_ready(self):
        """
        Return True or False whether the hologram instance is fully initialized and ready to calculate for its responses.
        Being ready means all of the attributes are not None.

        :return: True or False
        """
        return all(self.__dict__[x] is not None for x in list(self.__dict__.keys()))

    def n0(self, wavelengths):
        """
        Return refractive index of the material that makes the hologram. This will call the method Material.n(...), which
        returns a (N,) ndarray.

        :param wavelengths: a numeric value, a numeric list, or a numeric (N,) ndarray.

        :return: an (N,) ndarray.
        """
        isvalid, msg = validate_input_numeric(wavelengths, shape=(None,))
        if isvalid:
            return self.material.n(wavelengths)
        else:
            raise Exception("At wavelengths, "+msg)

    def get_points_from_xy_arrays(self, arrays, O=GCS):
        """
        Return an ndarray-type list of Point instances. For this class, all z values will be 0 size the hologram is flat.

        :param arrays: a (N, 2) ndarray containing (x, y) coordinates. N is the number of (x, y) pairs.
        :param O: an instance of sympy.vector.CoordSys3D. Default to the library's GCS (Global Coordinate System).
        :return: a (N,) ndarray where each item is an instance of sympy.vector.Point corresponding to (x, y, z).
        """
        assert isinstance(O, vec.CoordSys3D) or O is None, 'O must be an instance of {}.'.format(vec.CoordSys3D)
        isvalid, msg = validate_input_numeric(arrays, shape=(None, 2))
        if not isvalid:
            raise Exception(msg)
        if O is None:
            O = GCS
        arrays = arrays.astype(np.float)
        output = np.ndarray(shape=(arrays.shape[0]), dtype=vec.Point)
        for i in range(arrays.shape[0]):
            x, y = arrays[i]
            output[i] = O.origin.locate_new("({:.2}, {:.2}, {:.2})".format(x, y, 0.0), x*O.i+y*O.j+0.0*O.k)
        return output

    # def get_k_at_points(self, points):
    #     """
    #     **Abstract method**. Subclasses must override this method.
    #
    #     :param points: a list of instances of class sympy.vector.Point
    #     :return:
    #     """
    #     raise Exception('For {}, this method needs override implementation'.format(type(self)))
    #
    # def get_all_k_at_points(self, points):
    #     """
    #     **Abstract method**. Subclasses must override this method.
    #
    #     :param points: a list of instances of class sympy.vector.Point
    #     :return:
    #     """
    #     raise Exception('For {}, this method needs override implementation'.format(type(self)))

    def get_norm_vec_at_points(self, points):
        """
        **Abstract method**. Subclasses must override this method.

        :param points: a list of instances of class sympy.vector.Point
        :return:
        """
        raise Exception('For {}, this method needs override implementation'.format(type(self)))


class Hologram(HologramBase):

    def __init__(self, thickness=10e-6, material=None, dn=0.045, recording=None):
        super(Hologram, self).__init__(thickness=thickness, material=material, dn=dn, recording=recording)

    # def get_k_at_points(self, points):
    #     """
    #     Return a list of grating vectors for specified points on the hologram.
    #
    #     :param points: an instance of a list of instances of class sympy.vector.Point
    #     """
    #     if not isinstance(points, list):  # help converting to list when entering only one point
    #         points = [points]
    #     assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.' \
    #         .format(vec.Point)
    #     if self.is_ready():
    #         ks = []
    #         src1 = self.recording.source1
    #         src2 = self.recording.source2
    #         for p in points:
    #             k1 = src1.get_k_at_points([p])  # k1 is a list
    #             k2 = src2.get_k_at_points([p])  # k2 is a list
    #             ks.append(k1[0]-k2[0])
    #         return ks
    #     else:
    #         raise Exception('The hologram parameters are not fully configured.'
    #                         ' Use print() to see what attribute is still None.')
    #
    # def get_all_k_at_points(self, points):
    #     """
    #     Return a (N,3) 2D-ndarray of k1, k2, and K for each specified point. N is len(points). In the second dimension, the
    #     first is k1, second is k2, and third is K, where K = k1-k2. k1 is from source 1 and k2 is from source 2.
    #
    #     :param points: an instance of a list of instances of class sympy.vector.Point
    #     """
    #     # """
    #     #
    #     # :param points: an instance of a list of instances of class sympy.vector.Point
    #     # :return: ks: a (N,3) ndarray of k1, k2, and K for each point. N is len(points). In the second dimension, the
    #     # first is k1, second is k2, and third is K, where K = k1-k2. k1 is from source 1 and k2 is from source 2.
    #     # """
    #     if not isinstance(points, list):  # help converting to list when entering only one point
    #         points = [points]
    #     assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.' \
    #         .format(vec.Point)
    #     if self.is_ready():
    #         ks = np.ndarray(shape=(len(points), 3), dtype=np.object)
    #         for i, p in enumerate(points):
    #             local_list = np.ndarray(shape=(3,), dtype=np.object)
    #             local_list[0] = self.recording.source1.get_k_at_points([p])[0]
    #             local_list[1] = self.recording.source2.get_k_at_points([p])[0]
    #             local_list[2] = local_list[0]-local_list[1]
    #             ks[i] = local_list
    #         return ks
    #     else:
    #         raise Exception('The hologram parameters are not fully configured. Use print() to see what attribute is still None.')

    def get_norm_vec_at_points(self, points):
        """
        Return a list of surface normal vector at points on a hologram. For this hologram it's always flat. Hence, normal vectors
        are along z axis.

        :param points: a list of instances of class sympy.vector.Point
        :return:
        """
        if not isinstance(points, list):  # help converting to list when entering only one point
            points = [points]
        assert all(isinstance(p, vec.Point) for p in points), 'points must be a list of instances of {} class.' \
            .format(vec.Point)

        norm_vecs = np.ndarray(shape=(len(points),), dtype=np.object)

        for i, p in enumerate(points):
            norm_vecs[i] = GCS.k

        return list(norm_vecs)
