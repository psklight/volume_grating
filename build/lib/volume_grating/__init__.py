import sympy.vector as vec
from sympy.vector import CoordSys3D
GCS: CoordSys3D = vec.CoordSys3D('GCS')

from . import holograms, engines, illumination, materials, routines, solvers, sources, systems

__all__ = ["holograms",
           "engines",
           "illumination",
           "materials",
           "routines",
           "solvers",
           "sources",
           "systems",
           "GCS"]

