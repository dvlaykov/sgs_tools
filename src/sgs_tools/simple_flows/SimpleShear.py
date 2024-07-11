from dataclasses import dataclass
from typing import Iterable

import numpy as np
from xarray import DataArray

from ..geometry.grid import CoordScalar, Grid


@dataclass(frozen=True)
class SimpleShear:
    """prescribe velocity for a simple shear flow on a general grid

    :ivar grid: grid which provices coordinates
    :ivar dimensions: labels of shearing directions
    :ivar velcomp: component of velocity vector that is sheared (all others are set to 0)
    :ivar amplitude: value of gradient
    """

    grid: Grid
    dimensions: Iterable[str]
    velcomp: int
    amplitudes: np.typing.ArrayLike

    def velocity(self, shape: list[int]) -> DataArray:
        """produce a velocity field with a given shape

        :param shape: shape of velocity field
        """
        coord_mesh = self.grid.mesh(shape)
        v_shear = self.amplitude
        for d in self.dimensions:
            scalar_gdt = ScalarGradient(self.grid, d, 1.0, 1.0)
            v_shear *= coord_mesh[d]
        v_zero = np.zeros_like(v_shear)
        v = np.roll(np.stack([v_shear, v_zero, v_zero]), self.velcomp, axis=0)

        return DataArray(v, dims=["c1"] + list(coord_mesh.keys()), coords=mesh)


@dataclass(frozen=True)
class ScalarGradient:
    """prescribe a scalar field with a constant gradient on a cartesian mesh

    :ivar grid: grid which provices coordinates
    :ivar dimension: label of direcion of coordinate
    :ivar gdt: value of gradient
    :ivar offset: reference value at coordinate origin
    """

    grid: Grid
    dimension: str
    gdt: float
    offset: float

    def field(self, shape: list[int]) -> DataArray:
        """produce a scalar field with a given shape

        :param shape: shape of scalar field
        """
        coord = CoordScalar(self.grid, self.dimension, self.gdt)
        coord_array = coord.scalar(shape)
        coord_array += self.offset - coord_array.isel({self.dimension: 0})
        return coord_array
