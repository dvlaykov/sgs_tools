from dataclasses import dataclass
from typing import Hashable

import xarray as xr  # only used for type hints

from ..geometry.tensor_algebra import Frobenius_norm
from .filter import Filter
from .sgs_model import DynamicHeatModel, DynamicVelocityModel, SGSModel


# check that arr is uniform along `filter_dims` with spacing of `dx`
def _assert_coord_dx(filter_dims: list[Hashable], arr: xr.DataArray, dx: float) -> None:
    for c in filter_dims:
        assert (arr[c].diff(dim=c) == dx).all()


@dataclass(frozen=True)
class SmagorinskyVelocityModel(SGSModel):
    """Smagorinsky model for the velocity equation

    :ivar vel: grid-scale velocity
    :ivar strain: grid-scale rate-of-strain
    :ivar cs: Smagorinsky coefficient
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    vel: xr.DataArray
    strain: xr.DataArray
    cs: float
    dx: float
    tensor_dims: tuple[str, str] = ("c1", "c2")

    def _snorm(self, filter: Filter) -> xr.DataArray:
        """compute the rate of strain norm at a given scale"""
        sij = filter.filter(self.strain)
        s = Frobenius_norm(sij, self.tensor_dims)
        s.name = "S_norm"
        s.attrs["long_name"] = "|<S>|"
        return s

    def sgs_tensor(self, filter: Filter) -> xr.DataArray:
        """compute model for SGS tensor
            :math:`$\\tau = (c_s \Delta) ^2 |\overline{Sij}| \overline{Sij}$`
            for a given `filter` (which can be trivial, i.e. ``IdentityFilter``)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.vel, self.strain]:
            _assert_coord_dx(filter.filter_dims, arr, self.dx)

        snorm = self._snorm(filter)
        sij = filter.filter(self.strain)
        return (self.cs * self.dx) ** 2 * snorm * sij


@dataclass(frozen=True)
class SmagorinskyHeatModel(SGSModel):
    """Smagorinsky model for the Heat equation

    :ivar vel: grid-scale velocity
    :ivar grad_theta: grid-scale (potential) temperature gradient
    :ivar strain: grid-scale rate-of-strain
    :ivar ctheta: Smagorinsky coefficient for the heat equation
    :ivar dx: constant resolution with respect to dimension to-be-filtered
    :ivar tensor_dims: labels of dimensions indexing tensor components
    """

    vel: xr.DataArray
    grad_theta: xr.DataArray
    strain: xr.DataArray
    ctheta: float
    dx: float
    tensor_dims: tuple[str, str] = ("c1", "c2")

    def _snorm(self, filter: Filter) -> xr.DataArray:
        """compute the rate-of-strain norm at a given scale"""
        sij = filter.filter(self.strain)
        s = Frobenius_norm(sij, self.tensor_dims)
        s.name = "S_norm"
        s.attrs["long_name"] = "|<S>|"
        return s

    def sgs_tensor(self, filter):
        """compute model for SGS tensor
            :math:`$\\tau =  c_\\theta \\Delta^2 |\overline{Sij}| \overline{\\nabla \\theta} $`
            for a given filter (which can be trivial, i.e. IdentityFilter)

        :param filter: Filter used to separate "large" and "small" scales
        """

        # only makes sense for uniform coordinates in the filtering directions
        # with spacing of self.dx
        for arr in [self.vel, self.grad_theta, self.strain]:
            _assert_coord_dx(filter.filter_dims, arr, self.dx)

        snorm = self._snorm(filter)
        grad_theta = filter.filter(self.grad_theta)
        return self.ctheta * self.dx**2 * snorm * grad_theta


def DynamicSmagorinskyVelocityModel(
    smag_vel: SmagorinskyVelocityModel,
) -> DynamicVelocityModel:
    return DynamicVelocityModel(smag_vel, smag_vel.vel)


def DynamicSmagorinskyHeatModel(
    smag_theta: SmagorinskyHeatModel, theta: xr.DataArray
) -> DynamicHeatModel:
    return DynamicHeatModel(smag_theta, smag_theta.vel, theta)
