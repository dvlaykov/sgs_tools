from typing import Callable, Collection, Iterable

import xarray as xr

from ..geometry.tensor_algebra import symmetrise, traceless
from ..geometry.vector_calculus import grad_vector


def strain_from_vel(
    vel: xr.Dataset | Iterable[xr.DataArray],
    space_dims: Iterable[str],
    vec_dim: str,
    new_dim: str = "c2",
    incompressible: bool = True,
    grad_operator: Callable = grad_vector,
) -> xr.DataArray:
    """compute rate of strain from velocity
    if cache is given will attempt to store intermediate calcualtion
    of gradvel in it
    """
    gradvel = grad_operator(vel, space_dims, new_dim)
    sij = symmetrise(gradvel, [vec_dim, new_dim])
    if incompressible:
        sij = traceless(sij, [vec_dim, new_dim])
    return sij


def vertical_heat_flux(
    vert_vel: xr.DataArray, pot_temperature: xr.DataArray, hor_axes: Collection[str]
) -> xr.DataArray:
    """vertical heat flux w' theta'"""
    w_prime = vert_vel - vert_vel.mean(dim=hor_axes)
    theta_prime = pot_temperature - pot_temperature.mean(hor_axes)
    ans = w_prime * theta_prime
    ans.name = "vertical_heat_flux"
    ans["long_name"] = r"$w' \theta'$ "
    return ans
