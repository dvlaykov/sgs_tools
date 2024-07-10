from typing import Callable, Iterable, Union

import numpy as np
import xarray as xr

from .tensor_algebra import symmetrise, traceless
from .vector_calculus import grad_vector


def strain_from_vel(
    vel: Union[xr.Dataset, Iterable[xr.DataArray]],
    space_dims: Iterable[str],
    vec_dim: str,
    new_dim: str = "c2",
    incompressible: bool = True,
    grad_operator: Callable = grad_vector,
    cache: Union[None, xr.Dataset] = None,
) -> xr.DataArray:
    """compute rate of strain from velocity
    if cache is given will attempt to store intermediate calcualtion
    of gradvel in it
    """
    gradvel = grad_operator(vel, space_dims, new_dim)
    sij = symmetrise(gradvel, [vec_dim, new_dim])
    if incompressible:
        sij = traceless(sij, [vec_dim, new_dim])
    if cache is not None:
        cache["gradvel"] = gradvel
    return sij


def vertical_heat_flux(
    vert_vel: xr.DataArray, pot_temperature: xr.DataArray, hor_axes: Iterable[str]
) -> xr.DataArray:
    """vertical heat flux w' theta'"""
    w_prime = vert_vel - vert_vel.mean(hor_axes)
    theta_prime = pot_temperature - pot_temperature.mean(hor_axes)
    ans = w_prime * theta_prime
    ans.name = "vertical_heat_flux"
    ans["long_name"] = r"$w' \theta'$ "
    return ans
