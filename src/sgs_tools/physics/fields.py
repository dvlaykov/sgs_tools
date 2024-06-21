import xarray as xr
import numpy as np
from typing import Union, List

from .staggered_grid import interpolate_to_grid
from .tensor_algebra import grad_vector, traceless, symmetrise

def rate_of_strain(gradvel, dims = ('c1', 'c2')):
    ''' 0.5 (grad_vel + grad_vel.transpose) '''
    transpose_map = dict([dims, dims[::-1]])
    sij = 0.5 * ( gradvel + gradvel.rename(transpose_map))
    sij.name = 'Sij'
    return sij

def strain_from_vel(vel, space_dims, vec_dim, new_dim = 'c2', incompressible:bool=True,
            cache: Union[None, xr.Dataset] = None) -> xr.DataArray:
    ''' compute rate of strain from velocity
        if cache is given will attempt to store intermediate calcualtion
        of gradvel in it
    '''
    gradvel = grad_operator(vel, space_dims, new_dim)
    sij = symmetrise(gradvel, (vec_dim, new_dim))
    if incompressible:
        sij = traceless(sij,  (vec_dim, new_dim))
    if cache is not None:
        cache['gradvel'] = gradvel
    return sij

def vertical_heat_flux(vert_vel, pot_temperature, hor_axes):
    '''vertical heat flux w' theta' '''
    w_prime = vert_vel - vert_vel.mean(hor_axes)
    theta_prime = pot_temperature - pot_temperature.mean(hor_axes)
    ans = w_prime * theta_prime
    ans.name ='vertical_heat_flux'
    ans['long_name'] = r"$w' \theta'$ "
    return ans