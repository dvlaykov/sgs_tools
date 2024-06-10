import xarray as xr
import numpy as np
from typing import Union, List

from .tensor_algebra import grad_vector, traceless
from .staggered_grid import interpolate_to_grid

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
    gradvel = grad_vector(vel, space_dims, new_dim)
    sij = rate_of_strain(gradvel, (vec_dim, new_dim))
    if incompressible:
        sij = traceless(sij,  (vec_dim, new_dim))
    if cache is not None:
        cache['gradvel'] = gradvel
    return sij