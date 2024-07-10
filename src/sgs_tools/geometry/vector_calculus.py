from typing import List

import numpy as np
import xarray as xr


# Vector calculus
def grad_vector(
    vec: xr.DataArray, space_dims: List[str], new_dim_name: str = "c2", name=None
) -> xr.DataArray:
    """gradient tensor of a vector -- centred 2nd order difference, reduced to 1st order at  boundaries
    vec : xarray.DataArray, with spatial coordinates `space_dims`
    space_dims : list of names of spatial dimensions w.r.t. which to take the gradient
    new_dim_name: string, name of new dimension
    return : gradient (assuming cartesian geometry) as xarray.DataArray
    """

    gradvec = []
    for dim in space_dims:
        gradvec.append(vec.differentiate(dim, edge_order=1))
    gradvec_xarr = xr.concat(
        gradvec, dim=xr.DataArray(range(1, len(space_dims) + 1), dims=[new_dim_name])
    )
    if name is not None:
        gradvec_xarr.name = name
    return gradvec_xarr


def grad_vector_lin(
    vec: xr.DataArray, space_dims: List[str], new_dim_name: str = "c2", name=None
) -> xr.DataArray:
    """gradient tensor of a vector -- 1st order backward finite-difference,
    vec : xarray.DataArray, with spatial coordinates `space_dims`
    space_dims : list of names of spatial dimensions w.r.t. which to take the gradient
    new_dim_name: string, name of new dimension
    return : gradient (assuming cartesian geometry) as xarray.DataArray
    """
    gradvec = []
    for dim in space_dims:
        coord = vec[dim].astype(float)  # just in case
        val = vec.astype(float)

        gradvec.append(
            (val - val.shift({dim: -1}, fill_value=np.nan))
            / (coord - coord.shift({dim: -1}, fill_value=np.nan))
        )
    gradvec_xarr = xr.concat(
        gradvec, dim=xr.DataArray(range(1, len(space_dims) + 1), dims=[new_dim_name])
    )
    if name is not None:
        gradvec_xarr.name = name
    return gradvec_xarr
