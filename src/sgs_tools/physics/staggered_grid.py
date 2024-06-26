import xarray as xr
from typing import Union, List, Dict


def get_grid_spacing_coord(coord, new_dim):
    ''' get a coordinate for the grid spacing'''
    current_name = coord.dims[0]
    return coord.diff(dim= current_name, n = 1).rename({current_name: new_dim})

def interpolate_to_grid(ds: Union[xr.DataArray, xr.Dataset],
                        target_dims: List[str] = [],
                        coord_map: Dict[str, xr.DataArray] = None,
                        drop_coords: bool = True ) -> Union[xr.DataArray, xr.Dataset]:
    ''' Spatial interpolation to a target_grid
        ds: xarray Dataset/DataArray. Needs to have dimensions with coordinates
            that are labelled 'x*', 'y*', 'z*' etc. or

        target_dims: list of 3 dimension names  to interpolate to, in the order xdim, ydim, zdim.
            They must exist in ds as DataArry/coordinates
        coord_map: dictionary of {existing_dimension_in_ds : target_coordinate_as_DataArray}
        drop_coords: flag to exclude spatial coordinates relying on removed dims from output
    '''
    if isinstance(target_dims, str):
        target_dims = [target_dims]

    if target_dims:
        #assume spatial coordinates have x, y, z
        x_dims = [x for x in ds.dims if x.startswith('x')]
        y_dims = [y for y in ds.dims if y.startswith('y')]
        z_dims = [z for z in ds.dims if z.startswith('z')]
        missing_coords = [dim for dim in target_dims if dim not in ds.coords]
        assert len(missing_coords) == 0, f'missing target coordinages {missing_coords} from input data'
        x_target = [x for x in target_dims if x.startswith('x')]
        y_target = [y for y in target_dims if y.startswith('y')]
        z_target = [z for z in target_dims if z.startswith('z')]
        assert len(x_target) == 1
        x_target = x_target[0]
        assert len(y_target) == 1
        y_target = y_target[0]
        assert len(z_target) == 1
        z_target = z_target[0]

        coord_map    =   {}
        coord_map.update({x: ds[x_target] for x in x_dims if x != x_target})
        coord_map.update({y: ds[y_target] for y in y_dims if y != y_target})
        coord_map.update({z: ds[z_target] for z in z_dims if z != z_target})

    else:
        missing_dims = [dim for dim in coord_map.keys() if dim not in ds.dims]
        assert len(missing_dims) == 0, f'missing input dimensions {missing_dims} from input data'

    # rename any possible dimensions that will clash with coord-map targets:
    ds_interp = ds
    c_map = {}
    for d, c in coord_map.items():
        if c.name == d:
            ds_interp = ds_interp.rename({c.name:f'{c.name}_original'})
            c_map[f'{c.name}_original'] = coord_map[d]
        else:
            c_map[d] = coord_map[d]

    # drop any coordinates based on replaced dimensions
    drop_coords_list = []
    if drop_coords:
        for coord in ds_interp.coords:
            dims = ds_interp[coord].dims
            if dims and dims[0] in c_map:
                drop_coords_list.append(dims[0])

    #interpolate all fields to target grid and drop coordinates
    ds_interp = ds_interp.interp(c_map, method = "linear", assume_sorted = True).drop(drop_coords_list)
    return ds_interp


def compose_vector_components_on_grid(components: List[xr.DataArray],
                             target_dims: List[str]=[],
                             vector_dim: str = 'c1',
                             name: str = '',
                             long_name: str = '',
                             drop_coords: bool = True) -> xr.DataArray:
    ''' turn a list of arrays into a vector field
        if target_dims is given will interpolate onto it first,
        otherwise all componets must have the same dimesions and coordinates
    '''
    #interpolate
    if target_dims is []:
        assert all([all(components[0].dims == x.dims) for x in components[1:]] ),\
                "The components' dimensions don't match. "\
                "Choose a set of dimensions to interpolate t!"
        vec = components
    else:
        vec = [ interpolate_to_grid(comp, target_dims, drop_coords)
                    for comp in components]
    #combine into a vector
    vec = xr.concat(vec,
                    dim = xr.DataArray(range(1,4), dims = [vector_dim]))
    #add meta data
    if name:
        vec.name = name
    if long_name:
        vec.attrs['long_name'] = long_name
    return vec
