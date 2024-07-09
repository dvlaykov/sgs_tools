import xarray as xr
from numpy import nan
from typing import Union, List, Dict, Iterable


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

def diff_lin_on_grid(ds, dim, periodic_field = False):

    ''' differentiate on staggered grid
        return the derivative on the grid with offset staggering

        this assumes that we have index staggering:
        c_face[i] --  +  ------ c_face[i+1] -------- +
        |             |                 |            |
        + ------- c_cen[i] ------------ + ---- c_cen[i+1]

        BCs: coordinate is extrapolated with the neighbouring cell spacing;
        no BCs inclfield is treated as periodic

    '''

    def delta(coord, shift):
        ''' constant extrapolation '''
        if shift == 1:
            coord_n1 = 2* coord[0] - coord[1]
            return (coord - coord.shift({coord.dims[0]:shift}, fill_value=coord_n1))
        elif shift == -1:
            coord_Np1 = 2* coord[-1] - coord[-2]
            return (coord - coord.shift({coord.dims[0]:shift}, fill_value=coord_Np1))

    def centre_point(coord, shift):
        if shift == 1:
            extra = coord[0] - (coord[1] - coord[0])
        elif shift == -1:
            extra = coord[-1] + (coord[-1] - coord[-2])
        return (coord + coord.shift({coord.dims[0]:shift}, fill_value=extra))/2

    coord = ds[dim].astype(float) # just in case
    val = ds.astype(float)
    if dim.endswith('_face'):
        new_dim = dim.rstrip('_face') +  '_centre'
        shift = 1
    elif dim.endswith('_centre'):
        new_dim = dim.rstrip('_centre') + '_face'
        shift = -1
    else:
        raise ValueError(f"Unrecognizable coordinate staggering for dimension {dim}")

    new_coord = centre_point(coord, shift)

    if periodic_field:
        deriv = ((val - val.roll({dim:shift}))/
                   delta(coord, shift))
    else:
        deriv = ((val - val.shift({dim:shift}, fill_value = nan))/
                   delta(coord, shift))
    deriv = deriv.rename({dim:new_dim})
    deriv[new_dim] = new_coord.rename({dim:new_dim})
    for c in deriv.coords:
        if deriv[c].dims and deriv[c].dims[0] == new_dim and not c == new_dim:
                deriv[c] = centre_point(deriv[c], shift)
    return deriv

def grad_on_cart_grid(ds:xr.DataArray,
                      space_dims: Iterable[str],
                      periodic_field: Iterable[bool] = [False, False, False]
                      ) -> xr.Dataset:
    ''' differentiate a scalar with respect to space dims on staggered grid
        BCs: coordinate is extrapolated with the neighbouring cell spacing;
             no BCs inclfield is treated as periodic
    '''
    name = ds.name
    if name is None:
        name = 'd'
    grad = []
    for i, d in enumerate(space_dims):
         #label should turn into "{name}_dx", etc.
         pd = diff_lin_on_grid(ds, d, periodic_field[i])
         pd.name = f'{name}_{d[0]}'
         grad.append(pd)
    # merge with compat=minimal to drop conflicting non-dimension coordinates
    # with the same name but on staggered dimensions
    # e.g. d_x.x_cu(x_face) and d_y.x_cu(x_centre) dims
    return xr.merge(grad, compat = 'minimal')

def grad_vec_on_grid(ds: xr.Dataset,
                     target_dims: Iterable[str] = ['x_centre', 'y_centre', 'z_centre'],
                     new_dim_name: Iterable[str] = ['c1', 'c2'],
                     name: Union[str, None] = None) -> xr.DataArray:
    ''' computes gradient of a vector described onto target dimensions
        ds: should be a dataset which only contains the components of the vector in sorted order and target coordinates.
        target_dims: the dimensions to compute the derivative on (must be coordinates in input dataset)
        new_dim_name: the names of the new dimensions: [vector component, differential component]
        name : name for output dataarray (optional)
    '''
    vec_name = new_dim_name[0]
    d_name = new_dim_name[1]
    gradvec_comp = {}

    for i, f in enumerate(ds):
        #individual sapce dimensions for each staggered field
        space_dims = sorted([d for d in ds[f].dims if d[0] in 'xyz'])
        grad_f = grad_on_cart_grid(ds[f], space_dims)
        # interpolate onto target coordinates from original ds (must exist)
        # careful grad_on_cart_grid will create coordinates with possibly classhing names
        # so make an explicit coordinate map
        coord_map = {}
        for k in grad_f.dims:
            if k[0] in 'xyz':
                #name match by first character xyz
                target_coord = [c for c in target_dims if c[0] == k[0]][0]
                coord_map[k] = ds[target_coord]
        grad_f_on_cent = interpolate_to_grid(grad_f, coord_map = coord_map)
        #convert to a dataarray
        grad_f_on_cent = grad_f_on_cent.to_dataarray(d_name).sortby(d_name)
        # rename c1 coordinates: fragile this works only because of
        # the naming in interpolate_to_grid
        grad_f_on_cent[d_name] = [f'd{x.item()[-1]}' for x in grad_f_on_cent[d_name]]
        grad_f_on_cent[d_name] = [f'd{x.item()[-1]}' for x in grad_f_on_cent[d_name]]
        gradvec_comp[f] = grad_f_on_cent

    gradvec_da = xr.Dataset(gradvec_comp).to_dataarray(dim = vec_name)
    gradvec_da = gradvec_da.sortby(vec_name)
    # rename c2 coordinates: less fragile but still a bit idiosyncratic
    gradvec_da[vec_name] = [f'v{i+1}' for i in range(len(gradvec_comp))]
    if not name is None:
        gradvec_da.name = name
    return gradvec_da