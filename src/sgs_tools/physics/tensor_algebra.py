import xarray as xr
import numpy as np


# Vector algebra
def tensor_self_outer_product(arr):
    ''' tensor product arr_i arr_j from vector field arr.
        arr: xarray Dataset with dimension 'c1' which will be used for the tensor product
        returns: xarray Dataset with the 'i' and 'j' dimensions sorted to the front.
    '''
    return (arr * arr.rename({'c1': 'c2'})).transpose('c1','c2',...)

def trace(tensor, dims = ('c1', 'c2')):
    ''' trace along 2 dimesions '''
    assert( tensor[dims[0]].size == tensor[dims[1]].size ) # only for square arrays
    assert( len(dims) == 2 ) #only 2-dimensional trace
    diagonal =  tensor.sel({dims[0] : tensor[dims[1]]})
    tr = diagonal.sum(dims[1])
    tr.name = 'Tr '+tensor.name
    return tr

#Make a tensor Traceless along 2 dimensions
def traceless(tensor, dims = ('c1', 'c2')):
    ''' returns a traceless version of tensor
        bug/unexpected behaviour when nan in trace
    '''
    #compute trace along dims
    trace_normed = trace(tensor, dims)/tensor[dims[1]].size

    #copy input for modification
    traceless = tensor.copy()
    #remove trace from diagonal
    for i in tensor[dims[0]]:
        traceless.loc[{dims[0]:i.item(), dims[1]:i.item()}] -= trace_normed
    return traceless

def Frobenius_norm(tensor, tens_dims = ['c1', 'c2']):
    ''' Frobenius norm of a tensor: |A| = sqrt(Aij Aij) '''
    return np.sqrt(xr.dot(tensor, tensor, dims = tens_dims))

# Vector calculus
def grad_vector(vel, space_dims = [], new_dim_name = 'c2'):
    ''' velocity gradient tensor -- centred 2nd order difference, reduced to 1st order at  boundaries
        vel : xarray.DataArray, with spatial coordinates x{1,2,...,ndim}
        space_dims : list of names of spatial dimensions w.r.t. which to take the gradient
        new_dim_name: string, name of new dimension
        return : velocity gradient tensor (assuming cartesian geometry) as xarray.DataArray
    '''
    gradvel = []
    for dim in space_dims:
        gradvel.append(vel.differentiate(dim, edge_order=1))
    gradvel = xr.concat(gradvel,
                        dim = xr.DataArray(range(1,len(space_dims)+1),
                                           dims = [new_dim_name]))
    gradvel.name = 'vel_gradient'
    return gradvel

def grad_vector_lin(vel, ndim = 3, new_dim_name = 'c2'):
    ''' velocity gradient tensor -- 1st order backward finite-difference,
        vel : xarray.DataArray, with spatial coordinates x{1,2,...,ndim}
        ndim : number of spatial dimensions
        new_dim_name: string, name of new dimension
        return : velocity gradient tensor (assuming cartesian geometry) as xarray.DataArray
    '''
    gradvel = []
    for j in range(1, ndim+1):
        gradvel.append( (vel - vel.shift({f'x{j}':-1}, fill_value=np.nan))/
                        (vel[f'x{j}'] - vel[f'x{j}'].shift({f'x{j}':-1}, fill_value=np.nan))
                      )
    gradvel = xr.concat(gradvel,
                        dim = xr.DataArray(range(1,ndim+1), dims = [new_dim_name]))
    gradvel.name = 'vel_gradient'
    return gradvel
