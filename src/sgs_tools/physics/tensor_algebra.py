import xarray as xr
import numpy as np
from typing import Iterable


# Vector algebra
def tensor_self_outer_product(arr):
    """tensor product arr_i arr_j from vector field arr.
    arr: xarray Dataset with dimension 'c1' which will be used for the tensor product
    returns: xarray Dataset with the 'i' and 'j' dimensions sorted to the front.
    """
    return (arr * arr.rename({"c1": "c2"})).transpose("c1", "c2", ...)


def trace(tensor, dims=("c1", "c2"), name=None):
    """trace along 2 dimesions"""
    assert tensor[dims[0]].size == tensor[dims[1]].size  # only for square arrays
    assert len(dims) == 2  # only 2-dimensional trace
    diagonal = tensor.sel({dims[0]: tensor[dims[1]]})
    tr = diagonal.sum(dims[1])
    if name is not None:
        tr.name = "Tr " + tensor.name
    return tr


# Make a tensor Traceless along 2 dimensions
def traceless(tensor, dims=("c1", "c2")):
    """returns a traceless version of tensor
    bug/unexpected behaviour when nan in trace
    """
    # compute trace along dims
    trace_normed = trace(tensor, dims) / tensor[dims[1]].size

    # copy input for modification
    traceless = tensor.copy()
    # remove trace from diagonal
    for i in tensor[dims[0]]:
        traceless.loc[{dims[0]: i.item(), dims[1]: i.item()}] -= trace_normed
    return traceless


def Frobenius_norm(tensor, tens_dims=["c1", "c2"]):
    """Frobenius norm of a tensor: |A| = sqrt(Aij Aij)"""
    return np.sqrt(xr.dot(tensor, tensor, dims=tens_dims))


def symmetrise(gradvec, dims=("c1", "c2"), name=None):
    """0.5 (grad_vec + grad_vec.transpose)"""
    transpose_map = dict([dims, dims[::-1]])
    sij = 0.5 * (gradvec + gradvec.rename(transpose_map))
    if name is not None:
        sij.name = name
    return sij
