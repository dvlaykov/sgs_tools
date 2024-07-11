import numpy as np
import xarray as xr


# Vector algebra
def tensor_self_outer_product(arr: xr.DataArray) -> xr.DataArray:
    """tensor product :math:`a_i a_j` from vector field `arr`.
        Assumes that `arr` has dimensions ``c1`` but no dimension ``c2``

    :param arr: xarray Dataset with dimension `c1` which will be used for the tensor product
    :param returns: xarray DataArray with the 'i' and 'j' dimensions sorted to the front.
    """
    return (arr * arr.rename({"c1": "c2"})).transpose("c1", "c2", ...)


def trace(
    tensor: xr.DataArray, dims: list[str] = ["c1", "c2"], name=None
) -> xr.DataArray:
    """trace along 2 dimesions.

    :param tensor: tensor input
    :param dims: dimensions with respect to which to take the trace.
        The `tensor` must be square with respect to them
    """
    assert tensor[dims[0]].size == tensor[dims[1]].size  # only for square arrays
    assert len(dims) == 2  # only 2-dimensional trace
    diagonal = tensor.sel({dims[0]: tensor[dims[1]]})
    tr = diagonal.sum(dims[1])
    if name is not None:
        tr.name = "Tr " + str(tensor.name)
    return tr


# Make a tensor Traceless along 2 dimensions
def traceless(tensor: xr.DataArray, dims: list[str] = ["c1", "c2"]) -> xr.DataArray:
    """returns a traceless version of `tensor`. **NB** \: bug/unexpected behaviour when nan in trace

    :param tensor: tensor input
    :param dims: dimensions with respect to which to take the trace.
    """
    # compute trace along dims
    trace_normed = trace(tensor, dims) / tensor[dims[1]].size

    # copy input for modification
    traceless = tensor.copy()
    # remove trace from diagonal
    for i in tensor[dims[0]]:
        traceless.loc[{dims[0]: i.item(), dims[1]: i.item()}] -= trace_normed
    return traceless


def Frobenius_norm(
    tensor: xr.DataArray, tens_dims: list[str] = ["c1", "c2"]
) -> xr.DataArray:
    """Frobenius norm of a tensor\: :math:`|A| = \sqrt{Aij Aij}`

    :param tensor: tensor input
    :param dims: dimensions with respect to which to take the norm.
    """
    return np.sqrt(xr.dot(tensor, tensor, dims=tens_dims))


def symmetrise(
    tensor: xr.DataArray, dims: list[str] = ["c1", "c2"], name=None
) -> xr.DataArray:
    """:math:`0.5 (a + a^T)`.

    :param tensor: tensor input
    :param dims: dimensions with respect to which to take the transpose.
        Can be any length and the transpose means that the order is reversed.
        so ``[c1, c2, c3]`` will transpose to ``[c3, c2, c1]``.
        Note that no checks are performed whether `dims` are dimensions of `tensor` or
        whether `tensor` is square with respect to the transposed dimensions.
    :param name: name of symmetrized tensor.
    """
    transpose_map = dict([dims, dims[::-1]])
    sij = 0.5 * (tensor + tensor.rename(transpose_map))
    if name is not None:
        sij.name = name
    return sij
