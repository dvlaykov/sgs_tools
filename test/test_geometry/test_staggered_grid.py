import numpy as np
import pytest
import xarray as xr
from sgs_tools.geometry.grid import CoordScalar, UniformCartesianGrid
from sgs_tools.geometry.staggered_grid import diff_lin_on_grid
from sgs_tools.simple_flows.SimpleShear import ScalarGradient


@pytest.fixture
def u_grid():
    return UniformCartesianGrid([0.5, 0, 0.5], [1, 1, 1])


@pytest.fixture
def v_grid():
    return UniformCartesianGrid([0, 0.5, 0.5], [1, 1, 1])


@pytest.fixture
def w_grid():
    return UniformCartesianGrid([0.5, 0.5, 0], [1, 1, 1])


@pytest.fixture
def vel_shear(u_grid, v_grid, w_grid):
    u = (
        CoordScalar(grid=u_grid, direction=0, amplitude=1)
        .scalar([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_face", "x3": "z_centre"})
    )

    v = (
        CoordScalar(grid=v_grid, direction=1, amplitude=2)
        .scalar([64, 64, 64])
        .rename({"x1": "x_face", "x2": "y_centre", "x3": "z_centre"})
    )

    w = (
        CoordScalar(grid=w_grid, direction=2, amplitude=3)
        .scalar([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_centre", "x3": "z_face"})
    )
    return xr.Dataset({"u": u, "v": v, "w": w})


@pytest.fixture
def scalar_x_gdt(u_grid):
    return (
        ScalarGradient(u_grid, "x1", 1.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_face", "x3": "z_centre"})
    )


@pytest.fixture
def scalar_y_gdt(v_grid):
    return (
        ScalarGradient(v_grid, "x2", 2.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_face", "x2": "y_centre", "x3": "z_centre"})
    )


@pytest.fixture
def scalar_z_gdt(w_grid):
    return (
        ScalarGradient(w_grid, "x3", 3.0, 0.0)
        .field([64, 64, 64])
        .rename({"x1": "x_centre", "x2": "y_centre", "x3": "z_face"})
    )


def test_diff_lin_on_grid(scalar_x_gdt, scalar_y_gdt, scalar_z_gdt):
    grad = diff_lin_on_grid(scalar_x_gdt, "x_centre")
    const_nan_right = xr.full_like(grad, 1.0)
    const_nan_right[-1, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_x_gdt, "y_face")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, 0, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_x_gdt, "z_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, :, -1] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "x_face")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[0, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "y_centre")
    const_nan_right = xr.full_like(grad, 2.0)
    const_nan_right[:, -1, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_y_gdt, "z_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, :, -1] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "x_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[-1, :, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "y_centre")
    const_nan_right = xr.full_like(grad, 0.0)
    const_nan_right[:, -1, :] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)

    grad = diff_lin_on_grid(scalar_z_gdt, "z_face")
    const_nan_right = xr.full_like(grad, 3.0)
    const_nan_right[:, :, 0] = np.nan
    xr.testing.assert_equal(grad, const_nan_right)
