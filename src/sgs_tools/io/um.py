import xarray as xr
import numpy as np
import os, re
from typing import Iterable

from pathlib import Path


base_fields_dict = {
    "U_COMPNT_OF_WIND_AFTER_TIMESTEP": "u",
    "V_COMPNT_OF_WIND_AFTER_TIMESTEP": "v",
    "W_COMPNT_OF_WIND_AFTER_TIMESTEP": "w",
    "THETA_AFTER_TIMESTEP": "theta",
    #'TEMPERATURE_ON_THETA_LEVELS' : 'T',
    # 'PRESSURE_AT_THETA_LEVELS_AFTER_TS' : 'P'
}
Water_dict = {
    "CLD_LIQ_MIXING_RATIO__mcl__AFTER_TS": "q_l",
    "CLD_ICE_MIXING_RATIO__mcf__AFTER_TS": "q_i",
    "LARGE_SCALE_RAINFALL_RATE____KG_M2_S": "rain",
    "GRAUPEL_MIXING_RATIO__mg__AFTER_TS": "q_g",
    "SPECIFIC_HUMIDITY_AFTER_TIMESTEP": "q_v",
}
Smagorinsky_dict = {
    "SMAG__S__SHEAR_TERM_": "s_smag",
    "SMAG__VISC_M": "SMAG__VISC_M",
    "SMAG__VISC_H": "SMAG__VISC_H",
    "SHEAR_AT_SCALE_DELTA": "s",
    "MIXING_LENGTH_RNEUTML": "csDelta",
    "CS_THETA": "cs_theta",
    "TURBULENT_KINETIC_ENERGY": "tke",
}
dynamic_SGS_dict = {
    "CS_SQUARED_AT_2_DELTA": "cs2d",
    "CS_SQUARED_AT_4_DELTA": "cs4d",
    "CS_THETA_AT_SCALE_2DELTA": "cs_theta_2d",
    "CS_THETA_AT_SCALE_4DELTA": "cs_theta_4d",
}
dynamic_SGS_diag_dict = {
    "LijMij_CONT_TENSORS": "lm",
    "QijNij_CONT_TENSORS": "qn",
    "MijMij_CONT_TENSORS": "mm",
    "NijNij_CONT_TENSORS": "nn",
    "HjTj_CONT_VECTORS": "ht",
    "TjTj_CONT_VECTORS": "tt",
    "RjFj_CONT_VECTORS": "rf",
    "FjFj_CONT_VECTORS": "ff",
    "SHEAR_AT_SCALE_2DELTA": "s2d",
    "SHEAR_AT_SCALE_4DELTA": "s4d",
    "D11_TENSOR_COMPONENT": "diag11",
    "D22_TENSOR_COMPONENT": "diag22",
    "D33_TENSOR_COMPONENT": "diag33",
    "D13_TENSOR_COMPONENT": "diag13",
    "D23_TENSOR_COMPONENT": "diag23",
    "D12_TENSOR_COMPONENT": "diag12",
    "Lagrangian_averaged_LijMij_tensors": "LM",
    "Lagrangian_averaged_MijMij_tensors": "MM",
    "Lagrangian_averaged_QijNij_tensors": "QN",
    "Lagrangian_averaged_NijNij_tensors": "NN",
    "Lagrangian_averaged_HjTj_vector": "HT",
    "Lagrangian_averaged_TjTj_vector": "TT",
    "Lagrangian_averaged_RjFj_vector": "RF",
    "Lagrangian_averaged_FjFj_vector": "FF",
    "Tdecorr_momentum": "Tdecorr_momentum",
    "Tdecorr_heat": "Tdecorr_heat",
}
field_names_dict = base_fields_dict | Water_dict | Smagorinsky_dict | dynamic_SGS_dict


# IO
# open datasets
def read_stash_files(
    base_dir: Path | str, prefix: str, file_codes: Iterable[str]
) -> xr.Dataset:
    """combine a list of output Stash files
    base_dir: basic directory
    rose_suite: the name of the suite wihtout the "u-" prefix
    resolution: (string) -- used for filename parsing
    file_codes: list of strings, the codes used by Stash system,
                e.g. "r" for the main fields, we put diagnostics in "b"
    """
    datasets = []
    for c in file_codes:

        file = Path(base_dir) / f"{prefix}_p{c}000.nc"
        print(f"Reading {file}")
        datasets.append(xr.open_dataset(file, chunks={}))
    return xr.merge(datasets)


# Pre-process input UM arrays
def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    """rename stash variables to something meaningful, i.e. long_name"""

    def varname_str(varStr: str):
        """replace special characters in varStr with "_" """
        return re.sub(r"\W|^(?=\d)", "_", varStr)

    varname_dict = {}
    for var in ds:
        lname = ds[var].attrs.get("long_name", "None").rstrip("|").rstrip()
        if "STASH" in str(var) and lname is not None:
            ds[var].attrs["original_vname"] = var
            varname_dict[var] = varname_str(lname)
    ds = ds.rename(varname_dict)

    # swap vertical dimension to height about sea level (in m) to better compare across simulations
    vertical_dim_map = {
        "thlev_eta_theta": "thlev_zsea_theta",
        "thlev_bl_eta_theta": "thlev_bl_zsea_theta",
        "rholev_eta_rho": "rholev_zsea_rho",
    }
    vertical_dim_map = {
        k: v for k, v in vertical_dim_map.items() if k in ds and v in ds
    }

    ds = ds.swap_dims(vertical_dim_map)
    # swap to an easy time-dimension
    tname = "min15T0"
    torigin = ds["min15T0_0"][0]
    for tsuffix in "", "_0":
        delta_t = np.rint(
            (ds[tname + tsuffix] - torigin) / np.timedelta64(1, "m")
        ).astype(int)
        ds = ds.assign_coords({"t" + tsuffix: (tname + tsuffix, delta_t.data)})
        ds["t" + tsuffix].attrs = {"standard_name": "time", "axis": "T", "unit": "min"}
        ds = ds.swap_dims({tname + tsuffix: "t" + tsuffix})
    return ds


def restrict_ds(ds: xr.Dataset, fields=None) -> xr.Dataset:
    """restrict the dataset to fields of interest (if fields=None) or to
    a pre-selected list of fields]

    and rename vars, coords, dims
    """

    intersection = {k: v for k, v in field_names_dict.items() if k in ds}
    if not fields is None:
        intersection = {k: v for k, v in intersection.items() if v in fields}

    # print ("Missing fields:", {k for k in fields if k not in intersection})

    # drop all secondary fields
    ds = ds[list(intersection)]
    # rename primary fields for convenience
    ds = ds.rename(intersection)

    # rename dimensions fields for clarity
    dim_names = {
        "thlev_zsea_theta": "z_theta",
        "rholev_zsea_rho": "z_rho",
        "latitude_t": "y_theta",
        "longitude_t": "x_theta",
        "latitude_cu": "y_cu",
        "longitude_cu": "x_cu",
        "latitude_cv": "y_cv",
        "longitude_cv": "x_cv",
    }
    intersection = {k: v for k, v in dim_names.items() if k in ds}
    ds = ds.rename(intersection)
    return ds


# unify coordinates and implement correct x-spacing
# xarray doesn't handle duplicate dimensions well, so use clunkily split-rename-merge
def unify_coords(ds: xr.Dataset, res: int) -> xr.Dataset:
    """
    unify coordinate names
    implement correct x-spacing using res, assume res is given in meters
    rename coordinates with reference to logically-cartesian grid
    """
    # actual coordinates
    x_face = np.linspace(
        0, (ds.x_theta.size) * res, num=ds.x_theta.size, endpoint=False
    )
    x_centre = x_face + res / 2

    # split into centered and staggered variables
    cent_vars = [x for x in ds if "x_theta" in ds[x].dims and "y_theta" in ds[x].dims]
    stag_vars = [x for x in ds if x not in cent_vars]

    # rename dimensions/coords of staggered variables
    ds_stag = ds[stag_vars]
    ds_stag["x_centre"] = xr.DataArray(
        x_centre, coords={"x_cv": ds.x_cv}, dims="x_cv", name="x_centre"
    )
    ds_stag["y_centre"] = xr.DataArray(
        x_centre, coords={"y_cu": ds.y_cu}, dims="y_cu", name="y_centre"
    )
    ds_stag["x_face"] = xr.DataArray(
        x_face, coords={"x_cu": ds.x_cu}, dims="x_cu", name="x_face"
    )
    ds_stag["y_face"] = xr.DataArray(
        x_face, coords={"y_cv": ds.y_cv}, dims="y_cv", name="y_face"
    )

    ds_stag = ds_stag.swap_dims(
        {
            "x_cu": "x_face",
            "x_cv": "x_centre",
            "y_cu": "y_centre",
            "y_cv": "y_face",
        }
    )

    # rename dimensions/coords of centred variables
    ds_cent = ds[cent_vars]
    ds_cent["x_centre"] = xr.DataArray(
        x_centre, coords={"x_theta": ds.x_theta}, dims="x_theta", name="x_centre"
    )
    ds_cent["y_centre"] = xr.DataArray(
        x_centre, coords={"y_theta": ds.y_theta}, dims="y_theta", name="y_centre"
    )
    ds_cent = ds_cent.swap_dims({"x_theta": "x_centre", "y_theta": "y_centre"})

    ds = xr.merge([ds_stag, ds_cent])

    return ds


def compose_diagnostic_tensor(ds: xr.Dataset) -> xr.Dataset:
    diag_ij = xr.concat(
        [
            xr.concat([ds.diag11, ds.diag12, ds.diag13], "c1"),
            xr.concat([ds.diag12, ds.diag22, ds.diag23], "c1"),
            xr.concat([ds.diag13, ds.diag23, ds.diag33], "c1"),
        ],
        "c2",
    )
    diag_ij.name = "Diag_ij"
    diag_ij.attrs["long_name"] = "Diagnostic tensor"

    ds["Diag_ij"] = diag_ij
    ds = ds.drop_vars(["diag11", "diag22", "diag33", "diag12", "diag13", "diag23"])
    return ds
