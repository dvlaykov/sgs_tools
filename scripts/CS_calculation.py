from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import xarray as xr
from sgs_tools.geometry.staggered_grid import (
    compose_vector_components_on_grid,
    interpolate_to_grid,
)
from sgs_tools.geometry.vector_calculus import grad_scalar
from sgs_tools.io.um import (
    read_stash_files,
    rename_variables,
    restrict_ds,
    unify_coords,
)
from sgs_tools.physics.fields import strain_from_vel
from sgs_tools.sgs.dynamic_coefficient import dynamic_coeff
from sgs_tools.sgs.filter import Filter, box_kernel, weight_gauss_3d, weight_gauss_5d
from sgs_tools.sgs.Smagorinsky import (
    DynamicSmagorinskyHeatModel,
    DynamicSmagorinskyVelocityModel,
    SmagorinskyHeatModel,
    SmagorinskyVelocityModel,
)
from sgs_tools.util.timer import timer
from xarray.core.types import T_Xarray


def parser() -> dict[str, Any]:
    parser = ArgumentParser(
        description="Compute dynamic Smagorinsky coefficients as function of scale from UM NetCDF output and store them in a NetCDF files",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    fname = parser.add_argument_group("I/O datasets on disk")
    fname.add_argument(
        "data_dir",
        type=Path,
        help="location of UM data, recognize glob patterns, in case diagnostics are split across multiple NetCDF files",
    )
    fname.add_argument(
        "fname_prefix",
        type=str,
        help="""Common prefix of filenames to read (should belong to the same simulation).
                Will read files matching the glob pattern '{prefix}*[file_codes]*.nc'.""",
    )
    fname.add_argument(
        "--file_codes",
        type=str,
        nargs="+",
        default=("b", "r"),
        help="""STASH stream file codes. Will read files matching the glob pattern '{prefix}*[file_codes}]*.nc'.
                Expect them to contain the fields 'u', 'v', 'w', 'theta'.
             """,
    )
    fname.add_argument(
        "h_resolution",
        type=float,
        help="horizontal resolution (will use to overwrite horizontal coordinates). **NB** works for ideal simulations",
    )


    fname.add_argument(
        "output_path",
        type=Path,
        help="output path, will create/overwrite existing file and create any missing intermediate directories",
    )

    filter = parser.add_argument_group("Filter parameters")
    filter.add_argument(
        "--filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel to use for scale separation.",
    )

    filter.add_argument(
        "--filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform filter at, in number of cells. "
        "If a single value is given, it will be used for all `regularize_filter_scales`. "
        "Otherwise, must provide as many values as for `regularize_filter_scales`",
    )

    filter.add_argument(
        "--regularize_filter_type",
        type=str,
        default="box",
        choices=["box", "gaussian"],
        help="Shape of filter kernel used for coefficient regularization.",
    )

    filter.add_argument(
        "--regularize_filter_scales",
        type=int,
        default=(2,),
        nargs="+",
        help="Scales to perform regularization at, in number of cells. "
        "If a single value is given, it will be used for all `filter_scale`. "
        "Otherwise, must provide as many values as for `filter_scale`",
    )

    plotting = parser.add_argument_group("Plotting parameters")

    plotting.add_argument(
        "--plot_show",
        action="store_true",
        help="flag to display generated plots",
    )

    plotting.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="output directory, for storing generated plots",
    )

    # parse arguments into a dictionary
    args = vars(parser.parse_args())

    # parameter consistency checks
    if args["filter_type"] == "gaussian":
        assert all(
            [x in [2, 4] for x in args["filter_scales"]]
        ), "Gaussian filters only support scales 2 and 4 for now..."

    # singleton filter_scales or regularize_filter_scales means broadcast against the other
    if len(args["filter_scales"]) == 1:
        args["filter_scales"] = args["filter_scales"] * len(
            args["regularize_filter_scales"]
        )

    if len(args["regularize_filter_scales"]) == 1:
        args["regularize_filter_scales"] = args["regularize_filter_scales"] * len(
            args["filter_scales"]
        )

    assert len(args["filter_scales"]) == len(args["regularize_filter_scales"])
    return args


def data_ingest(
    dir_path: Path,
    fname_pattern: str,
    res: float,
    file_codes: list[str] = ["b", "r"],
    required_fields: list[str] = ["u", "v", "w", "theta"],
):
    """read and pre-process UM data

    :param dir_path: directory containing UM NetCDF files
    :param fname_prefix: common prefix of filenames to read (should belong to the same simulation)
    :param float: horizontal resolution (will use to overwrite horizontal coordinates). **NB** works for ideal simulations
    :param file_codes: single-letter filename codes distinguishing diagnostic output files, default is ['b', 'r']
    :parm  required_fields: list of fields to read and pre-process. Defaults to ['u', 'v', 'w', 'theta']
    """
    # all the fields we will need for the Cs calculations
    simulation = read_stash_files(dir_path, fname_pattern, file_codes=file_codes)
    # parse UM stash codes into variable names
    simulation = rename_variables(simulation)

    # restrict to interesting fields and rename to simple names
    simulation = restrict_ds(simulation, fields=required_fields)

    # unify coordinatesh
    simulation = unify_coords(simulation, res=res)

    # interpolate all vars to a cell-centred grid
    centre_dims = ["x_centre", "y_centre", "z_theta"]
    simulation = interpolate_to_grid(simulation, centre_dims, drop_coords=True)
    # rename spatial dimensions to 'xyz'
    simple_dims = ["x", "y", "z"]
    dim_names = {d_new: d_old for d_new, d_old in zip(centre_dims, simple_dims)}
    simulation = simulation.rename(dim_names)

    return simulation


def make_filter(shape: str, scale: int, dims=Sequence[str]) -> Filter:
    """make filter object. **NB** Choices are limited!!!

    :param shape: shape of filter kernel
    :param scale: length scale of filter kernel
    :param dims: dimensions along which to filter
    """
    if shape == "gaussian":
        if scale == 2:
            return Filter(weight_gauss_3d, dims)
        elif scale == 4:
            return Filter(weight_gauss_5d, dims)
        else:
            raise ValueError(f"Unsupported filter scale{scale} for gaussian filters")
    elif shape == "box":
        return Filter(box_kernel([scale, scale]), dims)
    else:
        raise ValueError(f"Unsupported filter shape {shape}")


def add_scale_coords(
    ds: T_Xarray, scales: list[float], regularization_scales: list[float]
) -> T_Xarray:
    """add scale dim and regularization_scale coordinate
    :param ds: input dataset/dataarray
    :param scales: the coordinates for the scale dimension
    :param regularization_scales: sequence of coordinates
    :return: the update dataset/dataarray
    """
    if "scale" not in ds.dims:
        ds = ds.expand_dims(scale=scales)
    else:
        ds = ds.assign_coords(scale=("scale", scales))
    ds = ds.assign_coords(regularization_scale=("scale", regularization_scales))
    ds["scale"].attrs["units"] = r"$\Delta x$"
    ds["regularization_scale"].attrs["units"] = r"$\Delta x$"
    return ds


def main() -> None:
    # read and pre-process simulation
    with timer("Arguments", "ms"):
        args = parser()

    # read UM stasth files: data
    with timer("Read Dataset", "s"):
        simulation = data_ingest(
            args["data_dir"],
            args["fname_prefix"],
            args["h_resolution"],
            file_codes=args["file_codes"],
            required_fields=["u", "v", "w", "theta"],
        )

    # check scales make sense
    nhoriz = min(simulation["x"].shape[0], simulation["y"].shape[0])
    for scale in args["filter_scales"]:
        assert scale in range(
            1, nhoriz
        ), f"scale {scale} must be less than horizontal number of grid cells {nhoriz}"
    for scale in args["regularize_filter_scales"]:
        assert (
            scale in range(1, nhoriz)
        ), f"regularization_scale {scale} must be less than horizontal number of grid cells {nhoriz}"

    with timer("Extract grid-based fields", "s"):
        # ensure velocity components are co-located
        simple_dims = ["x", "y", "z"]  # coordinates already exist in simulation
        vel = compose_vector_components_on_grid(
            [simulation["u"], simulation["v"], simulation["w"]],
            simple_dims,
            name="vel",
            vector_dim="c1",
        )

        # compute strain and potential temperature gradient
        sij = strain_from_vel(
            vel,
            space_dims=simple_dims,
            vec_dim="c1",
            new_dim="c2",
            make_traceless=True,
        )
        grad_theta = grad_scalar(
            simulation["theta"],
            space_dims=simple_dims,
            new_dim_name="c1",
            name="grad_theta",
        )

    with timer("Setup SGS models", "ms"):
        # setup dynamic Smagorinsky model for velocity
        smag_vel = SmagorinskyVelocityModel(
            vel, sij, cs=1.0, dx=args["h_resolution"], tensor_dims=["c1", "c2"]
        )
        dyn_smag_vel = DynamicSmagorinskyVelocityModel(smag_vel)

        # setup dynamic Smagorinsky model for potential temperature
        smag_theta = SmagorinskyHeatModel(
            vel,
            grad_theta,
            sij,
            ctheta=1.0,
            dx=args["h_resolution"],
            tensor_dims=["c1", "c2"],
        )
        dyn_smag_theta = DynamicSmagorinskyHeatModel(smag_theta, simulation["theta"])

    # process for each filter scale
    with timer("Setup Cs", "s", "Setup Cs"):
        cs_iso_at_scale = []
        cs_diag_at_scale = []
        ctheta_at_scale = []
        for scale, regularization_scale in zip(
            args["filter_scales"], args["regularize_filter_scales"]
        ):
            with timer(f"  At scale {scale}", "s", f"  At scale {scale}"):
                filter = make_filter(args["filter_type"], scale, ["x", "y"])
                regularization = make_filter(
                    args["regularize_filter_type"], regularization_scale, ["x", "y"]
                )
                with timer("    Cs isotropic", "s"):
                    # compute isotropic Cs for velocity
                    cs_isotropic = dynamic_coeff(
                        dyn_smag_vel, filter, regularization, ["c1", "c2"]
                    )
                    # force execution for timer logging
                    cs_iso_at_scale.append(cs_isotropic)  # .load())

                with timer("    Cs diagonal", "s"):
                    # compute diagonal Cs for velocity
                    cs_diagonal = dynamic_coeff(
                        dyn_smag_vel, filter, regularization, ["c2"]
                    )
                    # force execution for timer logging
                    cs_diag_at_scale.append(cs_diagonal)  # .load())

                with timer("    Cs theta isotropic", "s"):
                    # compute isotropic Cs for velocity
                    ctheta = dynamic_coeff(
                        dyn_smag_theta, filter, regularization, ["c1"]
                    )
                    # force execution for timer logging
                    ctheta_at_scale.append(ctheta)  # .load())

    with timer("Collect coefficients", "s"):
        cs_iso_at_scale = xr.concat(cs_iso_at_scale, dim="scale")
        cs_diag_at_scale = xr.concat(cs_diag_at_scale, dim="scale")
        ctheta_at_scale = xr.concat(ctheta_at_scale, dim="scale")

        coeff_at_scale = xr.Dataset(
            {
                "Cs_isotropic": cs_iso_at_scale,
                "Cs_diagonal": cs_diag_at_scale,
                "Ctheta_isotropic": ctheta_at_scale,
            }
        )
        # add scale coordinates
        coeff_at_scale = add_scale_coords(
            coeff_at_scale,
            list(args["filter_scales"]),
            list(args["regularize_filter_scales"]),
        )
    # plot horizontal mean profiles
    with timer("Plotting", "s"):
        if args["plot_show"] or args["plot_path"] is not None:
            if len(args["filter_scales"]) > 1:
                row_lbl = "scale"
            else:
                row_lbl = None

            fig_cs_diag = (
                cs_diag_at_scale.mean(["x", "y"])
                .plot(x="t_0", row=row_lbl, col="c1", robust=True)
                .fig
            )
            # -1 because no label on colorbar
            for ax in fig_cs_diag.axes[:-1]:
                ax.text(
                    0.05, 0.85, r"$C_s$ diagonal", fontsize=14, transform=ax.transAxes
                )

            plt.figure()
            q = cs_iso_at_scale.mean(["x", "y"]).plot(
                x="t_0", row=row_lbl, col_wrap=3, robust=True
            )
            q.axes.text(
                0.05, 0.85, r"$C_s$ isotropic", fontsize=14, transform=q.axes.transAxes
            )
            fig_cs = q.get_figure()

            plt.figure()
            q = ctheta_at_scale.mean(["x", "y"]).plot(
                x="t_0", row=row_lbl, col_wrap=3, robust=True
            )
            q.axes.text(
                0.05,
                0.85,
                r"$C_\theta$ isotropic",
                fontsize=14,
                transform=q.axes.transAxes,
            )
            fig_ctheta = q.get_figure()

        if args["plot_path"] is not None:
            print(f"Saving plots to {args['plot_path']}")
            args["plot_path"].mkdir(parents=True, exist_ok=True)
            fig_cs.savefig(args["plot_path"] / "Cs_isotropic.png", dpi=180)
            fig_cs_diag.savefig(args["plot_path"] / "Cs_diagonal.png", dpi=180)
            fig_ctheta.savefig(args["plot_path"] / "Ctheta_isotropic.png", dpi=180)

    # interactive plotting out of time
    if args["plot_show"]:
        plt.show()

    with timer("Write to disk", "s"):
        coeff_at_scale.to_netcdf(
            args["output_path"], mode="w", compute=True, unlimited_dims=["t_0", "scale"]
        )


if __name__ == "__main__":
    with timer("Total execution time", "min"):
        main()
