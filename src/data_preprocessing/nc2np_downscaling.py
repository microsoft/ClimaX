import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

HOURS_PER_YEAR = 1440  # 1460, but 1440 is dividable by 32

def nc2np(path_mpi, path_era5, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean_input = {}
        normalize_std_input = {}
        normalize_mean_output = {}
        normalize_std_output = {}

    for year in tqdm(years):
        np_mpi_vars = {}
        np_era5_vars = {}

        # non-constant fields
        for var in variables:
            ps_mpi = glob.glob(os.path.join(path_mpi, var, f"*{year}*.nc"))
            ps_era5 = glob.glob(os.path.join(path_era5, var, f"*{year}*.nc"))
            ds_mpi = xr.open_mfdataset(ps_mpi, combine="by_coords", parallel=True)
            ds_era5 = xr.open_mfdataset(ps_era5, combine="by_coords", parallel=True)
            code = NAME_TO_VAR[var]

            if len(ds_mpi[code].shape) == 3:  # surface level variables
                ds_mpi[code] = ds_mpi[code].expand_dims("val", axis=1)
                ds_era5[code] = ds_era5[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                np_mpi_vars[var] = ds_mpi[code].to_numpy()[:HOURS_PER_YEAR]
                np_era5_vars[var] = ds_era5[code].to_numpy()[:HOURS_PER_YEAR]

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_mpi_yearly = np_mpi_vars[var].mean(axis=(0, 2, 3))
                    var_std_mpi_yearly = np_mpi_vars[var].std(axis=(0, 2, 3))
                    var_mean_era5_yearly = np_era5_vars[var].mean(axis=(0, 2, 3))
                    var_std_era5_yearly = np_era5_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean_input:
                        normalize_mean_input[var] = [var_mean_mpi_yearly]
                        normalize_std_input[var] = [var_std_mpi_yearly]
                        normalize_mean_output[var] = [var_mean_era5_yearly]
                        normalize_std_output[var] = [var_std_era5_yearly]
                    else:
                        normalize_mean_input[var].append(var_mean_mpi_yearly)
                        normalize_std_input[var].append(var_std_mpi_yearly)
                        normalize_mean_output[var].append(var_mean_era5_yearly)
                        normalize_std_output[var].append(var_std_era5_yearly)

            else:  # multiple-level variables, only use a subset
                assert len(ds_mpi[code].shape) == 4
                # all_levels = ds_era5["level"][:].to_numpy()
                all_levels = ds_mpi["plev"][:].to_numpy() / 100  # 92500 --> 925
                all_levels = all_levels.astype(int)
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level_mpi = ds_mpi.sel(plev=[level * 100.0])
                    ds_level_era5 = ds_era5.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_mpi_vars[f"{var}_{level}"] = ds_level_mpi[code].to_numpy()[:HOURS_PER_YEAR]
                    np_era5_vars[f"{var}_{level}"] = ds_level_era5[code].to_numpy()[:HOURS_PER_YEAR]

                    if partition == "train":  # compute mean and std of each var in each year
                        var_mean_mpi_yearly = np_mpi_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_mpi_yearly = np_mpi_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        var_mean_era5_yearly = np_era5_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_era5_yearly = np_era5_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        if var not in normalize_mean_input:
                            normalize_mean_input[f"{var}_{level}"] = [var_mean_mpi_yearly]
                            normalize_std_input[f"{var}_{level}"] = [var_std_mpi_yearly]
                            normalize_mean_output[f"{var}_{level}"] = [var_mean_era5_yearly]
                            normalize_std_output[f"{var}_{level}"] = [var_std_era5_yearly]
                        else:
                            normalize_mean_input[f"{var}_{level}"].append(var_mean_mpi_yearly)
                            normalize_std_input[f"{var}_{level}"].append(var_std_mpi_yearly)
                            normalize_mean_output[f"{var}_{level}"].append(var_mean_era5_yearly)
                            normalize_std_output[f"{var}_{level}"].append(var_std_era5_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_input = {k: np_mpi_vars[k][start_id:end_id] for k in np_mpi_vars.keys()}
            sharded_output = {k: np_era5_vars[k][start_id:end_id] for k in np_era5_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id:02}_inp.npz"),
                **sharded_input,
            )
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id:02}_out.npz"),
                **sharded_output,
            )

    if partition == "train":
        for var in normalize_mean_input.keys():
            normalize_mean_input[var] = np.stack(normalize_mean_input[var], axis=0)
            normalize_std_input[var] = np.stack(normalize_std_input[var], axis=0)
            normalize_mean_output[var] = np.stack(normalize_mean_output[var], axis=0)
            normalize_std_output[var] = np.stack(normalize_std_output[var], axis=0)

        for var in normalize_mean_input.keys():  # aggregate over the years
            # input
            mean_input, std_input = normalize_mean_input[var], normalize_std_input[var]
            variance_input = (std_input**2).mean(axis=0) + (mean_input**2).mean(axis=0) - mean_input.mean(axis=0) ** 2
            std_input = np.sqrt(variance_input)
            mean_input = mean_input.mean(axis=0)
            normalize_mean_input[var] = mean_input
            normalize_std_input[var] = std_input

            # output
            mean_output, std_output = normalize_mean_output[var], normalize_std_output[var]
            variance_output = (std_output**2).mean(axis=0) + (mean_output**2).mean(axis=0) - mean_output.mean(axis=0) ** 2
            std_output = np.sqrt(variance_output)
            mean_output = mean_output.mean(axis=0)
            normalize_mean_output[var] = mean_output
            normalize_std_output[var] = std_output

        np.savez(os.path.join(save_dir, "normalize_mean_input.npz"), **normalize_mean_input)
        np.savez(os.path.join(save_dir, "normalize_std_input.npz"), **normalize_std_input)
        np.savez(os.path.join(save_dir, "normalize_mean_output.npz"), **normalize_mean_output)
        np.savez(os.path.join(save_dir, "normalize_std_output.npz"), **normalize_std_output)


@click.command()
@click.option("--path_mpi", type=str, default='/home/data/datasets/mpi/5.625deg_algined')
@click.option("--path_era5", type=str, default='/home/data/datasets/era5/1.40625deg_algined/')
@click.option("--save_dir", type=str, default='/home/data/datasets/downscale_mpi_5.625_era5_1.40625')
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2011)
@click.option("--start_test_year", type=int, default=2013)
@click.option("--end_year", type=int, default=2015)
@click.option("--num_shards", type=int, default=32)
def main(
    path_mpi,
    path_era5,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    nc2np(path_mpi, path_era5, variables, train_years, save_dir, "train", num_shards)
    nc2np(path_mpi, path_era5, variables, val_years, save_dir, "val", num_shards)
    nc2np(path_mpi, path_era5, variables, test_years, save_dir, "test", num_shards)

    # save lat and lon data
    ps = glob.glob(os.path.join(path_era5, variables[0], f"*{train_years[0]}*.nc"))
    x = xr.open_mfdataset(ps[0], parallel=True)
    lat = x["lat"].to_numpy()
    lon = x["lon"].to_numpy()
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":
    main()