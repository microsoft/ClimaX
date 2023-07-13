import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

def get_mpi_str(year):
    if year < 1980:
        str = '197501010600-198001010000'
    elif year >= 1980 and year < 1985:
        str = '198001010600-198501010000'
    elif year >= 1985 and year < 1990:
        str = '198501010600-199001010000'
    elif year >= 1990 and year < 1995:
        str = '199001010600-199501010000'
    elif year >= 1995 and year < 2000:
        str = '199501010600-200001010000'
    elif year >= 2000 and year < 2005:
        str = '200001010600-200501010000'
    elif year >= 2005 and year < 2010:
        str = '200501010600-201001010000'
    elif year >= 2010 and year < 2015:
        str = '201001010600-201501010000'
    else:
        str = None
    
    return str

@click.command()
@click.option("--path_mpi", type=str, default='/home/data/datasets/mpi/5.625deg')
@click.option("--path_era5", type=str, default='/home/data/datasets/era5/1.40625deg')
@click.option("--new_path_mpi", type=str, default='/home/data/datasets/mpi/5.625deg_algined')
@click.option("--new_path_era5", type=str, default='/home/data/datasets/era5/1.40625deg_algined')
@click.option("--mpi_res", type=str, default='5.625deg')
@click.option("--era5_res", type=str, default='1.40625deg')
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
    ],
)
def main(
    path_mpi,
    path_era5,
    new_path_mpi,
    new_path_era5,
    mpi_res,
    era5_res,
    variables
):
    era5_years = list(range(1979, 2019))

    for var in variables:
        print ('Aligning', var)
        os.makedirs(os.path.join(new_path_mpi, var), exist_ok=True)
        os.makedirs(os.path.join(new_path_era5, var), exist_ok=True)
        for year in tqdm(era5_years):
            mpi_str = get_mpi_str(year)
            if mpi_str is not None:
                mpi_ds = xr.open_dataset(os.path.join(path_mpi, var, f"{var}_{mpi_str}_{mpi_res}.nc"))
                era5_ds = xr.open_dataset(os.path.join(path_era5, var, f"{var}_{year}_{era5_res}.nc"))

                time_ids = np.isin(mpi_ds.time, era5_ds.time)
                mpi_ds = mpi_ds.sel(time=time_ids)

                time_ids = np.isin(era5_ds.time, mpi_ds.time)
                era5_ds = era5_ds.sel(time=time_ids)

                mpi_ds.to_netcdf(os.path.join(new_path_mpi, var, f"{var}_{year}_{mpi_res}.nc"))
                era5_ds.to_netcdf(os.path.join(new_path_era5, var, f"{var}_{year}_{era5_res}.nc"))


if __name__ == "__main__":
    main()