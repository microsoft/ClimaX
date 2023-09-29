import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

def extract_one_year(path, year, variables, len_to_extract, np_vars, normalize_mean, normalize_std):
    for var in variables:
        ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
        ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
        code = NAME_TO_VAR[var]
        lat = ds.lat.values
        lon = ds.lon.values

        if len(ds[code].shape) == 3:  # surface level variables
            ds[code] = ds[code].expand_dims("val", axis=1)
            
            # remove the last 24 hours if this year has 366 days
            np_vars[var] = ds[code].to_numpy()[:len_to_extract]
            if len(np_vars[var]) < len_to_extract:
                n_missing_data = len_to_extract - len(np_vars[var])
                np_vars[var] = np.concatenate((np_vars[var], np_vars[var][-n_missing_data:]), axis=0)

            var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
            var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
            if var not in normalize_mean:
                normalize_mean[var] = [var_mean_yearly]
                normalize_std[var] = [var_std_yearly]
            else:
                normalize_mean[var].append(var_mean_yearly)
                normalize_std[var].append(var_std_yearly)
        else:  # multiple-level variables, only use a subset
            assert len(ds[code].shape) == 4
            all_levels = ds["plev"][:].to_numpy() / 100  # 92500 --> 925
            all_levels = all_levels.astype(int)
            all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
            for level in all_levels:
                ds_level = ds.sel(plev=[level * 100.0])
                # level = int(level / 100) # 92500 --> 925

                # remove the last 24 hours if this year has 366 days
                np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[:len_to_extract]
                if len(np_vars[f"{var}_{level}"]) < len_to_extract:
                    n_missing_data = len_to_extract - len(np_vars[f"{var}_{level}"])
                    np_vars[f"{var}_{level}"] = np.concatenate((np_vars[f"{var}_{level}"], np_vars[f"{var}_{level}"][-n_missing_data:]), axis=0)

                var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                if var not in normalize_mean:
                    normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                    normalize_std[f"{var}_{level}"] = [var_std_yearly]
                else:
                    normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                    normalize_std[f"{var}_{level}"].append(var_std_yearly)
    
    return np_vars, normalize_mean, normalize_std, lat, lon

def aggregate_mean_std(normalize_mean, normalize_std):
    for var in normalize_mean.keys():
        normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
        normalize_std[var] = np.stack(normalize_std[var], axis=0)
        
        mean, std = normalize_mean[var], normalize_std[var]
        # var(X) = E[var(X|Y)] + var(E[X|Y])
        variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
        std = np.sqrt(variance)
        # E[X] = E[E[X|Y]]
        mean = mean.mean(axis=0)
        normalize_mean[var] = mean
        normalize_std[var] = std
    
    return normalize_mean, normalize_std

def nc2np(dataset, path, variables, years, hours_per_year, num_shards_per_year, save_dir):
    os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    normalize_mean = {}
    normalize_std = {}
    lat, lon = None, None
    
    for year in tqdm(years):
        np_vars = {}
        len_to_extract = hours_per_year
        if year == '201001010600-201501010000' and (dataset == 'hammoz' or dataset == 'tai'): # special case, only 7304 points
            len_to_extract = 7300
        else:
            len_to_extract = hours_per_year
        
        np_vars, normalize_mean, normalize_std, lat, lon = extract_one_year(
            path,
            year,
            variables,
            len_to_extract,
            np_vars,
            normalize_mean,
            normalize_std
        )
        if lat is None or lon is None:
            lat = lat
            lon = lon
        
        num_shards = num_shards_per_year
        if year == '201001010600-201501010000' and dataset == 'tai': # only 7300 points
            num_shards = num_shards // 2
        if year == '201001010600-201501010000' and dataset == 'hammoz':
            num_shards = num_shards // 4

        assert len_to_extract % num_shards == 0
        num_hrs_per_shard = len_to_extract // num_shards
        for shard_id in range(num_shards):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, "train", f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    normalize_mean, normalize_std = aggregate_mean_std(normalize_mean, normalize_std)

    np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
    np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


@click.command()
@click.option("--dataset", type=str, default='mpi')
@click.option("--path", type=click.Path(exists=True))
@click.option("--num_shards", type=int, default=10) ## recommended: 10 shards for MPI, 20 for tai, 2 for awi, 40 for hammoz, 2 for cmcc (must keep the same ratio to be able to train on multi gpus)
@click.option("--save_dir", type=click.Path(exists=False))
def main(
    dataset,
    path,
    num_shards,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    
    if dataset == 'mpi':
        hours_per_year = 7300
        year_strings = [f"{y}01010600-{y+5}01010000" for y in range(1850, 2015, 5)]
        variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
    elif dataset == 'tai':
        hours_per_year = 14600
        year_strings = [
            '185001010000-186001010000',
            '186001010600-187001010000',
            '187001010600-188001010000',
            '188001010600-189001010000',
            '189001010600-190001010000',
            '190001010600-191001010000',
            '191001010600-192001010000',
            '192001010600-193001010000',
            '193001010600-194001010000',
            '194001020000-195001010000',
            '195001010600-196001010000',
            '196001010600-197001010000',
            '197001010600-198001010000',
            '198001010600-199001010000',
            '199001010600-200001010000',
            '200001010600-201001010000',
            '201001010600-201501010000'
        ]
        variables = [
            "2m_temperature",
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
    elif dataset == 'awi':
        hours_per_year = 1460
        year_strings = [f'{y}01010600-{y+1}01010000' for y in range(1850, 2015, 1)]
        variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
    elif dataset == 'hammoz':
        hours_per_year = 29200
        year_strings = [
            '185001010600-187001010000',
            '187001010600-189001010000',
            '189001010600-191001010000',
            '191001010600-193001010000',
            '193001010600-195001010000',
            '195001010600-197001010000',
            '197001010600-199001010000',
            '199001010600-201001010000',
            '201001010600-201501010000'
        ]
        variables = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
    elif dataset == 'cmcc':
        hours_per_year = 1460
        year_strings = [f'{y}01010600-{y+1}01010000' for y in range(1850, 2015, 1)]
        variables = [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
    else:
        raise NotImplementedError(f'{dataset} is not supported')
    
    assert hours_per_year % num_shards == 0
    nc2np(
        dataset=dataset,
        path=path,
        variables=variables,
        years=year_strings,
        hours_per_year=hours_per_year,
        num_shards_per_year=num_shards,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()