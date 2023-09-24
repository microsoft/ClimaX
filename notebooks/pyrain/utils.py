"""
Collect data for benchmark tasks.
"""
import torch
import numpy as np
from datetime import datetime, timedelta
import yaml
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_device_hparams(hparams):
    num_gpus = torch.cuda.device_count() if hparams['gpus'] == -1 else hparams['gpus']
    if num_gpus > 0:
        hparams['batch_size'] *= num_gpus
        # hparams['num_workers'] *= num_gpus
    hparams['multi_gpu'] = num_gpus > 1


def get_vbl_name(var:str, grid: float):
    if grid == 5.625:
        if var == 'clbt':
            return 'simsat5625/clbt'
        if var == 'precipitationcal':
            return 'imerg5625/precipitationcal'
        if (var[:4] == 'ciwc') or (var[:4] == 'clwc'):
            return "era5625/" + (var.replace('-', '_') + 'hPa' if '-' in var else var)
        return "era5625/" + (var.replace('-', '_') + 'hPa' if '-' in var else var)
    else:
        if var == 'precipitationcal':
            return 'imerg140625/precipitationcal'
        if var == 'clbt':
            return 'simsat140625/clbt'
        if (var[:4] == 'ciwc') or (var[:4] == 'clwc'):
            return "era140625/" + (var.replace('-', '_') + 'hPa' if '-' in var else var)
        return "era140625/" + (var.replace('-', '_') + 'hPa' if '-' in var else var)


def get_var_name(vbl: str):
    return vbl.split('/')[1].replace(':', '-').replace('_', '-').replace('hPa', '')


def is_vbl_const(var: str):
    if var in ['lat', 'lon', 'orography', 'lsm', 'slt', 'lat2d', 'lon2d']:
        return True
    return False


def local_time_shift(longitude: float):
    return timedelta(hours=(np.mod(longitude + 180, 360) - 180) / 180 * 12)


def get_local_shift(grid, dataset):
    if grid == 5.625:
        lon2d = dataset['era5625/lon2d']
    else:
        lon = np.linspace(0, 358.59375, 256)
        lon2d = np.expand_dims(lon, axis=1).repeat(128, 1).T
    time_shift = np.vectorize(local_time_shift)(lon2d)
    return time_shift


def apply_normalization(inputs, output, categories, normalizer):
    for i, v in enumerate(categories['input']):
        if v not in ['hour', 'day', 'month']:
            inputs[:, :, i, :, :] = (inputs[:, :, i, :, :] - normalizer[v]['mean']) / normalizer[v]['std']
    
    target_v = categories['output'][0]
    # output[:, 0, :, :] = np.log(output[:, 0, :, :] / normalizer[target_v]['std'] + 1)
    output[:, 0, :, :] = (output[:, 0, :, :] - normalizer[target_v]['mean']) / normalizer[target_v]['std']
    return inputs, output


def leadtime_into_maxtrix(lead_times: list,
                    seq_len: int,
                    forecast_freq: int,
                    forecast_n_steps: int,
                    latlon: tuple):
    """
    return shape of [bsz, seq_len, forecast_n_steps, lat, lon]
    """
    bsz = len(lead_times)
    leadtime = np.zeros((bsz, seq_len, forecast_n_steps, latlon[0], latlon[1]))
    for batch_i, lt in enumerate(lead_times):
        num = int(torch.div(lt, forecast_freq, rounding_mode='floor')-1)
        leadtime[batch_i, :, num, :, :] = 1
    return leadtime



def collate_fn(x_list, hparams, normalizer, time_shift):
    """
    return 
        inputs = [bsz, seq_len, channels, lat, lon] (constants are repeated per timestep)
        output = [bsz, channels, lat, lon]
        lead_time = [bsz]
    """
    output = []
    inputs = []
    lead_times = []
    categories = hparams['categories']
    latlon = hparams['latlon']
    compute_time = [v for v in categories['input'] if v in ['hour', 'day', 'month']]
    tmp = 'input_temporal_clbt' if 'clbt-0' in categories['input_temporal'] else 'input_temporal'
    
    for sample in x_list:
        output.append(np.concatenate([sample[0]['target'][v] for v in categories['output']], 1))
        lead_times.append(int(sample[0]['__sample_modes__'].split('_')[-1]))
        inputs.append([])

        if categories['input_static']:
            inputs[-1] += [np.repeat(sample[0]['label'][v][None, :, :], hparams['seq_len'], 0) for v in categories['input_static']]

        # input_temporal
        inputs[-1] += [sample[0]['label'][v] for v in categories[tmp]]
        
        # hour, day, month
        if compute_time:
            time_scaling = {'hour': 24, 'day': 31, 'month': 12}
            timestamps = [datetime.fromtimestamp(t) for t in sample[0]['label'][categories[tmp][0]+ '__ts']]
            timestamps = np.transpose(np.tile(timestamps, (1, *latlon, 1)), (3,0,1,2))
            if time_shift is not None:
                timestamps -= time_shift
            for m in ['hour', 'day', 'month']:
                tfunc = np.vectorize(lambda t: getattr(t, m))
                inputs[-1] += [tfunc(timestamps)/ time_scaling[m]]
        
        inputs[-1] = np.concatenate(inputs[-1], 1)


    
    inputs = torch.Tensor(np.stack(inputs))
    output = torch.Tensor(np.concatenate(output))
    lead_times = torch.Tensor(lead_times).half()

    # apply normalization
    if normalizer is not None:
        inputs, output = apply_normalization(inputs, output, categories, normalizer)

    # concatenate lead times to inputs.
    one_hot_lt = leadtime_into_maxtrix(lead_times, hparams['seq_len'], hparams['forecast_freq'], hparams['forecast_n_steps'], latlon)
    one_hot_lt = torch.Tensor(one_hot_lt)
    inputs = torch.cat([inputs, one_hot_lt], 2)

    return inputs, output, lead_times