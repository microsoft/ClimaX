"""
Collect data for benchmark tasks.
"""
import json
import numpy as np
from datetime import datetime
from .utils import is_vbl_const, get_var_name, get_vbl_name



def read_normalization_stats(path):
    """Read json file storing normalization statistics."""
    with open(path) as f:
        tmp = json.load(f)
    n_dict = {}
    for vbl in tmp:
        n_dict[get_var_name(vbl)] = tmp[vbl]
    return n_dict


def write_partition_conf(sources: str, imerg: bool):
    """
    Write a time partition configuration dictionary.
    """
    if sources in ['simsat', 'simsat_era', 'era16_3']:
        train_timerange = (datetime(2016,4,1,0).timestamp(), datetime(2017, 12, 31,23).timestamp())
        sample_stride = 3
    
    elif sources == 'era':
        if imerg:
            train_timerange = (datetime(2000,6,1,0).timestamp(), datetime(2017, 12,31,23).timestamp())
        else:
            train_timerange = (datetime(1979,1,1,7).timestamp(), datetime(2017, 12,31,23).timestamp())
        sample_stride = 1

    val_timerange =  (datetime(2018,1,6,0).timestamp(), datetime(2018, 12,31,23).timestamp())
    test_timerange = (datetime(2019,1,6,0).timestamp(), datetime(2019, 12, 31, 17).timestamp())
    # train_timerange = (datetime(2016,4,1,0).timestamp(), datetime(2016, 4, 15,23).timestamp())
    # val_timerange =  (datetime(2018,1,6,0).timestamp(), datetime(2018, 1,15,23).timestamp())
    # test_timerange = (datetime(2019,1,6,0).timestamp(), datetime(2019, 1, 15, 23).timestamp())

    increments = int(sample_stride * 60 * 60)

    partition_conf = {
                        "train":
                            {"timerange": train_timerange,
                            "increment_s": increments},
                        "valid":
                            {"timerange": val_timerange,
                            "increment_s": increments},
                        "test":
                            {"timerange": test_timerange,
                            "increment_s": increments}
                    }
    return partition_conf


def write_sample_conf(
        categories: dict,
        history: list,
        lead_times: list,
        interporlation: str = "nearest_past",
        grid: float = 5.625):
    """
    Write a sample configuration dictionary.
    """
    sample_conf = {}

    if 'clbt-0' in categories['input']:
        samples = {}
        for var in categories['input']:
            if is_vbl_const(var):
                samples[var] = {"vbl": get_vbl_name(var, grid)}
            elif var not in ['hour', 'day', 'month', 'clbt-1', 'clbt-2', 'clbt-0']:
                samples[var]  =  {"vbl": get_vbl_name(var, grid), "t": history, "interpolate": interporlation}
            elif var == 'clbt-0':
                samples['clbt']  = {"vbl": get_vbl_name('clbt', grid), "t": history, "interpolate": interporlation}
    else:
        samples = {var: {"vbl": get_vbl_name(var, grid)} if is_vbl_const(var) else \
            {"vbl": get_vbl_name(var, grid), "t": history, "interpolate": interporlation} \
            for var in categories['input'] if var not in ['hour', 'day', 'month']}

    for lt in lead_times:
        sample_conf["lead_time_{}".format(int(lt/3600))] = {
            "label": samples,
            "target": {var: {"vbl": get_vbl_name(var, grid), "t": np.array([lt]), "interpolate": interporlation} \
                for var in categories['output']}
            }
    
    return sample_conf


def define_categories(sources: str, inc_time: bool, imerg: bool):
    """
    Write a dictionary which holds lists specifying the model input / output variables.
    """
    simsat_vars_list = ['clbt-0', 'clbt-1', 'clbt-2'] if 'simsat' in sources else []
    era_vars_list = ['sp', 't2m', 'z-300', 'z-500', 'z-850', 't-300', 't-500', 't-850', \
        'q-300', 'q-500', 'q-850', 'clwc-300', 'clwc-500', 'ciwc-500', 'clwc-850', 'ciwc-850'] if 'era' in sources else []
    simsat_vars_list = ['clbt-0', 'clbt-1', 'clbt-2'] if 'simsat' in sources else []
    simsat_vars_list_clbt = ['clbt'] if 'simsat' in sources else []
    input_temporal = simsat_vars_list + era_vars_list
    input_temporal_clbt = simsat_vars_list_clbt + era_vars_list
    constants = ['lsm','orography', 'lat2d', 'lon2d', 'slt']
    
    inputs =  constants + input_temporal + (['hour', 'day', 'month'] if inc_time else [])
    # inputs = input_temporal + (['hour', 'day', 'month'] if inc_time else []) + constants
    output = ['precipitationcal'] if imerg else ['tp']

    
    categories = {
                'input': inputs,
                'input_temporal': input_temporal,
                'input_temporal_clbt': input_temporal_clbt,
                'input_static': constants,
                'output': output}

    return categories


def write_data_config(hparams: dict):
    """
    Define configurations for collecting data.
    """
    hparams['latlon'] = (32, 64) if hparams['grid'] == 5.625 else (128, 256)

    # define paths
    datapath = hparams['data_paths']

    # define data configurations
    categories = define_categories(hparams['sources'], inc_time=hparams['inc_time'], imerg=hparams['imerg'])
    history = np.flip(np.arange(0, hparams['sample_time_window'] + hparams['sample_freq'], hparams['sample_freq']) * -1 * 3600)
    lead_times = np.arange(hparams['forecast_freq'], hparams['forecast_time_window'] + hparams['forecast_freq'], hparams['forecast_freq']) * 3600
    partition_conf = write_partition_conf(hparams['sources'], hparams['imerg'])
    sample_conf = write_sample_conf(categories, history, lead_times, grid=hparams['grid'])

    # define new parameters in hparams
    hparams['categories'] = categories
    hparams['seq_len'] = len(history)
    hparams['forecast_n_steps'] = len(lead_times)
    hparams['out_channels'] = len(categories['output'])
    hparams['num_channels'] = len(categories['input']) + hparams['forecast_n_steps']
    hparams['lead_times'] = lead_times // 3600
    return datapath, partition_conf, sample_conf
