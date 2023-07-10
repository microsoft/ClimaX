### Adapted from https://github.com/duncanwp/ClimateBench/blob/main/prep_input_data.ipynb

import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def load_x_y(data_path, list_simu, out_var):
    x_all, y_all = {}, {}
    for simu in list_simu:
        input_name = 'inputs_' + simu + '.nc'
        output_name = 'outputs_' + simu + '.nc'
        if 'hist' in simu:
            # load inputs 
            input_xr = xr.open_dataset(os.path.join(data_path, input_name))
                
            # load outputs                                                             
            output_xr = xr.open_dataset(os.path.join(data_path, output_name)).mean(dim='member')
            output_xr = output_xr.assign({
                "pr": output_xr.pr * 86400,
                "pr90": output_xr.pr90 * 86400
            }).rename({
                'lon':'longitude',
                'lat': 'latitude'
            }).transpose('time','latitude', 'longitude').drop(['quantile'])
        
        # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
        else:
            # load inputs 
            input_xr = xr.open_mfdataset([
                os.path.join(data_path, 'inputs_historical.nc'), 
                os.path.join(data_path, input_name)
            ]).compute()
                
            # load outputs                                                             
            output_xr = xr.concat([
                xr.open_dataset(os.path.join(data_path, 'outputs_historical.nc')).mean(dim='member'),
                xr.open_dataset(os.path.join(data_path, output_name)).mean(dim='member')
            ], dim='time').compute()
            output_xr = output_xr.assign({
                "pr": output_xr.pr * 86400,
                "pr90": output_xr.pr90 * 86400
            }).rename({
                'lon':'longitude',
                'lat': 'latitude'
            }).transpose('time','latitude', 'longitude').drop(['quantile'])

        print(input_xr.dims, output_xr.dims, simu)

        x = input_xr.to_array().to_numpy()
        x = x.transpose(1, 0, 2, 3).astype(np.float32) # N, C, H, W
        x_all[simu] = x

        y = output_xr[out_var].to_array().to_numpy() # 1, N, H, W
        # y = np.expand_dims(y, axis=1) # N, 1, H, W
        y = y.transpose(1, 0, 2, 3).astype(np.float32)
        y_all[simu] = y

    temp = xr.open_dataset(os.path.join(data_path, 'inputs_' + list_simu[0] + '.nc')).compute()
    if 'latitude' in temp:
        lat = np.array(temp['latitude'])
        lon = np.array(temp['longitude'])
    else:
        lat = np.array(temp['lat'])
        lon = np.array(temp['lon'])

    return x_all, y_all, lat, lon

def input_for_training(x, skip_historical, history, len_historical):
    time_length = x.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([
            x[i:i+history] for i in range(len_historical-history+1, time_length-history+1)
        ])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([x[i:i+history] for i in range(0, time_length-history+1)])
    
    return X_train_to_return

def output_for_training(y, skip_historical, history, len_historical):    
    time_length = y.shape[0]
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([
            y[i+history-1] for i in range(len_historical-history+1, time_length-history+1)
        ])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([y[i+history-1] for i in range(0, time_length-history+1)])
    
    return Y_train_to_return

def split_train_val(x, y, train_ratio=0.9):
    shuffled_ids = np.random.permutation(x.shape[0])
    train_len = int(train_ratio * x.shape[0])
    train_ids = shuffled_ids[:train_len]
    val_ids = shuffled_ids[train_len:]
    return x[train_ids], y[train_ids], x[val_ids], y[val_ids]

class ClimateBenchDataset(Dataset):
    def __init__(self, X_train_all, Y_train_all, variables, out_variables, lat, partition='train'):
        super().__init__()
        self.X_train_all = X_train_all
        self.Y_train_all = Y_train_all
        self.len_historical = 165
        self.variables = variables
        self.out_variables = out_variables
        self.lat = lat
        self.partition = partition
    
        if partition == 'train':
            self.inp_transform = self.get_normalize(self.X_train_all)
            # self.out_transform = self.get_normalize(self.Y_train_all)
            self.out_transform = transforms.Normalize(np.array([0.]), np.array([1.]))
        else:
            self.inp_transform = None
            self.out_transform = None

        if partition == 'test':
            # only use 2080 - 2100 according to ClimateBench
            self.X_train_all = self.X_train_all[-21:]
            self.Y_train_all = self.Y_train_all[-21:]
            self.get_rmse_normalization()

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 1, 3, 4))
        std = np.std(data, axis=(0, 1, 3, 4))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_rmse_normalization(self):
        y_avg = torch.from_numpy(self.Y_train_all).squeeze(1).mean(0) # H, W
        w_lat = np.cos(np.deg2rad(self.lat)) # (H,)
        w_lat = w_lat / w_lat.mean()
        w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y_avg.dtype, device=y_avg.device) # (H, 1)
        self.y_normalization = torch.abs(torch.mean(y_avg * w_lat))

    def __len__(self):
        return self.X_train_all.shape[0]

    def __getitem__(self, index):
        inp = self.inp_transform(torch.from_numpy(self.X_train_all[index]))
        out = self.out_transform(torch.from_numpy(self.Y_train_all[index]))
        # lead times = 0
        lead_times = torch.Tensor([0.0]).to(dtype=inp.dtype)
        return inp, out, lead_times, self.variables, self.out_variables
