import os
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from climax.climate_projection.dataset import ClimateBenchDataset, input_for_training, load_x_y, output_for_training, split_train_val


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    return inp, out, lead_times, variables, out_variables


class ClimateBenchDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        history=10,
        list_train_simu=[
            'ssp126',
            'ssp370',
            'ssp585',
            'historical',
            'hist-GHG',
            'hist-aer'
        ],
        list_test_simu=[
            'ssp245'
        ],
        variables=[
            'CO2',
            'SO2',
            'CH4',
            'BC'
        ],
        out_variables='tas',
        train_ratio=0.9,
        batch_size: int = 128,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        # split train and val datasets
        dict_x_train_val, dict_y_train_val, lat, lon = load_x_y(os.path.join(root_dir, 'train_val'), list_train_simu, out_variables)
        self.lat, self.lon = lat, lon
        x_train_val = np.concatenate([
            input_for_training(
                dict_x_train_val[simu], skip_historical=(i<2), history=history, len_historical=165
            ) for i, simu in enumerate(dict_x_train_val.keys())
        ], axis = 0) # N, T, C, H, W
        y_train_val = np.concatenate([
            output_for_training(
                dict_y_train_val[simu], skip_historical=(i<2), history=history, len_historical=165
            ) for i, simu in enumerate(dict_y_train_val.keys())
        ], axis=0) # N, 1, H, W
        x_train, y_train, x_val, y_val = split_train_val(x_train_val, y_train_val, train_ratio)
        
        self.dataset_train = ClimateBenchDataset(
            x_train, y_train, variables, out_variables, lat, 'train'
        )
        self.dataset_val = ClimateBenchDataset(
            x_val, y_val, variables, out_variables, lat, 'val'
        )
        self.dataset_val.set_normalize(self.dataset_train.inp_transform, self.dataset_train.out_transform)

        dict_x_test, dict_y_test, _, _ = load_x_y(os.path.join(root_dir, 'test'), list_test_simu, out_variables)
        x_test = input_for_training(
            dict_x_test[list_test_simu[0]], skip_historical=True, history=history, len_historical=165
        )
        y_test = output_for_training(
            dict_y_test[list_test_simu[0]], skip_historical=True, history=history, len_historical=165
        )
        self.dataset_test = ClimateBenchDataset(
            x_test, y_test, variables, out_variables, lat, 'test'
        )
        self.dataset_test.set_normalize(self.dataset_train.inp_transform, self.dataset_train.out_transform)

    def get_lat_lon(self):
        return self.lat, self.lon

    def set_patch_size(self, p):
        self.patch_size = p

    def get_test_clim(self):
        return self.dataset_test.y_normalization

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )