# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)
from climax.utils.data_utils import get_region_info


def collate_fn_regional(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    region_info = batch[0][5]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
        region_info,
    )


class RegionalForecastDataModule(LightningDataModule):
    """DataModule for regional forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        variables (list): List of input variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        region (str, optional): The name of the region to finetune ClimaX on.
        predict_range (int, optional): Predict range.
        hrs_each_step (int, optional): Hours each step.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        root_dir,
        variables,
        buffer_size,
        out_variables=None,
        region: str = 'NorthAmerica',
        predict_range: int = 6,
        hrs_each_step: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
        self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
        self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(out_variables)

        self.val_clim = self.get_climatology("val", out_variables)
        self.test_clim = self.get_climatology("test", out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def get_climatology(self, partition="val", variables=None):
        path = os.path.join(self.hparams.root_dir, partition, "climatology.npz")
        clim_dict = np.load(path)
        if variables is None:
            variables = self.hparams.variables
        clim = np.concatenate([clim_dict[var] for var in variables])
        clim = torch.from_numpy(clim)
        return clim

    def set_patch_size(self, p):
        self.patch_size = p

    def setup(self, stage: Optional[str] = None):
        lat, lon = self.get_lat_lon()
        region_info = get_region_info(self.hparams.region, lat, lon, self.patch_size)
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                            multi_dataset_training=False,
                        ),
                        max_predict_range=self.hparams.predict_range,
                        random_lead_time=False,
                        hrs_each_step=self.hparams.hrs_each_step,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    region_info=region_info
                ),
                buffer_size=self.hparams.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.hparams.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                region_info=region_info
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    max_predict_range=self.hparams.predict_range,
                    random_lead_time=False,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                region_info=region_info
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_regional,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_regional,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_regional,
        )
