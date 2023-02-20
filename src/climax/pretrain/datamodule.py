# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from climax.pretrain.dataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
    )


class MultiSourceDataModule(LightningDataModule):
    """DataModule for multi-source data.

    Args:
        dict_root_dirs (Dict): Dictionary of root directories for each source.
        dict_start_idx (Dict): Dictionary of start indices ratio (between 0.0 and 1.0) for each source.
        dict_end_idx (Dict): Dictionary of end indices ratio (between 0.0 and 1.0) for each source.
        dict_buffer_sizes (Dict): Dictionary of buffer sizes for each source.
        dict_in_variables (Dict): Dictionary of input variables for each source.
        dict_out_variables (Dict): Dictionary of output variables for each source.
        dict_max_predict_ranges (Dict, optional): Dictionary of maximum predict ranges for each source.
        dict_random_lead_time (Dict, optional): Dictionary of whether to use random lead time for each source.
        dict_hrs_each_step (Dict, optional): Dictionary of hours each step for each source.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        dict_root_dirs: Dict,
        dict_start_idx: Dict,
        dict_end_idx: Dict,
        dict_buffer_sizes: Dict,
        dict_in_variables: Dict,
        dict_out_variables: Dict,
        dict_max_predict_ranges: Dict = {"mpi-esm": 28},
        dict_random_lead_time: Dict = {"mpi-esm": True},
        dict_hrs_each_step: Dict = {"mpi-esm": 6},
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        out_variables = {}
        for k, list_out in dict_out_variables.items():
            if list_out is not None:
                out_variables[k] = list_out
            else:
                out_variables[k] = dict_in_variables[k]
        self.hparams.dict_out_variables = out_variables

        self.dict_lister_trains = {
            k: list(dp.iter.FileLister(os.path.join(root_dir, "train"))) for k, root_dir in dict_root_dirs.items()
        }
        self.train_dataset_args = {
            k: {
                "max_predict_range": dict_max_predict_ranges[k],
                "random_lead_time": dict_random_lead_time[k],
                "hrs_each_step": dict_hrs_each_step[k],
            }
            for k in dict_root_dirs.keys()
        }

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(self.hparams.dict_out_variables)

        self.dict_data_train: Optional[Dict] = None

    def get_normalize(self, dict_variables: Optional[Dict] = None):
        if dict_variables is None:
            dict_variables = self.hparams.dict_in_variables
        dict_transforms = {}
        for k in dict_variables.keys():
            root_dir = self.hparams.dict_root_dirs[k]
            variables = dict_variables[k]
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if var != "total_precipitation":
                    mean.append(normalize_mean[var])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            normalize_std = np.concatenate([normalize_std[var] for var in variables])
            dict_transforms[k] = transforms.Normalize(normalize_mean, normalize_std)
        return dict_transforms

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lat.npy"))
        lon = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.dict_data_train:
            dict_data_train = {}
            for k in self.dict_lister_trains.keys():
                lister_train = self.dict_lister_trains[k]
                start_idx = self.hparams.dict_start_idx[k]
                end_idx = self.hparams.dict_end_idx[k]
                variables = self.hparams.dict_in_variables[k]
                out_variables = self.hparams.dict_out_variables[k]
                max_predict_range = self.hparams.dict_max_predict_ranges[k]
                random_lead_time = self.hparams.dict_random_lead_time[k]
                hrs_each_step = self.hparams.dict_hrs_each_step[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.hparams.dict_buffer_sizes[k]
                dict_data_train[k] = ShuffleIterableDataset(
                    IndividualForecastDataIter(
                        Forecast(
                            NpyReader(
                                lister_train,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                out_variables=out_variables,
                                shuffle=True,
                                multi_dataset_training=True,
                            ),
                            max_predict_range=max_predict_range,
                            random_lead_time=random_lead_time,
                            hrs_each_step=hrs_each_step,
                        ),
                        transforms,
                        output_transforms,
                    ),
                    buffer_size,
                )
            self.dict_data_train = dict_data_train

    def train_dataloader(self):
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
        else:
            node_rank = int(os.environ["NODE_RANK"])
            # assert that number of datasets is the same as number of nodes
            num_nodes = os.environ.get("NODES", None)
            if num_nodes is not None:
                num_nodes = int(num_nodes)
                assert num_nodes == len(self.dict_data_train.keys())

            for idx, k in enumerate(self.dict_data_train.keys()):
                if idx == node_rank:
                    data_train = self.dict_data_train[k]
                    break

        # This assumes that the number of datapoints are going to be the same for all datasets
        return DataLoader(
            data_train,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
