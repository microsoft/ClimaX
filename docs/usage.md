# Usage

## Pretraining

### Data Preparation

The code for downloading and preprocessing CMIP6 data is coming soon

### Training

```
python src/climax/pretrain/train.py --config <path/to/config>
```
For example, to pretrain ClimaX on MPI-ESM dataset on 8 GPUs use
```bash
python src/climax/pretrain/train.py --config configs/pretrain_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=8 \
    --trainer.max_epochs=100 \
    --data.batch_size=16 \
    --model.lr=5e-4 --model.beta_1="0.9" --model.beta_2="0.95" \
    --model.weight_decay=1e-5
```

!!! tip
    Make sure to update the paths of the data directories in the config files (or override them via the CLI).

### Pretrained checkpoints
We provide two pretrained checkpoints, one was pretrained on [5.625deg](https://climaxrelease.blob.core.windows.net/checkpoints/ClimaX-5.625deg.ckpt) data, and the other was pretrained on [1.40625deg](https://climaxrelease.blob.core.windows.net/checkpoints/ClimaX-1.40625deg.ckpt) data. Both checkpoints were pretrained using all 5 CMIP6 datasets.

**Usage:** We can load the checkpoint by passing the checkpoint url to the training script. See below for examples.

## Global Forecasting

### Data Preparation

First, download ERA5 data from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895). The data directory should look like the following
```
5.625deg
   |-- 10m_u_component_of_wind
   |-- 10m_v_component_of_wind
   |-- 2m_temperature
   |-- constants.nc
   |-- geopotential
   |-- relative_humidity
   |-- specific_humidity
   |-- temperature
   |-- toa_incident_solar_radiation
   |-- total_precipitation
   |-- u_component_of_wind
   |-- v_component_of_wind
```

Then, preprocess the netcdf data into small numpy files and compute important statistics
```bash
python src/data_preprocessing/nc2np_equally_era5.py \
    --root_dir /mnt/data/5.625deg \
    --save_dir /mnt/data/5.625deg_npz \
    --start_train_year 1979 --start_val_year 2016 \
    --start_test_year 2017 --end_year 2019 --num_shards 8
```

The preprocessed data directory will look like the following
```
5.625deg_npz
   |-- train
   |-- val
   |-- test
   |-- normalize_mean.npz
   |-- normalize_std.npz
   |-- lat.npy
   |-- lon.npy
```

### Training

To finetune ClimaX for global forecasting, use
```
python src/climax/global_forecast/train.py --config <path/to/config>
```
For example, to finetune ClimaX on 8 GPUs use
```bash
python src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=8 \
    --trainer.max_epochs=50 \
    --data.root_dir=/mnt/data/5.625deg_npz \
    --data.predict_range=72 --data.out_variables=['z_500','t_850','t2m'] \
    --data.batch_size=16 \
    --model.pretrained_path='https://climaxrelease.blob.core.windows.net/checkpoints/ClimaX-5.625deg.ckpt' \
    --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
```
To train ClimaX from scratch, set `--model.pretrained_path=""`.

## Regional Forecasting

### Data Preparation

We use the same ERA5 data as in global forecasting and extract the regional data on the fly during training. If you have already downloaded and preprocessed the data, you do not have to do it again.

### Training

To finetune ClimaX for regional forecasting, use
```
python src/climax/regional_forecast/train.py --config <path/to/config>
```
For example, to finetune ClimaX on North America using 8 GPUs, use
```bash
python src/climax/regional_forecast/train.py --config configs/regional_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=8 \
    --trainer.max_epochs=50 \
    --data.root_dir=/mnt/data/5.625deg_npz \
    --data.region="NorthAmerica"
    --data.predict_range=72 --data.out_variables=['z_500','t_850','t2m'] \
    --data.batch_size=16 \
    --model.pretrained_path='https://climaxrelease.blob.core.windows.net/checkpoints/ClimaX-5.625deg.ckpt' \
    --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5
```
To train ClimaX from scratch, set `--model.pretrained_path=""`.

## Visualization

Coming soon
