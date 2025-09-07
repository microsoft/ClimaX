
Steps:

```
brew install cmake protobuf rust python@3.10 git wget
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors
```


https://dataserv.ub.tum.de/s/?dir=/5.625deg/2m_temperature


```
python predict_only_climax.py \
  --ckpt ./data-ch/5.625deg.ckpt \   
  --res 5.625 \
  --device mps
```


```
python demo_compare_resolutions.py \
  --ckpt_5625 ./data-ch/5.625deg.ckpt \
  --ckpt_1406 ./data-ch/1.40625deg.ckpt \
  --device mps \
  --out_vars geopotential_500 temperature_850 2m_temperature
```

Train:
```
python src/climax/regional_forecast/train.py \
  --config configs/regional_forecast_climax.yaml \
  --trainer.callbacks=null \
  --trainer.accelerator=mps --trainer.devices=1 --trainer.precision=32-true \
  --data.root_dir=/mnt/data/5.625deg_npz \
  --data.region="NorthAmerica" \
  --data.predict_range=72 \
  --data.out_variables="['z_500','t_850','t2m']" \
  --data.batch_size=1 --data.num_workers=0 \
  --trainer.max_epochs=1 \
  --model.pretrained_path="./data-ch/1.40625deg.ckpt" \
  --model.lr=5e-7 --model.beta_1=0.9 --model.beta_2=0.99 \
  --model.weight_decay=1e-5
```

Deps:
```
absl-py==2.3.1
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiosignal==1.4.0
antlr4-python3-runtime==4.9.3
asciitree==0.3.3
async-timeout==5.0.1
attrs==25.3.0
cdsapi==0.7.6
certifi==2025.8.3
cftime==1.6.4.post1
charset-normalizer==3.4.3
-e git+https://github.com/microsoft/ClimaX@6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c#egg=ClimaX
contourpy==1.3.2
cycler==0.12.1
docstring_parser==0.17.0
ecmwf-datastores-client==0.4.0
einops==0.8.1
fasteners==0.20
filelock==3.19.1
fonttools==4.59.2
frozenlist==1.7.0
fsspec==2024.12.0
grpcio==1.74.0
hf-xet==1.1.9
huggingface-hub==0.34.4
hydra-core==1.3.2
idna==3.10
importlib_resources==6.5.2
Jinja2==3.1.6
jsonargparse==4.41.0
kiwisolver==1.4.9
lightning-utilities==0.15.2
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.2
matplotlib==3.10.6
mdurl==0.1.2
mpmath==1.3.0
multidict==6.6.4
multiurl==0.3.7
netCDF4==1.7.2
networkx==3.4.2
numcodecs==0.13.1
numpy==1.26.4
omegaconf==2.3.0
packaging==24.2
pillow==11.3.0
propcache==0.3.2
protobuf==6.32.0
Pygments==2.19.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytorch-lightning==2.1.4
pytz==2025.2
PyYAML==6.0.2
requests==2.32.5
rich==14.1.0
safetensors==0.6.2
scipy==1.15.3
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
timm==0.6.13
torch==2.2.2
torchaudio==2.2.2
torchdata==0.7.1
torchmetrics==1.8.2
torchvision==0.17.2
tqdm==4.67.1
typeshed_client==2.8.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
xarray==2025.6.1
yarl==1.20.1
zarr==2.18.3

```
