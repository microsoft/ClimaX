# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from inspect import signature

from climax.regional_forecast.datamodule import RegionalForecastDataModule
from climax.regional_forecast.module import RegionalForecastModule

# Prefer modern import path; fall back if needed
try:
    from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
except Exception:  # pragma: no cover
    from pytorch_lightning.cli import LightningCLI, SaveConfigCallback


def main():
    # Build kwargs in a way that works across Lightning versions
    cli_sig = signature(LightningCLI.__init__)
    cli_kwargs = dict(
        model_class=RegionalForecastModule,
        datamodule_class=RegionalForecastDataModule,
        seed_everything_default=42,
        # Replaces deprecated `save_config_overwrite=...`
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
        # trainer_defaults can stay as-is if you add them later
    )
    if "auto_registry" in cli_sig.parameters:
        cli_kwargs["auto_registry"] = True

    # Initialize Lightning with the model and data modules
    cli = LightningCLI(**cli_kwargs)

    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # Wire datamodule ↔ model shapes & normalization
    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    # Train & test
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
