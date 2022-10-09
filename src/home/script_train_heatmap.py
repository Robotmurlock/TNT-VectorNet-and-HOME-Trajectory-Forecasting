"""
Script to train heatmap estimation model
"""
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import configparser
from datasets.heatmap_dataset import HeatmapOutputRasterScenarioDatasetTorchWrapper
from home.architecture import LightningHeatmapModel
from configparser.utils import steps


def run(config: configparser.GlobalConfig):
    train_config = config.raster.train_heatmap
    parameters = train_config.parameters

    vectornet_model_storage_path = os.path.join(config.model_storage_path, 'home', 'heatmap_targets')
    model_storage_path = os.path.join(vectornet_model_storage_path, train_config.model_name)
    checkpoint_path = os.path.join(model_storage_path, train_config.starting_checkpoint_name)

    train_dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(config, 'train')
    train_loader = DataLoader(train_dataset, batch_size=parameters.batch_size, num_workers=train_config.n_workers)
    val_dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(config, 'val')
    val_loader = DataLoader(val_dataset, batch_size=parameters.batch_size, num_workers=train_config.n_workers)

    model = LightningHeatmapModel(
        encoder_input_shape=(9, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=config.global_parameters.trajectory_history_window_length,
        sampler_radius=parameters.sampler_radius,
        sampler_targets=parameters.sampler_targets,
        base_lr=parameters.base_lr,
        sched_step=parameters.sched_step,
        sched_gamma=parameters.sched_gamma
    )

    tb_logger = TensorBoardLogger(vectornet_model_storage_path, name=train_config.model_name)
    trainer = Trainer(
        gpus=1,
        accelerator='cuda',
        max_epochs=parameters.epochs,
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_storage_path,
                monitor='e2e/min_fde',
                save_last=True,
                save_top_k=1
            )
        ],
        resume_from_checkpoint=checkpoint_path if train_config.resume and os.path.exists(checkpoint_path) else None
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    run(configparser.GlobalConfig.load(steps.get_config_path()))
