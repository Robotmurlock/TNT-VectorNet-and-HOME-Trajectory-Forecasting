"""
Script to train Trajectory forecaster model given trajectory end point
"""
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import config_parser
from home.architecture import LightningTrajectoryForecaster
from library.datasets import ScenarioDatasetTorchWrapper
from config_parser.utils import steps


def run(config: config_parser.GlobalConfig):
    train_input_path = os.path.join(config.global_path, config.raster.train_tf.train_input_path)
    val_input_path = os.path.join(config.global_path, config.raster.train_tf.val_input_path)
    model_name = config.raster.train_tf.model_name
    model_path = os.path.join(config.model_storage_path, 'home', 'forecaster')

    train_config = config.raster.train_tf
    train_parameters = train_config.parameters
    train_loader = DataLoader(ScenarioDatasetTorchWrapper(train_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)
    val_loader = DataLoader(ScenarioDatasetTorchWrapper(val_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)

    tnt = LightningTrajectoryForecaster(
        in_features=3,  # coords + mask
        trajectory_hist_length=config.global_parameters.trajectory_history_window_length,
        trajectory_future_length=config.global_parameters.trajectory_future_window_length,
        train_config=train_parameters,
        traj_scale=config.graph.data_process.normalization_parameter
    )
    logger = TensorBoardLogger(model_path, name=train_config.model_name)
    trainer = Trainer(
        gpus=1,
        accelerator='cuda',
        max_epochs=train_parameters.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(model_path, model_name),
                monitor='e2e/fde',
                save_last=True,
                save_top_k=1
            )
        ]
    )
    trainer.fit(
        model=tnt,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == '__main__':
    run(config_parser.GlobalConfig.load(steps.get_config_path()))
