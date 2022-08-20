"""
Trains VectorNet model using PytorchLighting trainer
"""
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import logging


import configparser
from utils import steps
from architectures.vectornet import TargetDrivenForecaster
from datasets.vectornet_dataset import VectorNetScenarioDataset


logger = logging.getLogger('TrainVectornet')


def run(config: configparser.GlobalConfig):
    """
    python3 <path>/script_train_vectornet.py --config config_path

    Args:
        config: Global Config
    """
    train_input_path = os.path.join(config.global_path, config.graph.train.train_input_path)
    val_input_path = os.path.join(config.global_path, config.graph.train.val_input_path)

    train_config = config.graph.train
    train_parameters = train_config.parameters
    vectornet_model_storage_path = os.path.join(config.model_storage_path, 'vectornet')
    model_storage_path = os.path.join(vectornet_model_storage_path, train_config.model_name)
    checkpoint_path = os.path.join(model_storage_path, train_config.starting_checkpoint_name)

    # Config Summary
    logger.info(train_config)
    logger.info(f'Global path: "{config.global_path}"')
    logger.info(f'Train dataset path: "{train_input_path}".')
    logger.info(f'Validation dataset path: "{val_input_path}".')
    logger.info(f'Model storage path: "{model_storage_path}".')

    train_loader = DataLoader(VectorNetScenarioDataset(train_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)
    val_loader = DataLoader(VectorNetScenarioDataset(val_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)

    tnt = TargetDrivenForecaster(
        cluster_size=config.graph.data_process.max_polyline_segments,
        trajectory_length=config.global_parameters.trajectory_future_window_length,
        polyline_features=14,
        n_targets=train_parameters.n_targets,
        n_trajectories=train_parameters.n_trajectories,
        use_traj_scoring=train_parameters.use_traj_scoring,
        traj_scale=config.graph.data_process.normalization_parameter,

        train_config=train_parameters
    )
    tb_logger = TensorBoardLogger(vectornet_model_storage_path, name=train_config.model_name)
    trainer = Trainer(
        gpus=1,
        accelerator='cuda',
        max_epochs=train_parameters.epochs,
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
        model=tnt,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
