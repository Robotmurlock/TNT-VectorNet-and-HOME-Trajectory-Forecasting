"""
Trains VectorNet model using PytorchLighting trainer
"""
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


import configparser
from utils import steps
from architectures.vectornet import TargetDrivenForecaster
from datasets.vectornet_dataset import VectorNetScenarioDataset


def run(config: configparser.GlobalConfig):
    """
    python3 <path>/script_train_vectornet.py --config config_path

    Args:
        config: Global Config
    """
    train_input_path = os.path.join(config.global_path, config.graph.train.train_input_path)
    val_input_path = os.path.join(config.global_path, config.graph.train.val_input_path)
    model_storage_path = os.path.join(config.global_path, config.graph.train.output_path)

    train_config = config.graph.train
    train_parameters = train_config.parameters
    train_loader = DataLoader(VectorNetScenarioDataset(train_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)
    val_loader = DataLoader(VectorNetScenarioDataset(val_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)

    tnt = TargetDrivenForecaster(
        cluster_size=config.graph.data_process.max_polyline_segments,
        trajectory_length=config.global_parameters.trajectory_future_window_length,
        polyline_features=14,
        n_targets=train_parameters.n_targets,
        n_trajectories=train_parameters.n_trajectories,
        use_traj_scoring=train_parameters.use_traj_scoring,

        train_config=train_parameters
    )
    logger = TensorBoardLogger(model_storage_path, name='vectornet_logs')
    trainer = Trainer(
        gpus=1,
        accelerator='cuda',
        max_epochs=train_parameters.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_storage_path,
                monitor='min_fde_val',
                save_last=True,
                save_top_k=1
            )
        ]
    )
    trainer.fit(
        model=tnt,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
