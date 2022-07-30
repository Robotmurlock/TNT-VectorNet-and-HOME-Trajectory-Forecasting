"""
Trains Trajectory Forecaster for Heatmap Raster model using PytorchLighting trainer
"""
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


import configparser
from utils import steps
from architectures.heatmap import LightningTrajectoryForecaster
from datasets.dataset import ScenarioDatasetTorchWrapper


def run(config: configparser.GlobalConfig):
    train_input_path = os.path.join(config.global_path, config.raster.train_tf.train_input_path)
    val_input_path = os.path.join(config.global_path, config.raster.train_tf.val_input_path)
    model_storage_path = os.path.join(config.global_path, config.raster.train_tf.output_path)

    train_config = config.raster.train_tf
    train_parameters = train_config.parameters
    train_loader = DataLoader(ScenarioDatasetTorchWrapper(train_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)
    val_loader = DataLoader(ScenarioDatasetTorchWrapper(val_input_path), batch_size=train_parameters.batch_size, num_workers=train_config.n_workers)

    tnt = LightningTrajectoryForecaster(
        in_features=3,  # coords + mask
        trajectory_hist_length=config.global_parameters.trajectory_history_window_length,
        trajectory_future_length=config.global_parameters.trajectory_future_window_length,
        train_config=train_parameters
    )
    logger = TensorBoardLogger(model_storage_path, name='heatmap_trajectory_forecaster_logs')
    trainer = Trainer(
        gpus=1,
        accelerator='cuda',
        max_epochs=train_parameters.epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_storage_path,
                monitor='val_loss',
                save_last=True,
                save_top_k=1
            )
        ]
    )
    trainer.fit(
        model=tnt,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=os.path.join(model_storage_path, 'last.ckpt')
    )

    index = 0
    device = 'cuda:0'
    tnt.eval()
    tnt.to(device)
    import matplotlib.pyplot as plt
    for agent_hist, agent_gt, agent_gt_end_point in val_loader:
        agent_hist, agent_gt, agent_gt_end_point = agent_hist.to(device), agent_gt.to(device), agent_gt_end_point.to(device)
        agent_pred = tnt(agent_hist, agent_gt_end_point)
        agent_hist, agent_gt, agent_gt_end_point, agent_pred = [x.detach().cpu().numpy() for x in [agent_hist, agent_gt, agent_gt_end_point, agent_pred]]
        agent_pred = agent_pred.cumsum(axis=2)
        agent_gt = agent_gt.cumsum(axis=1)

        for i in range(agent_hist.shape[0]):
            fig = plt.figure(figsize=(10, 10))
            plt.ylim((-2.5, 2.5))
            plt.xlim((-2.5, 2.5))
            plt.plot(agent_hist[i, :, 0], agent_hist[i, :, 1], color='red')
            plt.plot(agent_pred[i, 0, :, 0], agent_pred[i, 0, :, 1], color='green')
            plt.plot(agent_gt[i, :, 0], agent_gt[i, :, 1], color='blue')
            plt.scatter(agent_gt_end_point[i, :, 0], agent_gt_end_point[i, :, 1], color='orange', s=100)
            fig.savefig(f'tmp_{index}.png')
            index += 1


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
