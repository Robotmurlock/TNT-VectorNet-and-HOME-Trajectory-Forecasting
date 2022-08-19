import os

import configparser
from utils import steps
from datasets.heatmap_dataset import HeatmapOutputRasterScenarioDatasetTorchWrapper
from architectures.heatmap.heatmap_proba import HeatmapModel
from architectures.heatmap.loss import PixelFocalLoss
from pathlib import Path
from tqdm import tqdm
from typing import Any

from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn
import logging


def train_epoch(
    epoch: int,
    data_loader: DataLoader,
    model: nn.Module,
    criteria: nn.Module,
    optimizer: torch.optim.Optimizer,
    sched: Any,
    device: Any,
    global_path: str
):
    total_loss = 0.0
    n_steps = 0

    model.train()
    for data in data_loader:
        optimizer.zero_grad()

        raster, agent_hist, objects_hist, da_area, true_heatmap = data['raster'].to(device), data['agent_traj_hist'].to(device), \
            data['objects_traj_hist'].to(device), data['da_area'].to(device), data['heatmap'].to(device)
        pred_heatmap = model(raster, agent_hist, objects_hist)

        loss = criteria(pred_heatmap, true_heatmap, da_area)
        loss.backward()

        optimizer.step()
        total_loss += loss.detach().item()
        n_steps += 1

    sched.step()
    logging.info(f'[Epoch-{epoch}]: train_loss={total_loss / n_steps:.6f}.')
    Path(os.path.join(global_path, 'model_storage', 'heatmap_targets')).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), os.path.join(global_path, 'model_storage', 'heatmap_targets', f'epoch_{epoch}.ckpt'))


@torch.no_grad()
def eval_epoch(
    epoch: int,
    data_loader: DataLoader,
    model: nn.Module,
    criteria: nn.Module,
    device: Any
) -> None:
    total_loss = 0.0
    n_steps = 0

    model.eval()
    for data in data_loader:
        raster, agent_hist, objects_hist, da_area, true_heatmap = data['raster'].to(device), data['agent_traj_hist'].to(device), \
            data['objects_traj_hist'].to(device), data['da_area'].to(device), data['heatmap'].to(device)
        pred_heatmap = model(raster, agent_hist, objects_hist)

        loss = criteria(pred_heatmap, true_heatmap, da_area)

        total_loss += loss.detach().item()
        n_steps += 1

    logging.info(f'[Epoch-{epoch}]: val_loss={total_loss / n_steps:.6f}.')


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')

    parameters = config.raster.train_heatmap
    train_dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(config, 'train')
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=parameters.batch_size,
        num_workers=parameters.n_workers
    )
    val_dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(config, 'val')
    val_data_loader = DataLoader(val_dataset, batch_size=parameters.batch_size, num_workers=parameters.n_workers)

    model = HeatmapModel(
        encoder_input_shape=(9, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=config.global_parameters.trajectory_history_window_length
    ).to(device)
    criteria = PixelFocalLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=15)
    epochs = parameters.epochs

    model.train()
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(
            epoch=epoch,
            data_loader=train_data_loader,
            model=model,
            criteria=criteria,
            optimizer=optimizer,
            sched=sched,
            device=device,
            global_path=config.global_path
        )
        eval_epoch(
            epoch=epoch,
            data_loader=val_data_loader,
            model=model,
            criteria=criteria,
            device=device
        )


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
