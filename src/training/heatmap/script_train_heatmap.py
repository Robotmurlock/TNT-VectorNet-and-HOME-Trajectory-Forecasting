import os

import configparser
from utils import steps
from datasets.heatmap_dataset import HeatmapOutputRasterScenarioDatasetTorchWrapper
from architectures.heatmap.sampler import ModalitySampler
from architectures.heatmap.heatmap_proba import HeatmapModel
from architectures.heatmap.loss import PixelFocalLoss
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch import optim
import torch
import logging


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')

    dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(config, 'train')
    data_loader = DataLoader(dataset, batch_size=16, num_workers=4)
    model = HeatmapModel(
        encoder_input_shape=(9, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=20
    ).to(device)
    criteria = PixelFocalLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=15)
    epochs = 30

    model.train()
    for epoch in tqdm(range(1, epochs+1)):
        total_loss = 0.0
        n_steps = 0

        for data in data_loader:
            optimizer.zero_grad()

            raster, trajectory, da_area, true_heatmap = data['raster'].to(device), data['agent_traj_hist'].to(device), \
                data['da_area'].to(device), data['heatmap'].to(device)
            pred_heatmap = model(raster, trajectory)

            loss = criteria(pred_heatmap, true_heatmap, da_area)
            loss.backward()

            optimizer.step()
            total_loss += loss.detach().item()
            n_steps += 1

        sched.step()
        logging.info(f'[Epoch-{epoch}]: loss={total_loss:.6f}.')

    Path(os.path.join(config.global_path, 'model_storage', 'heatmap_targets')).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), os.path.join(config.global_path, 'model_storage', 'heatmap_targets', 'last.ckpt'))

    return
    fig = plt.figure(figsize=(10, 8))
    model.eval()
    result_path = os.path.join(config.global_path, 'test_data', 'result')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=4)
    sampler = ModalitySampler(n_targets=6, radius=2, device=device, swap_rc=True)

    for data in tqdm(data_loader, total=len(dataset)):
        raster, trajectory, da_area, true_heatmap = data['raster'].to(device), data['agent_traj_hist'].to(device), \
            data['da_area'].to(device), data['heatmap'].to(device)
        scenario_city, scenario_id = data['city'][0], data['id'][0]
        scenario_dirname = f'{scenario_city}_{scenario_id}'

        pred_heatmap = model(raster, trajectory)
        targets = sampler(pred_heatmap).detach().cpu().numpy()[0]
        pred_heatmap = (pred_heatmap * da_area).detach().cpu().numpy()
        pred_heatmap = pred_heatmap / pred_heatmap.max()
        true_heatmap = true_heatmap.detach().cpu().numpy()
        fig.clf()
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(pred_heatmap[0][0], cmap='YlOrBr', origin='lower', vmin=0, vmax=1)
        ax.scatter(targets[:, 0], targets[:, 1], s=100, color='red')
        ax.set_title('pred')

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(true_heatmap[0][0], cmap='YlOrBr', origin='lower', vmin=0, vmax=1)
        ax.set_title('true')
        plt.tight_layout()
        fig.savefig(os.path.join(result_path, f'heatmap_{scenario_dirname}.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
