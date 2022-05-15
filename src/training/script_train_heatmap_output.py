import os

import configparser
from utils import steps
from datasets.heatmap_output_dataset import HeatmapOutputRasterScenarioDatasetTorchWrapper
from architectures.heatmap_output.model import HeatmapModel, PixelFocalLoss

from torch.utils.data import DataLoader
from torch import optim
import torch
import logging


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')

    input_path = os.path.join(steps.SOURCE_PATH, config.raster.train.input_path)

    dataset = HeatmapOutputRasterScenarioDatasetTorchWrapper(input_path)
    data_loader = DataLoader(dataset, batch_size=16, num_workers=4)
    model = HeatmapModel(
        encoder_input_shape=(10, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=20
    ).to(device)
    criteria = PixelFocalLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    epochs = 100

    model.train()
    for epoch in range(1, epochs+1):
        for (raster, trajectory, da_area), heatmap in data_loader:
            optimizer.zero_grad()

            raster, trajectory, da_area, true_heatmap = raster.to(device), trajectory.to(device), da_area.to(device), heatmap.to(device)
            pred_heatmap = model(raster, trajectory)

            loss = criteria(pred_heatmap, true_heatmap, da_area)
            logging.info(f'[Epoch-{epoch}]: loss={loss.detach().item():.4f}.')
            loss.backward()

            optimizer.step()

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    model.eval()
    for (raster, trajectory, da_area), heatmap in data_loader:
        raster, trajectory, da_area, true_heatmap = raster.to(device), trajectory.to(device), da_area.to(device), heatmap.to(device)
        pred_heatmap = (model(raster, trajectory) * da_area).detach().cpu().numpy()
        true_heatmap = true_heatmap.detach().cpu().numpy()
        for index in range(pred_heatmap.shape[0]):
            fig.clf()
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(pred_heatmap[index][0], cmap='YlOrBr', origin='lower')
            ax.set_title('pred')

            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(true_heatmap[index][0], cmap='YlOrBr', origin='lower')
            ax.set_title('true')
            plt.tight_layout()
            fig.savefig(os.path.join(steps.SOURCE_PATH, 'test_data', 'result', f'heatmap_{index}.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
