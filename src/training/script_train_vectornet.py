import os
from pathlib import Path

import configparser
from utils import steps
from datasets.vectornet_dataset import VectorNetScenarioDataset
from architectures.vectornet.model import VectorNet

from torch import optim
import torch.nn as nn
import torch
import logging
from tqdm import tqdm


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')

    input_path = os.path.join(steps.SOURCE_PATH, config.graph.train.input_path)

    dataset = VectorNetScenarioDataset(input_path)
    model = VectorNet(polyline_features=8, trajectory_length=30).to(device)
    criteria = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    epochs = 150

    model.train()
    for epoch in tqdm(range(1, epochs+1)):
        total_loss = 0.0
        for data in dataset:
            optimizer.zero_grad()

            polylines, ground_truth = [d.to(device) for d in data.inputs], data.ground_truth.to(device)
            forecast = model(polylines)

            loss = criteria(ground_truth, forecast)
            total_loss += loss.detach().item()
            loss.backward()

            optimizer.step()

        logging.info(f'[Epoch-{epoch}]: loss={(total_loss / len(dataset)):.4f}.')

    fig = None
    model.eval()
    with torch.no_grad():
        for scenario in tqdm(dataset, total=len(dataset)):
            polylines, ground_truth = [d.to(device) for d in scenario.inputs], scenario.ground_truth.to(device)
            forecast = model(polylines).detach().cpu().numpy()

            scenario_viz_path = os.path.join(steps.SOURCE_PATH, config.graph.train.output_path, 'visualization')
            Path(scenario_viz_path).mkdir(parents=True, exist_ok=True)
            fig = scenario.visualize(fig=fig, agent_traj_forecast=forecast)
            fig.savefig(os.path.join(scenario_viz_path, f'{scenario.dirname}.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
