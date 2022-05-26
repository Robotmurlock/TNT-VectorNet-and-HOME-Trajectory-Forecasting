import os
from pathlib import Path

import configparser
from utils import steps
from datasets.vectornet_dataset import VectorNetScenarioDataset
from datasets.data_models import GraphScenarioData
from architectures.vectornet import AnchorsLoss, AnchorGenerator, TrajectoryForecaster

from torch import optim
import torch.nn as nn
import torch
import logging
from tqdm import tqdm


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')
    train_config = config.graph.train
    input_path = os.path.join(steps.SOURCE_PATH, config.graph.train.input_path)

    dataset = VectorNetScenarioDataset(input_path)

    # Models
    anchor_generator = AnchorGenerator(polyline_features=8).to(device)
    forecaster = TrajectoryForecaster(n_features=16, trajectory_length=config.global_parameters.trajectory_future_window_length).to(device)

    # Loss functions
    ag_criteria = AnchorsLoss()
    tf_criteria = nn.MSELoss()  # Todo: Use Huber loss instead

    # Optimizers
    ag_optimizer = optim.Adam(params=anchor_generator.parameters(), lr=train_config.parameters.anchor_generator_lr)
    f_optimizer = optim.Adam(params=forecaster.parameters(), lr=train_config.parameters.trajectory_forecaster_lr)
    epochs = train_config.parameters.epochs
    n_targets = train_config.parameters.n_targets

    anchor_generator.train()
    forecaster.train()
    for epoch in tqdm(range(1, epochs+1)):
        total_loss = 0.0
        total_ce_loss = 0.0
        total_mse_loss = 0.0
        total_forecast_loss = 0.0

        for data in dataset:
            ag_optimizer.zero_grad()
            f_optimizer.zero_grad()

            polylines, anchors, ground_truth, gt_traj = \
                [d.to(device) for d in data.inputs], data.target_proposals.to(device), \
                data.target_ground_truth.to(device), data.ground_truth_trajectory_difference.to(device)
            features, targets, confidences = anchor_generator(polylines, anchors)

            # anchor loss
            loss, ce_loss, mse_loss = ag_criteria(targets, confidences, ground_truth)

            # trajectory losses
            filter_indexes = torch.argsort(confidences, descending=True)[:n_targets]
            filtered_features = features[filter_indexes]
            # Using ground truth instead targets during training
            ground_truth_expanded = ground_truth.unsqueeze(0).expand(n_targets, 2)
            forecasted_trajectories = forecaster(filtered_features, ground_truth_expanded)

            # loss
            gt_traj_expanded = gt_traj.unsqueeze(0).expand(n_targets, 30, 2)
            forecast_loss = tf_criteria(forecasted_trajectories, gt_traj_expanded)
            all_loss = 0.1*loss + 1.0*forecast_loss
            all_loss.backward()
            ag_optimizer.step()
            f_optimizer.step()

            # stats
            total_loss += all_loss.detach().item()
            total_ce_loss += ce_loss.detach().item()
            total_mse_loss += mse_loss.detach().item()
            total_forecast_loss += forecast_loss.detach().item()

        logging.info(f'[Epoch-{epoch}]: loss={(total_loss / len(dataset)):.4f}, '
                     f'mse={total_mse_loss / len(dataset):.4f}, '
                     f'ce={total_ce_loss / len(dataset):.4f}, '
                     f'fmse={total_forecast_loss / len(dataset):.4f}')

    fig = None
    anchor_generator.eval()
    forecaster.eval()
    with torch.no_grad():
        for scenario in tqdm(dataset, total=len(dataset)):
            scenario: GraphScenarioData

            polylines, anchors, ground_truth, gt_traj = \
                [d.to(device) for d in scenario.inputs], scenario.target_proposals.to(device), \
                scenario.target_ground_truth.to(device), scenario.ground_truth_trajectory_difference.to(device)
            features, targets, confidences = anchor_generator(polylines, anchors)

            filter_indexes = torch.argsort(confidences, descending=True)[:n_targets]
            filtered_targets = targets[filter_indexes]
            filtered_features = features[filter_indexes]
            forecasted_trajectories = forecaster(filtered_features, filtered_targets)

            scenario_viz_path = os.path.join(steps.SOURCE_PATH, config.graph.train.output_path, 'visualization')
            Path(scenario_viz_path).mkdir(parents=True, exist_ok=True)
            fig = scenario.visualize(
                fig=fig,
                targets_prediction=filtered_targets.detach().cpu().numpy(),
                agent_traj_forecast=forecasted_trajectories.detach().cpu().numpy().cumsum(axis=1))
            fig.savefig(os.path.join(scenario_viz_path, f'{scenario.dirname}.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
