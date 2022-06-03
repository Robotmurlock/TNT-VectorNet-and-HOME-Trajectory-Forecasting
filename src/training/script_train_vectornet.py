import os
from pathlib import Path

import configparser
from utils import steps
from datasets.vectornet_dataset import VectorNetScenarioDataset
from datasets.data_models import GraphScenarioData
from architectures.vectornet import TargetsLoss, TargetGenerator, TrajectoryForecaster, ForecastingLoss

from torch import optim
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
    anchor_generator = TargetGenerator(polyline_features=9, device=device).to(device)
    forecaster = TrajectoryForecaster(n_features=16, trajectory_length=config.global_parameters.trajectory_future_window_length).to(device)

    # Loss functions
    ag_criteria = TargetsLoss(delta=train_config.parameters.huber_delta)
    tf_criteria = ForecastingLoss(delta=train_config.parameters.huber_delta)

    # Optimizers
    ag_optimizer = optim.Adam(params=anchor_generator.parameters(), lr=train_config.parameters.anchor_generator_lr)
    f_optimizer = optim.Adam(params=forecaster.parameters(), lr=train_config.parameters.trajectory_forecaster_lr)
    epochs = train_config.parameters.epochs
    n_targets = train_config.parameters.n_targets

    anchor_generator.train()
    forecaster.train()
    for epoch in tqdm(range(1, epochs+1)):
        total_loss = 0.0
        total_ag_ce_loss = 0.0
        total_ag_huber_loss = 0.0
        total_f_huber_loss = 0.0

        for data in dataset:
            ag_optimizer.zero_grad()
            f_optimizer.zero_grad()

            polylines, anchors, ground_truth, gt_traj = \
                data.inputs.to(device), data.target_proposals.to(device), \
                data.target_ground_truth.to(device), data.ground_truth_trajectory_difference.to(device)
            features, targets, confidences = anchor_generator(polylines, anchors)

            # anchor loss
            loss, ag_ce_loss, ag_huber_loss = ag_criteria(targets, confidences, ground_truth)

            # trajectory losses
            filter_indexes = torch.argsort(confidences, descending=True)[:n_targets]
            filtered_features = features[filter_indexes]
            # Using ground truth instead targets during training
            ground_truth_expanded = ground_truth.unsqueeze(0).expand(n_targets, 2)
            forecasted_trajectories = forecaster(filtered_features, ground_truth_expanded)

            # loss
            f_huber_loss = tf_criteria(forecasted_trajectories, gt_traj)
            all_loss = 0.3*ag_ce_loss + 0.2*ag_huber_loss + 1.0*f_huber_loss
            all_loss.backward()
            ag_optimizer.step()
            f_optimizer.step()

            # stats
            total_loss += all_loss.detach().item()
            total_ag_ce_loss += ag_ce_loss.detach().item()
            total_ag_huber_loss += ag_huber_loss.detach().item()
            total_f_huber_loss += f_huber_loss.detach().item()

        logging.info(f'[Epoch-{epoch}]: weighted-loss={(total_loss / len(dataset)):.2f}, '
                     f'target-huber={total_ag_huber_loss / len(dataset):.6f}, '
                     f'target-ce={total_ag_ce_loss / len(dataset):.2f}, '
                     f'traj-huber={total_f_huber_loss / len(dataset):.6f}')

    model_path = os.path.join(steps.SOURCE_PATH, config.graph.train.output_path, 'model')
    Path(model_path).mkdir(exist_ok=True, parents=True)
    torch.save(anchor_generator.state_dict(), os.path.join(model_path, 'target_generator.pt'))
    torch.save(forecaster.state_dict(), os.path.join(model_path, 'forecaster.pt'))

    fig = None
    anchor_generator.eval()
    forecaster.eval()
    with torch.no_grad():
        for scenario in tqdm(dataset, total=len(dataset)):
            scenario: GraphScenarioData

            polylines, anchors, ground_truth, gt_traj = \
                scenario.inputs.to(device), scenario.target_proposals.to(device), \
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
