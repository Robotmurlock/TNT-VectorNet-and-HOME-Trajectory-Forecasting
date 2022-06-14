import os
from pathlib import Path

import configparser
from utils import steps
from datasets.vectornet_dataset import VectorNetScenarioDataset
from architectures.vectornet import TargetsLoss, TargetGenerator, TrajectoryForecaster, ForecastingLoss

from torch import optim
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm


def run(config: configparser.GlobalConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: Move to config
    logging.info(f'Training on `{device}` device')
    train_config = config.graph.train
    input_path = os.path.join(steps.SOURCE_PATH, config.graph.train.input_path)

    dataset = VectorNetScenarioDataset(input_path, device=device)
    loader = DataLoader(dataset, batch_size=1)

    # Models
    anchor_generator = TargetGenerator(cluster_size=20, polyline_features=14, device=device).to(device)
    forecaster = TrajectoryForecaster(n_features=256, trajectory_length=config.global_parameters.trajectory_future_window_length).to(device)

    # Loss functions
    ag_criteria = TargetsLoss(delta=train_config.parameters.huber_delta)
    tf_criteria = ForecastingLoss(delta=train_config.parameters.huber_delta)

    # Optimizers
    ag_optimizer = optim.Adam(params=anchor_generator.parameters(), lr=train_config.parameters.anchor_generator_lr)
    ag_scheduler = optim.lr_scheduler.StepLR(ag_optimizer, step_size=train_config.parameters.anchor_generator_sched_step,
                                             gamma=train_config.parameters.anchor_generator_sched_gamma)
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

        for polylines, anchors, ground_truth, gt_traj in loader:
            ag_optimizer.zero_grad()
            f_optimizer.zero_grad()

            features, offsets, confidences = anchor_generator(polylines, anchors)

            # anchor loss
            loss, ag_ce_loss, ag_huber_loss = ag_criteria(anchors, offsets, confidences, ground_truth)

            # trajectory losses
            # Using ground truth instead targets during training
            ground_truth_expanded = ground_truth.unsqueeze(1).repeat(1, n_targets, 1)
            forecasted_trajectories = forecaster(features, ground_truth_expanded)

            # loss
            f_huber_loss = tf_criteria(forecasted_trajectories, gt_traj)
            all_loss = 0.1*ag_ce_loss + 0.1*ag_huber_loss + 1.0*f_huber_loss
            all_loss.backward()
            ag_optimizer.step()
            f_optimizer.step()

            # stats
            total_loss += all_loss.detach().item()
            total_ag_ce_loss += ag_ce_loss.detach().item()
            total_ag_huber_loss += ag_huber_loss.detach().item()
            total_f_huber_loss += f_huber_loss.detach().item()

        # schedulers
        ag_scheduler.step()

        logging.info(f'[Epoch-{epoch}]: weighted-loss={(total_loss / len(dataset)):.2f}, '
                     f'target-huber={total_ag_huber_loss / len(dataset):.6f}, '
                     f'target-ce={total_ag_ce_loss / len(dataset):.4f}, '
                     f'traj-huber={total_f_huber_loss / len(dataset):.6f}')

    model_path = os.path.join(steps.SOURCE_PATH, config.graph.train.output_path, 'model')
    Path(model_path).mkdir(exist_ok=True, parents=True)
    torch.save(anchor_generator.state_dict(), os.path.join(model_path, 'target_generator.pt'))
    torch.save(forecaster.state_dict(), os.path.join(model_path, 'forecaster.pt'))

    if not train_config.visualize:
        return


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
