import os
from pathlib import Path
import json
from typing import Any, Union, Optional
from tqdm import tqdm
import logging
import torch

from evaluation import metrics


logger = logging.getLogger('Evaluation')


def evaluate(
    model: torch.nn.Module,
    dataset: Any,
    output_path: str,
    device: Union[str, torch.device],
    visualize: bool = False,
    scale: Optional[float] = None,
    title: str = 'unknown'
) -> None:
    """
    Evaluates model on Argoverse dataset and outputs all metrics in `output_path` with optional visualizations.
    - For global metrics meanADE and meanFDE are used
    - For scenario metrics ADE and FDE are used for agent and (ADE and FDE are averaged for all other objects on scenario)

    Args:
        model: Model
        dataset: Dataset
        output_path: Evaluation output path
        device: Device
        visualize: Optionally visualize scenarios with forecasts
        scale: Scale (optional)
        title: Evaluation title
    """
    fig = None
    n_scenarios = len(dataset)
    total_agent_min_ade = 0.0
    total_agent_min_fde = 0.0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for scenario_index in tqdm(range(len(dataset))):
            scenario = dataset.scenario(scenario_index)
            scenario_metrics = {'scenario_city': scenario.city, 'scenario_id': scenario.id}

            # forecasting
            polylines, anchors, _, gt_traj = \
                scenario.inputs.to(device).unsqueeze(0), scenario.target_proposals.to(device).unsqueeze(0), \
                scenario.target_ground_truth.to(device), scenario.ground_truth_trajectory_difference.to(device)

            outputs = model(polylines, anchors)
            forecasts, targets, anchors = outputs['forecasts'][0], outputs['targets'][0], outputs['anchors'][0]
            forecasts = forecasts.cumsum(axis=1)  # transform differences to trajectory
            gt_traj = gt_traj.cumsum(axis=0)  # transform differences to trajectory
            forecasts_scaled = forecasts * scale
            gt_traj_scaled = gt_traj * scale

            # Agent evaluation
            agent_min_ade, _ = metrics.minADE(forecasts_scaled, gt_traj_scaled)
            agent_min_fde, _ = metrics.minFDE(forecasts_scaled, gt_traj_scaled)
            agent_min_ade = agent_min_ade.detach().item()
            agent_min_fde = agent_min_fde.detach().item()
            logger.debug(f'[{scenario.dirname}]: minADE={agent_min_ade:.2f}, minFDE={agent_min_fde:.2f}')

            # Deducing error class
            scenario_metrics['agent'] = {
                'minADE': agent_min_ade,
                'minFDE': agent_min_fde
            }

            # Update global agent metrics
            total_agent_min_ade += agent_min_ade
            total_agent_min_fde += agent_min_fde

            # Saving metrics
            scenario_output_path = os.path.join(output_path, scenario.dirname)
            Path(scenario_output_path).mkdir(exist_ok=True, parents=True)
            with open(os.path.join(scenario_output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
                json.dump(scenario_metrics, stream, indent=2)

            if visualize:
                # Visualization
                fig = scenario.visualize(
                    fig=fig,
                    chosen_anchors=anchors.cpu().numpy(),
                    agent_traj_forecast=forecasts.cpu().numpy(),
                    targets_prediction=targets.cpu().numpy(),
                    scale=scale
                )
                fig.savefig(os.path.join(scenario_output_path, 'scenario.png'))

        dataset_metrics = {
            'agent-mean-minADE': total_agent_min_ade / n_scenarios,
            'agent-mean-minFDE': total_agent_min_fde / n_scenarios,
        }
        logger.info(f'Metrics for "{title}"')
        logger.info(json.dumps(dataset_metrics, indent=4))
        with open(os.path.join(output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
            json.dump(dataset_metrics, stream, indent=2)
