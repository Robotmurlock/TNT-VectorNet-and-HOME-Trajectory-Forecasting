import os
from pathlib import Path
import json
from typing import Any, Union, Optional
from tqdm import tqdm
import logging
import torch

from evaluation import metrics
from datasets.data_models import GraphScenarioData


logger = logging.getLogger('Evaluation')


def evaluate(
    model: torch.nn.Module,
    loss: torch.nn.Module,
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
        loss: Loss
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
    total_total_loss = 0.0
    total_tg_ce_loss = 0.0
    total_tg_huber_loss = 0.0
    total_tf_huber_loss = 0.0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for scenario_index in tqdm(range(len(dataset))):
            scenario: GraphScenarioData = dataset[scenario_index]
            scenario_metrics = {'scenario_city': scenario.city, 'scenario_id': scenario.id}

            # forecasting
            polylines, anchors, ground_truth, gt_traj = \
                scenario.inputs.to(device).unsqueeze(0), scenario.target_proposals.to(device).unsqueeze(0), \
                scenario.target_ground_truth.to(device).unsqueeze(0), scenario.ground_truth_trajectory_difference.to(device).unsqueeze(0)

            outputs = model(polylines, anchors)
            total_loss, tg_ce_loss, tg_huber_loss, tf_huber_loss = loss(
                anchors=outputs['all_anchors'],
                offsets=outputs['all_offsets'],
                confidences=outputs['all_confidences'],
                ground_truth=ground_truth,
                forecasts=outputs['forecasts'],
                gt_traj=gt_traj)

            forecasts, targets, anchors = outputs['forecasts'][0], outputs['targets'][0], outputs['anchors'][0]
            forecasts = forecasts.cumsum(axis=1)  # transform differences to trajectory
            gt_traj = gt_traj[0].cumsum(axis=0)  # transform differences to trajectory
            forecasts_scaled = forecasts * scale
            gt_traj_scaled = gt_traj * scale

            # Agent evaluation
            agent_min_ade, _ = metrics.minADE(forecasts_scaled, gt_traj_scaled)
            agent_min_fde, _ = metrics.minFDE(forecasts_scaled, gt_traj_scaled)
            agent_min_ade = agent_min_ade.detach().item()
            agent_min_fde = agent_min_fde.detach().item()

            # Loss info
            total_loss = total_loss.detach().item()
            tg_ce_loss = tg_ce_loss.detach().item()
            tg_huber_loss = tg_huber_loss.detach().item()
            tf_huber_loss = tf_huber_loss.detach().item()
            logger.debug(f'[{scenario.dirname}]')
            logger.debug(f'\tEvaluation (end-to-end): minADE={agent_min_ade:.2f}, minFDE={agent_min_fde:.2f}')
            logger.debug(f'\tLosses (all): total_loss={total_loss:.4f}')
            logger.debug(f'\tLosses (targets): tg_ce_loss={tg_ce_loss:.4f}, tg_huber_loss={tg_huber_loss:.4f}')
            logger.debug(f'\tLosses (trajs): tf_huber_loss={tf_huber_loss:.4f}')

            # Deducing error class
            scenario_metrics['agent'] = {
                'minADE': agent_min_ade,
                'minFDE': agent_min_fde,
                'total_loss': total_loss,
                'tg_ce_loss': tg_ce_loss,
                'tg_huber_loss': tg_huber_loss,
                'tf_huber_loss': tf_huber_loss,
            }

            # Update global agent metrics
            total_agent_min_ade += agent_min_ade
            total_agent_min_fde += agent_min_fde
            total_total_loss += total_loss
            total_tg_ce_loss += tg_ce_loss
            total_tg_huber_loss += tg_huber_loss
            total_tf_huber_loss += tf_huber_loss

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
                    scale=scale,
                    visualize_anchors=True,
                    visualize_candidate_centerlines=True
                )
                fig.savefig(os.path.join(scenario_output_path, 'scenario.png'))

        dataset_metrics = {
            'agent-mean-minADE': total_agent_min_ade / n_scenarios,
            'agent-mean-minFDE': total_agent_min_fde / n_scenarios,
            'total_loss': total_total_loss / n_scenarios,
            'tg_ce_loss': total_tg_ce_loss / n_scenarios,
            'tg_huber_loss': total_tg_huber_loss / n_scenarios,
            'tf_huber_loss': total_tf_huber_loss / n_scenarios,
        }
        logger.info(f'Metrics for "{title}"')
        logger.info(json.dumps(dataset_metrics, indent=4))
        with open(os.path.join(output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
            json.dump(dataset_metrics, stream, indent=2)
