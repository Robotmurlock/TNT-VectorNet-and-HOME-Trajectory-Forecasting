"""
TODO: Refactor
"""
import os
from pathlib import Path
import json
from typing import Any, Union, Optional
from tqdm import tqdm
import logging
import torch

from evaluation import metrics
from datasets.data_models import RasterScenarioData


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
    torch.set_printoptions(precision=2, sci_mode=False)

    fig = None
    n_scenarios = len(dataset)
    total_agent_min_ade = 0.0
    total_agent_min_fde = 0.0
    completed_scenarios = 0
    logger.info(f'Total number of scenario for evaluation: {n_scenarios}')

    model.to(device)
    model.eval()
    n_misses = 0

    with torch.no_grad():
        for scenario_index in tqdm(range(len(dataset))):
            data: RasterScenarioData = dataset[scenario_index]
            scenario_metrics = {'scenario_city': data.city, 'scenario_id': data.id}

            # forecasting
            raster, agent_traj_hist, heatmap_gt, da_area, objects_traj_hists = torch.tensor(data.raster_features, dtype=torch.float32).to(device).unsqueeze(0), \
                torch.tensor(data.agent_traj_hist, dtype=torch.float32).to(device).unsqueeze(0) / 25.0, \
                torch.tensor(data.heatmap, dtype=torch.float32).to(device).unsqueeze(0), \
                torch.tensor(data.raster_features[0, ...], dtype=torch.float32).to(device), \
                torch.tensor(data.objects_traj_hists, dtype=torch.float32).to(device).unsqueeze(0)

            outputs = model(raster, agent_traj_hist, objects_traj_hists, da_area)

            forecasts, targets, heatmap, confidences = outputs['forecasts'][0].detach().cpu(), outputs['targets'][0].detach().cpu().numpy(), \
                outputs['heatmap'][0][0].detach().cpu().numpy(), outputs['confidences'][0]

            forecasts = forecasts.cumsum(dim=1)
            gt_traj = torch.tensor(data.agent_traj_gt, dtype=torch.float32)  # transform differences to trajectory
            forecasts_scaled = forecasts * 25.0
            gt_traj_scaled = gt_traj

            # Agent evaluation
            agent_min_ade, _ = metrics.minADE(forecasts_scaled, gt_traj_scaled)
            agent_min_fde, _ = metrics.minFDE(forecasts_scaled, gt_traj_scaled)
            agent_min_ade = agent_min_ade.detach().item()
            agent_min_fde = agent_min_fde.detach().item()
            if agent_min_fde > 2.0:
                n_misses += 1

            # Deducing error class
            scenario_metrics['agent'] = {
                'minADE': agent_min_ade,
                'minFDE': agent_min_fde
            }

            # Update global agent metrics
            total_agent_min_ade += agent_min_ade
            total_agent_min_fde += agent_min_fde

            # Saving metrics
            scenario_output_path = os.path.join(output_path, data.dirname)
            Path(scenario_output_path).mkdir(exist_ok=True, parents=True)
            with open(os.path.join(scenario_output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
                json.dump(scenario_metrics, stream, indent=2)

            if visualize:
                # Visualization
                fig = data.visualize(
                    fig=fig,
                    targets=targets,
                    agent_forecast=forecasts_scaled.numpy(),
                    heatmap=heatmap,
                )
                fig.savefig(os.path.join(scenario_output_path, 'scenario.png'))

                fig = data.visualize_heatmap(targets=targets, fig=fig)
                fig.savefig(os.path.join(scenario_output_path, 'heatmap_targets.png'))

                fig = data.visualize(
                    fig=fig,
                    targets=targets,
                    agent_forecast=forecasts_scaled.numpy(),
                    heatmap=heatmap,
                    map_radius=32
                )
                fig.savefig(os.path.join(scenario_output_path, 'scenario_zoomed.png'))

            completed_scenarios += 1
            logger.debug(f'[Current-Stats]: minADE={total_agent_min_ade / completed_scenarios}')
            logger.debug(f'[Current-Stats]: minFDE={total_agent_min_fde / completed_scenarios}')
            logger.debug(f'[Current-Stats]: MissRate={n_misses / completed_scenarios}')

        dataset_metrics = {
            'agent-mean-minADE': total_agent_min_ade / n_scenarios,
            'agent-mean-minFDE': total_agent_min_fde / n_scenarios
        }
        logger.info(f'Metrics for "{title}"')
        logger.info(json.dumps(dataset_metrics, indent=4))
        with open(os.path.join(output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
            json.dump(dataset_metrics, stream, indent=2)
