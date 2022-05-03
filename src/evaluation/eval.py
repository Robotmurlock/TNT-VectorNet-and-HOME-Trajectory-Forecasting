import os
import logging
from pathlib import Path
import json

import torch

from architectures.base import BaseModel
from datasets.dataset import ScenarioDataset
from evaluation import metrics


def evaluate(model: BaseModel, dataset: ScenarioDataset, output_path: str, visualize: bool = False) -> None:
    """
    Evaluates model on Argoverse dataset and outputs all metrics in `output_path` with optional visualizations.
    - For global metrics meanADE and meanFDE are used
    - For scenario metrics ADE and FDE are used for agent and (ADE and FDE are averaged for all other objects on scenario)

    Args:
        model: Model
        dataset: Dataset
        output_path: Evaluation output path
        visualize: Optionally visualize scenarios with forecasts
    """
    fig = None
    n_scenarios = len(dataset)
    total_agent_ade = 0.0
    total_agent_fde = 0.0
    total_objects_ade = 0.0
    total_objects_fde = 0.0

    with torch.no_grad():
        for scenario in dataset:
            scenario_metrics = {'scenario_city': scenario.city, 'scenario_id': scenario.id}

            # forecasting
            agent_gt, objects_gt = scenario.ground_truth
            agent_prediction, objects_prediction = model.forecast(scenario.features)

            # Agent evaluation
            agent_ade = metrics.ADE(agent_prediction, agent_gt).item()
            agent_fde = metrics.FDE(agent_prediction, agent_gt).item()
            logging.debug(f'[Scenario={scenario.id}] - (agent evaluation): ADE={agent_ade:.2f}, FDE={agent_fde:.2f}')
            scenario_metrics['agent'] = {
                'ADE': agent_ade,
                'FDE': agent_fde
            }

            # Update global agent metrics
            total_agent_ade += agent_ade
            total_agent_fde += agent_fde

            # Objects evaluation
            # noinspection PyTypedDict
            scenario_metrics['objects'] = []
            object_scenario_total_ade = 0.0
            object_scenario_total_fde = 0.0
            n_objects = objects_prediction.shape[0]

            scenario_metrics['objects'] = {'all': []}
            for object_index in range(n_objects):
                object_prediction, object_gt = objects_prediction[object_index], objects_gt[object_index]
                object_ade = metrics.ADE(object_prediction, object_gt).item()
                object_fde = metrics.FDE(object_prediction, object_gt).item()
                logging.debug(f'[Scenario={scenario.id}] - (object {object_index} evaluation): ADE={object_ade:.2f}, FDE={object_fde:.2f}')
                scenario_metrics['objects']['all'].append({
                    'ADE': object_ade,
                    'FDE': object_fde
                })

                # Averaged metrics
                object_scenario_total_ade += object_ade
                object_scenario_total_fde += object_fde

            objects_mean_ade = object_scenario_total_ade / n_objects
            objects_mean_fde = object_scenario_total_fde / n_objects
            scenario_metrics['objects']['meanADE'] = objects_mean_ade
            scenario_metrics['objects']['meanFDE'] = objects_mean_fde

            # Update global object metrics
            total_objects_ade += objects_mean_ade
            total_objects_fde += objects_mean_fde

            # Saving metrics
            scenario_output_path = os.path.join(output_path, scenario.dirname)
            Path(scenario_output_path).mkdir(exist_ok=True, parents=True)
            with open(os.path.join(scenario_output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
                json.dump(scenario_metrics, stream, indent=2)

            if visualize:
                # Visualization
                fig = scenario.visualize(fig, agent_forecast=agent_prediction, objects_forecast=objects_prediction)
                fig.savefig(os.path.join(scenario_output_path, 'scenario.png'))

        dataset_metrics = {
            'agent-meanADE': total_agent_ade / n_scenarios,
            'agent-meanFDE': total_agent_fde / n_scenarios,
            'object-meanADE': total_agent_ade / n_scenarios,
            'object-meanFDE': total_objects_fde / n_scenarios,
        }
        dataset_metrics['weighted-meanADE'] = 0.8 * dataset_metrics['agent-meanADE'] + 0.2 * dataset_metrics['object-meanADE']
        dataset_metrics['weighted-meanFDE'] = 0.8 * dataset_metrics['agent-meanFDE'] + 0.2 * dataset_metrics['object-meanFDE']
        with open(os.path.join(output_path, 'metrics.json'), 'w', encoding='utf-8') as stream:
            json.dump(dataset_metrics, stream, indent=2)
