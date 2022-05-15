import os
import logging
from tqdm import tqdm
import numpy as np
from typing import List

from utils import steps, trajectories
import configparser
from datasets.data_models import ScenarioData, GraphScenarioData, ObjectType


logger = logging.getLogger('DataGraph')


def create_polyline(polyline_data: np.ndarray, object_type: ObjectType) -> np.ndarray:
    """
    Creates polyline from trajectory data
    Dimension - Polyline dimension is equal to number of polyline edges
    Features - (start_x, start_y, end_x, end_y, is_agent, is_neighbor, is_lane, is_centerline)

    Args:
        polyline_data: Trajectory
        object_type: ObjectType

    Returns: Trajectory transformed into a polyline
    """
    assert object_type in [ObjectType.AGENT, ObjectType.NEIGHBOUR, ObjectType.CANDIDATE_CENTERLINE], 'Invalid object type!'

    polyline_edges_list: List[np.ndarray] = []
    last_point = None
    for point_index in range(polyline_data.shape[0]):
        point = polyline_data[point_index]
        if point[2] == 0:
            continue
        point = point[:2]

        if last_point is not None:
            edge = np.hstack([last_point, point, object_type.one_hot])
            polyline_edges_list.append(edge)
        last_point = point

    polyline = np.stack(polyline_edges_list)
    return polyline


def create_polylines(polylines_data: np.ndarray, object_type: ObjectType) -> List[np.ndarray]:
    """
    Creates list of polylines from stacked array of polylines

    Args:
        polylines_data: Stacked trajectories
        object_type: Trajectories object type

    Returns: List of polylines
    """
    return [create_polyline(polylines_data[object_index], object_type) for object_index in range(polylines_data.shape[0])]


def create_lane_polyline(lane_feature: np.ndarray) -> np.ndarray:
    """
    Creates lane polyline

    Args:
        lane_feature: Lane initial features

    Returns: Lane Polyline
    """
    polyline_edges_list: List[np.ndarray] = []
    last_point = None
    for point_index in range(lane_feature.shape[0]):
        point = lane_feature[point_index][:2]
        if last_point is not None:
            edge = np.hstack([last_point, point, ObjectType.CENTERLINE.one_hot])
            polyline_edges_list.append(edge)
        last_point = point

    polyline = np.stack(polyline_edges_list)
    return polyline


def create_lane_polylines(lane_features: np.ndarray) -> List[np.ndarray]:
    """
    Creates lane polylines from stacked lane features

    Args:
        lane_features: Lane features (lanes with features)

    Returns: List of polylines for each lane segment
    """
    return [create_lane_polyline(lane_features[object_index]) for object_index in range(lane_features.shape[0])]


def run(config: configparser.GlobalConfig):
    """
    Converts vectorized structured data to vectorized polyline structured data 
    Args:
        config: Config
    """
    dpg_config = config.graph.data_process
    scenario_path = os.path.join(steps.SOURCE_PATH, dpg_config.input_path)
    scenario_paths = [os.path.join(scenario_path, dirname) for dirname in os.listdir(scenario_path)]
    output_path = os.path.join(steps.SOURCE_PATH, dpg_config.output_path)
    completed_scenarios = set(os.listdir(output_path) if os.path.exists(output_path) else [])

    scenarios = [ScenarioData.load(path) for path in scenario_paths]
    logger.info(f'Found {len(scenarios)} scenarios.')

    fig = None
    logger.info('Started datasets processing.')
    for scenario in tqdm(scenarios):
        scenario: ScenarioData
        if scenario.dirname in completed_scenarios:
            logger.debug(f'Already processed "{scenario.dirname}".')
            continue

        agent_polyline = create_polyline(scenario.agent_traj_hist, ObjectType.AGENT)
        neighbor_polylines = create_polylines(scenario.objects_traj_hists, ObjectType.NEIGHBOUR)
        lane_polylines = create_lane_polylines(scenario.lane_features)
        candidate_polylines = create_polylines(scenario.centerline_candidate_features, ObjectType.CANDIDATE_CENTERLINE)
        polylines = [agent_polyline] + neighbor_polylines + lane_polylines + candidate_polylines
        polylines = [trajectories.normalize_polyline(polyline, last_index=4) for polyline in polylines]

        graph_scenario = GraphScenarioData(
            id=scenario.id,
            city=scenario.city,
            center_point=scenario.center_point,
            polylines=polylines,
            agent_traj_gt=trajectories.normalize_polyline(scenario.agent_traj_gt, last_index=2),
            objects_traj_gts=trajectories.normalize_polyline(scenario.objects_traj_gts, last_index=2)
        )
        graph_scenario.save(output_path)

        if dpg_config.visualize:
            fig = graph_scenario.visualize(fig)
            fig.savefig(os.path.join(output_path, graph_scenario.dirname, 'polylines.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
