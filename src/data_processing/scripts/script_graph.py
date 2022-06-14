import os
import logging
from tqdm import tqdm
import numpy as np
from typing import List
import random

from utils import steps, trajectories
import configparser
from datasets.data_models import ScenarioData, GraphScenarioData, ObjectType


logger = logging.getLogger('DataGraph')
random.seed(42)
np.random.seed(42)


def create_polyline(polyline_data: np.ndarray, object_type: ObjectType, max_segments: int) -> np.ndarray:
    """
    Creates polyline from trajectory data
    Dimension - Polyline dimension is equal to number of polyline edges
    Features - (start_x, start_y, end_x, end_y, is_agent, is_neighbor, is_lane, is_centerline)

    Args:
        polyline_data: Trajectory
        object_type: ObjectType
        max_segments: Max polyline segments

    Returns: Trajectory transformed into a polyline
    """
    assert object_type in [ObjectType.AGENT, ObjectType.NEIGHBOUR, ObjectType.CANDIDATE_CENTERLINE], 'Invalid object type!'

    polyline_edges_list: List[np.ndarray] = []
    last_point = None
    for point_index in range(polyline_data.shape[0]):
        point = polyline_data[point_index]
        empty_metadata = np.zeros(5)  # metadata is only for lanes
        if point[2] == 0:
            continue
        point = point[:2]

        if last_point is not None:
            edge = np.hstack([last_point, point, object_type.one_hot, empty_metadata])
            polyline_edges_list.append(edge)
        last_point = point

    polyline = np.stack(polyline_edges_list)
    return polyline[-max_segments:, :]


def create_polylines(polylines_data: np.ndarray, object_type: ObjectType, max_segments: int) -> List[np.ndarray]:
    """
    Creates list of polylines from stacked array of polylines

    Args:
        polylines_data: Stacked trajectories
        object_type: Trajectories object type
        max_segments: Max polyline segmets

    Returns: List of polylines
    """
    return [create_polyline(polylines_data[object_index], object_type, max_segments)
            for object_index in range(polylines_data.shape[0])]


def create_lane_polyline(lane_feature: np.ndarray, max_segments: int) -> np.ndarray:
    """
    Creates lane polyline

    Args:
        lane_feature: Lane initial features
        max_segments: Max polyline segmets

    Returns: Lane Polyline
    """
    polyline_edges_list: List[np.ndarray] = []
    last_point = None
    for point_index in range(lane_feature.shape[0]):
        point = lane_feature[point_index][:2]
        metadata = lane_feature[point_index][2:]
        if last_point is not None:
            edge = np.hstack([last_point, point, ObjectType.CENTERLINE.one_hot, metadata])
            polyline_edges_list.append(edge)
        last_point = point

    polyline = np.stack(polyline_edges_list)
    return polyline[-max_segments:, :]


def create_lane_polylines(lane_features: np.ndarray, max_segments: int) -> List[np.ndarray]:
    """
    Creates lane polylines from stacked lane features

    Args:
        lane_features: Lane features (lanes with features)
        max_segments: Max polyline segmets

    Returns: List of polylines for each lane segment
    """
    return [create_lane_polyline(lane_features[object_index], max_segments)
            for object_index in range(lane_features.shape[0])]


def sample_anchor_points(lane_points: np.ndarray, sample_size: int, threshold: float) -> np.ndarray:
    """
    Sample anchor points from lane points values and agent points values (positions)

    Args:
        lane_points: All lane points
        sample_size: TODO
        threshold: TODO

    Returns: Anchors (targets)
    """
    n_points = lane_points.shape[0]
    np.random.shuffle(lane_points)
    assert n_points >= sample_size, 'Not enough points!'
    sampled_point_indices = []

    next_point_index = 0
    while len(sampled_point_indices) < sample_size and next_point_index < n_points:
        point = lane_points[next_point_index, :]
        skip = False
        for sp_index in sampled_point_indices:
            if np.sqrt((point[0] - lane_points[sp_index, 0]) ** 2 + (point[1] - lane_points[sp_index, 1]) ** 2) <= threshold:
                skip = True
                break

        if not skip:
            sampled_point_indices.append(next_point_index)
        next_point_index += 1

    if len(sampled_point_indices) < sample_size:
        missing_indices = sample_size - len(sampled_point_indices)
        all_indices = list(range(n_points))
        unused_indices = list(set(all_indices) - set(sampled_point_indices))
        chosen_unused_indices = random.choices(unused_indices, k=missing_indices)
        sampled_point_indices.extend(chosen_unused_indices)

    return np.stack(lane_points[sampled_point_indices])


def anchor_min_error(anchors: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    Finds the closest anchor to ground truth (used for anchor sampling evaluation)

    Args:
        anchors: Set of anchors
        ground_truth: Ground truth points

    Returns: Evaluates anchor sampling
    """
    distances = np.sqrt((anchors[:, 0] - ground_truth[0]) ** 2 + (anchors[:, 1] - ground_truth[1]) ** 2)
    return np.min(distances)


def polyline_distance_from_center(polyline: np.ndarray) -> np.ndarray:
    """
    Calculates distance of all polyline from center (agent)

    Args:
        polyline: Polyline features

    Returns: Distance from center point
    """
    return np.mean(np.sqrt(polyline[:, 0] ** 2 + polyline[:, 1] ** 2))


def run(config: configparser.GlobalConfig):
    """
    Converts vectorized structured data to vectorized polyline structured data
    Args:
        config: Config
    """
    dpg_config = config.graph.data_process
    sigma = dpg_config.normalization_parameter

    scenario_path = os.path.join(steps.SOURCE_PATH, dpg_config.input_path)
    scenario_paths = [os.path.join(scenario_path, dirname) for dirname in os.listdir(scenario_path)]
    output_path = os.path.join(steps.SOURCE_PATH, dpg_config.output_path)
    completed_scenarios = set(os.listdir(output_path) if os.path.exists(output_path) else [])

    logger.info(f'Found {len(scenario_paths)} scenarios.')

    fig = None
    logger.info('Started datasets processing.')
    total_anchor_error = 0.0
    total_scenarios = 0

    for scenario_path in tqdm(scenario_paths):
        scenario = ScenarioData.load(scenario_path)
        if scenario.dirname in completed_scenarios:
            logger.debug(f'Already processed "{scenario.dirname}".')
            continue

        # create polylines
        agent_polyline = create_polyline(
            polyline_data=scenario.agent_traj_hist,
            object_type=ObjectType.AGENT,
            max_segments=dpg_config.max_polyline_segments)

        neighbor_polylines = create_polylines(
            polylines_data=scenario.objects_traj_hists,
            object_type=ObjectType.NEIGHBOUR,
            max_segments=dpg_config.max_polyline_segments)

        lane_polylines = create_lane_polylines(
            lane_features=scenario.lane_features,
            max_segments=dpg_config.max_polyline_segments)

        candidate_polylines = create_polylines(
            polylines_data=scenario.centerline_candidate_features,
            object_type=ObjectType.CANDIDATE_CENTERLINE,
            max_segments=dpg_config.max_polyline_segments)

        polylines = [agent_polyline] + neighbor_polylines + lane_polylines + candidate_polylines

        # Pad all polylines to dimension (20, 9) where last dimension is mask
        polylines = [trajectories.pad_trajectory(p, dpg_config.max_polyline_segments, trajectories.PadType.PAST)[0]
                     for p in polylines]

        # create anchors
        anchors = sample_anchor_points(np.vstack(candidate_polylines)[:, :2].copy(), sample_size=120, threshold=0.2)

        # filter polylines
        polylines = polylines[:dpg_config.max_polylines]

        # normalize polylines and anchors
        polylines = [trajectories.normalize_polyline(polyline, last_index=4, sigma=sigma) for polyline in polylines]

        polylines = np.stack(polylines)  # convert to numpy

        # stack polylines
        missing_polylines = max(0, dpg_config.max_polylines - polylines.shape[0])
        polylines = np.vstack([polylines, np.zeros([missing_polylines, *polylines.shape[1:]])])

        agent_traj_gt_normalized = trajectories.normalize_polyline(scenario.agent_traj_gt, last_index=2, sigma=sigma)
        objects_traj_gts_normalized = trajectories.normalize_polyline(scenario.objects_traj_gts, last_index=2, sigma=sigma)
        anchors = trajectories.normalize_polyline(anchors, last_index=2, sigma=sigma)

        # Anchor quality analysis
        anchor_error = anchor_min_error(anchors, agent_traj_gt_normalized[-1, :])
        logger.debug(f'[{scenario.dirname}]: Closest anchor distance is: {anchor_error:.2f}')
        total_anchor_error += anchor_error
        total_scenarios += 1

        graph_scenario = GraphScenarioData(
            id=scenario.id,
            city=scenario.city,
            center_point=scenario.center_point,
            polylines=polylines,
            agent_traj_gt=agent_traj_gt_normalized,
            objects_traj_gts=objects_traj_gts_normalized,
            anchors=anchors,
            ground_truth_point=agent_traj_gt_normalized[-1, :]
        )
        graph_scenario.save(output_path)

        if dpg_config.visualize:
            fig = graph_scenario.visualize(fig)
            fig.savefig(os.path.join(output_path, graph_scenario.dirname, 'polylines.png'))

    if total_scenarios > 0:
        logger.info(f'Average anchor error is: {total_anchor_error / total_scenarios:.2f}')
    else:
        logger.info('No scenarios processed.')


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
