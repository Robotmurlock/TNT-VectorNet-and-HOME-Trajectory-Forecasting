import os
import logging
import numpy as np
from typing import List
import random
import math

from utils import steps, trajectories, time
import configparser
from datasets.data_models import ScenarioData, GraphScenarioData, ObjectType
import conventions
from data_processing import pipeline


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
            direction = point - last_point
            center_point = (point + last_point) / 2
            edge = np.hstack([center_point, direction, object_type.one_hot, empty_metadata])
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
            direction = point - last_point
            center_point = (point + last_point) / 2
            edge = np.hstack([center_point, direction, ObjectType.CENTERLINE.one_hot, metadata])
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


def sample_anchor_points(candidate_polylines: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Sample anchor points from lane points values and agent points values (positions)

    Args:
        candidate_polylines: Candidate Polylines
        sample_size: Number of points to be sampled

    Returns: Anchors (targets)
    """
    anchors_samples = []
    candidate_polylines = np.unique(candidate_polylines, axis=0)

    n_candidate_polylines = candidate_polylines.shape[0]
    points_per_candidate = [sample_size // n_candidate_polylines for _ in range(n_candidate_polylines-1)]
    points_per_candidate.append(sample_size - sum(points_per_candidate))

    for cp_index, points_to_sample in enumerate(points_per_candidate):
        polyline_length = candidate_polylines.shape[1]
        ts = np.linspace(0, polyline_length-1, points_to_sample)
        for t in ts:
            start_point_index, end_point_index = int(math.floor(t)), int(math.ceil(t))
            direction = candidate_polylines[cp_index, end_point_index, :] - candidate_polylines[cp_index, start_point_index, :]
            t_point = candidate_polylines[cp_index, start_point_index, :] + (t - start_point_index) * direction
            anchors_samples.append(t_point)

    anchors = np.vstack(anchors_samples)
    while anchors.shape != np.unique(anchors, axis=0).shape:
        anchors = np.unique(anchors, axis=0)
        n_missing = sample_size - anchors.shape[0]
        anchors = np.vstack([anchors, 10 * np.random.randn(n_missing, 2)])

    assert (sample_size, 2) == anchors.shape, f'Expected shape {(sample_size, 2)} but found {anchors.shape}'
    assert anchors.shape == np.unique(anchors, axis=0).shape, 'Found duplicates! '
    return anchors


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


class GraphPipeline(pipeline.Pipeline):
    def __init__(
        self,
        output_path: str,
        config: configparser.GlobalConfig,
        visualize: bool = False
    ):
        super().__init__(output_path=output_path, visualize=visualize)
        self._config = config
        self._dpg_config = config.graph.data_process
        self._test = 0

    def process(self, data: str) -> GraphScenarioData:
        scenario = ScenarioData.load(data)
        # create polylines
        agent_polyline = create_polyline(
            polyline_data=scenario.agent_traj_hist,
            object_type=ObjectType.AGENT,
            max_segments=self._dpg_config.max_polyline_segments)

        neighbor_polylines = create_polylines(
            polylines_data=scenario.objects_traj_hists,
            object_type=ObjectType.NEIGHBOUR,
            max_segments=self._dpg_config.max_polyline_segments)

        lane_polylines = create_lane_polylines(
            lane_features=scenario.lane_features,
            max_segments=self._dpg_config.max_polyline_segments)

        candidate_polylines = create_polylines(
            polylines_data=scenario.centerline_candidate_features,
            object_type=ObjectType.CANDIDATE_CENTERLINE,
            max_segments=self._dpg_config.max_polyline_segments)

        polylines = [agent_polyline] + neighbor_polylines + lane_polylines + candidate_polylines

        # Pad all polylines to dimension (20, 9) where last dimension is mask
        polylines = [trajectories.pad_trajectory(p, self._dpg_config.max_polyline_segments, trajectories.PadType.PAST)[0]
                     for p in polylines]

        # create anchors
        candidate_polylines = [trajectories.pad_trajectory(cp, length=20, pad_type=trajectories.PadType.PAST)[0]
                               for cp in candidate_polylines]
        candidate_polylines_points = np.stack(candidate_polylines)[:, :, :2].copy()
        anchors = sample_anchor_points(candidate_polylines_points, sample_size=50)

        # filter polylines
        polylines = polylines[:self._dpg_config.max_polylines]

        # normalize polylines and anchors
        polylines = [trajectories.normalize_polyline(polyline, last_index=4,
                                                     sigma=self._dpg_config.normalization_parameter) for polyline in polylines]

        polylines = np.stack(polylines)  # convert to numpy

        # stack polylines
        missing_polylines = max(0, self._dpg_config.max_polylines - polylines.shape[0])
        polylines = np.vstack([polylines, np.zeros([missing_polylines, *polylines.shape[1:]])])

        agent_traj_gt_normalized = trajectories.normalize_polyline(scenario.agent_traj_gt, last_index=2,
                                                                   sigma=self._dpg_config.normalization_parameter)
        objects_traj_gts_normalized = trajectories.normalize_polyline(scenario.objects_traj_gts, last_index=2,
                                                                      sigma=self._dpg_config.normalization_parameter)
        anchors = trajectories.normalize_polyline(anchors, last_index=2, sigma=self._dpg_config.normalization_parameter)

        # Anchor quality analysis
        anchor_error = anchor_min_error(anchors, agent_traj_gt_normalized[-1, :])
        logger.debug(f'[{scenario.dirname}]: Closest anchor distance is: {anchor_error:.2f}')

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

        return graph_scenario

    def save(self, data: GraphScenarioData) -> None:
        data.save(self._output_path)

    def visualize(self, data: GraphScenarioData) -> None:
        self._fig = data.visualize(self._fig, visualize_anchors=self._config.graph.data_process.visualize_anchors)
        figpath = os.path.join(self._output_path, data.dirname, 'polylines.png')
        self._fig.savefig(figpath)


@time.timeit
def run(config: configparser.GlobalConfig):
    """
    Converts vectorized structured data to vectorized polyline structured data
    Args:
        config: Config
    """
    dpg_config = config.graph.data_process
    inputs_path = os.path.join(steps.SOURCE_PATH, dpg_config.input_path)
    outputs_path = os.path.join(steps.SOURCE_PATH, dpg_config.output_path)

    assert set(conventions.SPLIT_NAMES).issubset(set(os.listdir(inputs_path))), f'Format is not valid. Required splits: {conventions.SPLIT_NAMES}'

    for split_name in conventions.SPLIT_NAMES:
        ds_path = os.path.join(inputs_path, split_name)
        output_path = os.path.join(outputs_path, split_name)

        scenario_paths = [os.path.join(ds_path, dirname) for dirname in os.listdir(ds_path)]
        completed_scenarios = set(os.listdir(output_path) if os.path.exists(output_path) else [])
        scenario_paths = list(set(scenario_paths) - set(completed_scenarios))

        logger.info(f'Found {len(scenario_paths)} scenarios for split {split_name}.')

        graph_pipeline = GraphPipeline(
            output_path=output_path,
            config=config,
            visualize=dpg_config.visualize
        )
        pipeline.run_pipeline(pipeline=graph_pipeline, data_iterator=scenario_paths, n_processes=config.data_process.n_processes)

        logger.info(f'Started datasets processing for {split_name}.')


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
