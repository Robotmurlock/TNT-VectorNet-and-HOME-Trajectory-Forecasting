"""
Transforms raw HD map data into vectorized format
"""
import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional, Any, Union, Set, Collection

import fastdtw
import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from scipy.spatial.distance import euclidean

import config_parser
from common_data_processing import exceptions, pipeline
from library import conventions
from library.datasets.data_models.scenario import ScenarioData
from library.utils import trajectories, lists, time
from config_parser.utils import steps

logger = logging.getLogger('DataProcess')


def process_agent_data(
    df: pd.DataFrame,
    trajectory_history_window_length: int,
    trajectory_future_window_length: int,
    trajectory_min_history_window_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Extracts Agent trajectory:
        - history (features for predictions)
        - ground truth / future (datasets for evaluation)
        - center point for normalization

    Agent features are (x, y, timestamp, mask) where mask is 0 if that part of trajectory is padded.
    (timestamp is removed later but is required for syncing with neighbor objects)
    Average time difference between two observations is 0.1 second

    Args:
        df: Sequence DataFrame
        trajectory_history_window_length: History window length (parameter)
        trajectory_future_window_length: Future window length (parameter)
        trajectory_min_history_window_length: Minimum history window length (parameter)
            if history window length is too small then scenario is skipped

    Returns:
        - Agent history trajectory features
        - Ground truth trajectory
        - Agent (scenario) center point
        - Agent naive velocity approximation
        - Angle for rotation to y-axis based on last observed agent point
    """
    df_agent_traj = df[df.OBJECT_TYPE == 'AGENT']
    df_agent_traj = df_agent_traj.sort_values(by='TIMESTAMP')
    logger.debug(f'Agent full trajectory length: {df_agent_traj.shape[0]}.')

    len_threshold = trajectory_future_window_length + trajectory_min_history_window_length
    if df_agent_traj.shape[0] < len_threshold:
        msg = f'Trajectory length threshold is {len_threshold} but found length is {df_agent_traj.shape[0]}!'
        raise exceptions.AgentTrajectoryMinLengthThresholdException(msg)

    # extract future (ground truth) and history trajectory
    traj = df_agent_traj[['X', 'Y', 'TIMESTAMP']]  # timestamp is required to sync with neighboring objects
    agent_traj_gt = traj.iloc[-trajectory_future_window_length:].values
    agent_traj_hist = traj.iloc[-trajectory_future_window_length-trajectory_history_window_length:-trajectory_future_window_length].values
    base_agent_traj_hist = agent_traj_hist.copy()

    # approximate agent angle to y-axis
    direction_at_last_observed_step = agent_traj_hist[-1, :] - agent_traj_hist[0, :]
    agent_y_angle = trajectories.calc_angle_to_y_axis(direction_at_last_observed_step)
    logger.debug(f'Agent angle: {agent_y_angle}')

    # normalize trajectory by center coordinate (all coordinates are relative to agent trajectory center)
    traj_center = base_agent_traj_hist[-1, :2].copy()  # center point (all values)
    agent_traj_gt[:, :2] -= traj_center  # Third dimension is timestamp
    agent_traj_hist[:, :2] -= traj_center  # Third dimension is timestamp
    logger.debug(f'Agent center point: ({traj_center[0]}, {traj_center[1]})')

    # normalize trajectory by direction (last observed agent position is on y-axis)
    agent_traj_hist = trajectories.rotate_points(agent_traj_hist, agent_y_angle)
    agent_traj_gt = trajectories.rotate_points(agent_traj_gt, agent_y_angle)

    # add paddings to traj_hist
    agent_traj_hist, n_missing_points = trajectories.pad_trajectory(agent_traj_hist, trajectory_history_window_length,
                                                                    pad_type=trajectories.PadType.PAST)
    if n_missing_points > 0:
        logger.debug(f'Padded {n_missing_points} on agent history trajectory.')

    # approximate agent velocity
    base_agent_velocity = trajectories.approximate_trajectory_velocity(base_agent_traj_hist)
    logger.debug(f'Agent velocity: ({base_agent_velocity[0]}, {base_agent_velocity[1]})')

    # Asserts
    expected_traj_hist_shape = (trajectory_history_window_length, 4)
    assert agent_traj_hist.shape == expected_traj_hist_shape, f'Wrong shape: Expected {expected_traj_hist_shape} but found {agent_traj_hist.shape}'
    expected_traj_future_shape = (trajectory_future_window_length, 3)
    assert agent_traj_gt.shape == expected_traj_future_shape, f'Wrong shape: Expected {expected_traj_future_shape} but found {agent_traj_gt.shape}'

    return agent_traj_hist, agent_traj_gt, traj_center, base_agent_velocity, agent_y_angle, base_agent_traj_hist


def process_neighbors_data(
    df: pd.DataFrame,
    center_point: np.ndarray,
    agent_angle: float,
    agent_traj_hist: np.ndarray,
    agent_traj_gt: np.ndarray,
    base_agent_velocity: np.ndarray,
    object_distance_threshold: Optional[float],
    trajectory_history_window_length: int,
    trajectory_future_window_length: int,
    object_trajectory_min_history_window_length: int,
    object_trajectory_min_future_window_length: int
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Processes neighbors datasets (similar to process_agent_data)

    Args:
        df: Sequence DataFrame
        center_point: Agent center point
        agent_angle: Rotation angle (normalization)
        agent_traj_hist: Agent trajectory history
        agent_traj_gt: Agent trajectory ground truth
        base_agent_velocity: Approximated agent velocity
        object_distance_threshold: Filter neighbor objects using agent velocity (if not None)
            If objects are too far (compared to approximated agent velocity) then they are ignored
        trajectory_history_window_length: History window length (parameter)
        trajectory_future_window_length: Future window length (parameter)
        object_trajectory_min_history_window_length: Min history window length
            if criteria is not met then object is ignored
        object_trajectory_min_future_window_length: Min future (ground truth) window length
            if criteria is not met then object is ignored

    Returns: Stacked features
        - Trajectory history for all objects
        - Trajectory ground truth (future) for all objects
        - Object center points (used for lane segments search)
    """
    start_point_timestamp = agent_traj_hist[0, 2]  # start of agent trajectory
    center_point_timestamp = agent_traj_hist[-1, 2]  # end of agent history trajectory
    end_point_timestamps = agent_traj_gt[-1, 2]  # end of agent trajectory (end of agent future trajectory)
    logger.debug(f'Agent timestamps points: Start - {start_point_timestamp}, Center - {center_point_timestamp}, End - {end_point_timestamps}')

    df = df[df.OBJECT_TYPE == 'OTHERS']  # 'OTHERS' := neighbors
    df = df[(df.TIMESTAMP >= start_point_timestamp) | (df.TIMESTAMP <= end_point_timestamps)]  # syncing trajectory by time

    n_objects = df.TRACK_ID.nunique()
    objects_traj_hist_list = []
    objects_traj_gt_list = []
    objects_center_points = []
    for object_id, df_object in df.groupby('TRACK_ID'):
        df_object = df_object.sort_values(by='TIMESTAMP')
        logger.debug(f'Object "{object_id}" full trajectory length {df_object.shape[0]} after syncing.')

        # extract trajectory history
        df_object_hist = df_object[df_object.TIMESTAMP <= center_point_timestamp]
        object_traj_hist = df_object_hist.iloc[-trajectory_history_window_length:][['X', 'Y']].values
        if object_traj_hist.shape[0] < object_trajectory_min_history_window_length:
            logger.debug(f'Object "{object_id}" trajectory history length is {object_traj_hist.shape[0]} '
                         f'but threshold is {object_trajectory_min_history_window_length}. Ignoring this neighbor!')
            n_objects -= 1
            continue

        # extract trajectory future (ground truth)
        df_object_gt = df_object[df_object.TIMESTAMP > center_point_timestamp]
        object_traj_gt = df_object_gt[:trajectory_future_window_length][['X', 'Y']].values
        if object_traj_gt.shape[0] < object_trajectory_min_future_window_length:
            logger.debug(f'Object "{object_id}" trajectory ground truth length is {object_traj_hist.shape[0]} '
                         f'but threshold is {object_trajectory_min_history_window_length}. Ignoring this neighbor!')
            n_objects -= 1
            continue

        object_center_point = object_traj_hist[-1, :2].copy()  # mask values removed
        if object_distance_threshold is not None:
            alpha_x = np.abs(object_center_point[0] - center_point[0]) / base_agent_velocity[0]
            alpha_y = np.abs(object_center_point[1] - center_point[1]) / base_agent_velocity[1]

            if alpha_x > object_distance_threshold:
                n_objects -= 1
                logger.debug(f'Object "{object_id} is too far on x axis: {alpha_x:.2f} > {object_distance_threshold}.')
                continue

            if alpha_y > object_distance_threshold:
                n_objects -= 1
                logger.debug(f'Object "{object_id} is too far on y axis: {alpha_y:.2f} > {object_distance_threshold}.')
                continue

        # normalize trajectories position - agent last observed value is (0, 0)
        object_traj_hist -= center_point
        object_traj_gt -= center_point

        # normalize trajectories direction - agent last observed value is on y-axis
        object_traj_hist = trajectories.rotate_points(object_traj_hist, agent_angle)
        object_traj_gt = trajectories.rotate_points(object_traj_gt, agent_angle)

        # pad trajectories
        object_traj_hist, n_hist_missing = trajectories.pad_trajectory(object_traj_hist, trajectory_history_window_length,
                                                                       pad_type=trajectories.PadType.PAST)
        object_traj_gt, n_gt_missing = trajectories.pad_trajectory(object_traj_gt, trajectory_future_window_length,
                                                                   pad_type=trajectories.PadType.FUTURE)
        logger.debug(f'Object "{object_id}" trajectory history padded points is: {n_hist_missing}.')
        logger.debug(f'Object "{object_id}" trajectory ground truth padded points is: {n_gt_missing}.')

        objects_center_points.append(object_center_point)
        objects_traj_hist_list.append(object_traj_hist)
        objects_traj_gt_list.append(object_traj_gt)

    if len(objects_traj_hist_list) == 0:
        # No objects found, return empty arrays
        objects_traj_hists = np.zeros(shape=(0, trajectory_history_window_length, 3))
        objects_traj_gts = np.zeros(shape=(0, trajectory_future_window_length, 3))
        return objects_traj_hists, objects_traj_gts, objects_center_points

    objects_traj_hists = np.stack(objects_traj_hist_list)
    objects_traj_gts = np.stack(objects_traj_gt_list)

    logger.debug(f'Total number of neighbors is {n_objects}.')
    # Asserts
    expected_traj_hist_shape = (n_objects, trajectory_history_window_length, 3)
    assert objects_traj_hists.shape == expected_traj_hist_shape, \
        f'Wrong shape: Expected {expected_traj_hist_shape} but found {objects_traj_hists.shape}'
    expected_traj_future_shape = (n_objects, trajectory_future_window_length, 3)
    assert objects_traj_gts.shape == expected_traj_future_shape, \
        f'Wrong shape: Expected {expected_traj_future_shape} but found {objects_traj_gts.shape}'

    return objects_traj_hists, objects_traj_gts, objects_center_points


def encode_direction(direction: str) -> List[float]:
    """
    Encode direction as one-hot

    Args:
        direction: Direction as string

    Returns: Direction as one-hot vector
    """
    if direction == 'NONE':
        return [1, 0, 0]
    elif direction == 'RIGHT':
        return [0, 1, 0]
    elif direction == 'LEFT':
        return [0, 0, 1]
    else:
        raise ValueError(f'Invalid direction "{direction}"!')


def drop_agent_traj_timestamps(agent_traj_hist: np.ndarray, agent_traj_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop timestamps from agent features after synchronization with neighbor objects

    Args:
        agent_traj_hist: Agent trajectory history
        agent_traj_gt: Agent trajectory ground truth

    Returns: Agent trajectories without timestamps
    """
    hist_len, gt_len = agent_traj_hist.shape[0], agent_traj_gt.shape[0]

    agent_traj_hist = np.delete(agent_traj_hist, 2, axis=1)
    agent_traj_gt = np.delete(agent_traj_gt, 2, axis=1)

    # Asserts
    expected_traj_hist_shape = (hist_len, 3)
    assert agent_traj_hist.shape == expected_traj_hist_shape, f'Wrong shape: Expected {expected_traj_hist_shape} but found {agent_traj_hist.shape}'
    expected_traj_future_shape = (gt_len, 2)
    assert agent_traj_gt.shape == expected_traj_future_shape, f'Wrong shape: Expected {expected_traj_future_shape} but found {agent_traj_gt.shape}'

    return agent_traj_hist, agent_traj_gt


def process_lane_data(
    avm: ArgoverseMap,
    city: str,
    agent_center_point: np.ndarray,
    agent_angle: float,
    agent_velocity: np.ndarray,
    radius_scale: float,
    objects_center_points: List[np.ndarray],
    add_neighboring_lanes: bool
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate feature for lane datasets that are nearby to agent or some object

    Args:
        avm: ArgoverseMap API
        city: Map (Miami or Pittsburgh)
        agent_center_point: Agent center point
        agent_angle: Rotation angle (normalization)
        agent_velocity: Agent velocity is used for radius approximation
        radius_scale: Radius scale (multiplied with agent_velocity)
        objects_center_points: Object center points
        add_neighboring_lanes: Add neighboring lanes segments (not just closest ones) (parameter)

    Returns: Lane with features
        * 10 lane centerline coordinates encoded with metadata (7 values)
            - centerline x and y values (normalized)
            - is lane segment in intersection
            - is there traffic control
            - direction (none, right, left) encoded as one-hot
    """
    lane_segment_id_list: List[List[int]] = []
    center_points = [agent_center_point] + objects_center_points

    # Calculate radius
    agent_velocity_vector_length = np.sqrt(agent_velocity[0] ** 2 + agent_velocity[1] ** 2)
    radius: float = agent_velocity_vector_length * radius_scale

    for center_point in center_points:
        cx, cy = center_point

        object_lane_segment_ids = avm.get_lane_ids_in_xy_bbox(cx, cy, city, radius)
        if add_neighboring_lanes:
            successors_ls_ids_output = [avm.get_lane_segment_successor_ids(ls, city) for ls in object_lane_segment_ids]
            predecessor_ls_ids_output = [avm.get_lane_segment_predecessor_ids(ls, city) for ls in object_lane_segment_ids]
            adjacent_ls_ids_output = [avm.get_lane_segment_adjacent_ids(ls, city) for ls in object_lane_segment_ids]

            successor_lane_segment_ids = lists.flatten([ls_id for ls_id in successors_ls_ids_output if ls_id is not None])
            predecessor_lane_segment_ids = lists.flatten([ls_id for ls_id in predecessor_ls_ids_output if ls_id is not None])
            adjacent_lane_segment_ids = lists.flatten([ls_id for ls_id in adjacent_ls_ids_output if ls_id is not None])

            object_lane_segment_ids = object_lane_segment_ids + successor_lane_segment_ids + predecessor_lane_segment_ids + adjacent_lane_segment_ids
            object_lane_segment_ids = [lsi for lsi in object_lane_segment_ids if lsi is not None]  # filter `None` values
        lane_segment_id_list.append(object_lane_segment_ids)
    lane_segment_ids = lists.flatten(lane_segment_id_list)
    lane_segment_ids = list(set(lane_segment_ids))  # remove duplicates

    n_lsi = len(lane_segment_ids)
    logger.debug(f'Total number of lane segments is {n_lsi}.')

    lsi_data_list: List[np.ndarray] = []
    for lsi in lane_segment_ids:
        # extract polygon
        centerline = avm.get_lane_segment_centerline(lsi, city)[:, :-1]  # Ignoring height coordinate

        centerline -= agent_center_point  # normalize coordinates
        centerline = trajectories.rotate_points(centerline, agent_angle)  # normalize angle

        # extract metadata features
        is_intersection = avm.lane_is_in_intersection(lsi, city)
        is_traffic_control = avm.lane_has_traffic_control_measure(lsi, city)
        direction = encode_direction(avm.get_lane_turn_direction(lsi, city))
        # noinspection PyTypeChecker
        metadata = np.array([is_intersection, is_traffic_control] + direction, dtype=np.float)

        # stack all features into one vector
        tiled_metadata = np.tile(metadata, centerline.shape[0]).reshape(centerline.shape[0], metadata.shape[0])
        lsi_data = np.hstack([centerline, tiled_metadata])
        lsi_data_list.append(lsi_data)

    if len({lsi_data.shape[0] for lsi_data in lsi_data_list}) != 1:
        raise exceptions.InvalidLaneLengthSequencesException('Lane polygon do not match or missing!')
    data = np.stack(lsi_data_list)

    # Asserts
    expected_data_shape = (n_lsi, 10, 7)  # polygon shape is 21, number of features is 7
    assert data.shape == expected_data_shape, f'Wrong shape: Expected {expected_data_shape} but found {data.shape}'

    return data, lane_segment_ids


def find_and_process_centerline_features(
    avm: ArgoverseMap,
    city: str,
    lane_features: np.ndarray,
    lane_ids: List[int],
    agent_traj_hist: np.ndarray,
    base_agent_traj_hist: np.ndarray,
    center_point: np.ndarray,
    agent_angle: float,
    trajectory_future_window_length: int,
    centerline_radius_scale: float,
    min_lane_radius: float
) -> Optional[np.ndarray]:
    """
    Args:
        avm: ArgoverseMap API
        city: Map (Miami of Pittsburgh)
        lane_features: Lane segments features
        lane_ids: Lane segments ids
        agent_traj_hist: Agent trajectory
        base_agent_traj_hist: not normalized agent trajectory history (used to find candidate centerlines)
        center_point: Center point
        agent_angle: Rotation angle (normalization)
        trajectory_future_window_length: Trajectory future window length (parameter)
        centerline_radius_scale: Maximum centerline distance (parameter)
        min_lane_radius: Minimum value for lane radius

    Returns:
        - None if not centerlines are found
        - Centerlines with normalized x and y coordinates
    """
    agent_velocity = base_agent_traj_hist[-1, :2] - base_agent_traj_hist[-2, :2]  # velocity is here approximated using distance in last step
    radius = max(min_lane_radius, np.sqrt(agent_velocity[0] ** 2 + agent_velocity[1] ** 2) * centerline_radius_scale)

    # Find initial candidates using dtw with L2 distance
    dist_lane_id = []
    agent_traj_coords = agent_traj_hist[:, :2]
    for index in range(len(lane_ids)):
        lane = lane_features[index, :, :2]
        lane_id = lane_ids[index]
        dist, _ = fastdtw.dtw(lane, agent_traj_coords, dist=euclidean)
        dist_lane_id.append((dist, lane, lane_id))

    # Take closes 3 lanes as initial candidates
    dist_lane_id = sorted(dist_lane_id, key=lambda x: x[0])[:3]

    # Filter bad candidates by angle distance
    filtered_cl_ids = []
    agent_direction = agent_traj_coords[-1] - agent_traj_coords[0]
    agent_direction_angle = np.degrees(np.arctan2(agent_direction[1], agent_direction[0]))

    for _, lane, lane_id in dist_lane_id:
        lane_direction = lane[-1] - lane[0]
        lane_direction_angle = np.degrees(np.arctan2(lane_direction[1], lane_direction[0]))

        diff_angle = np.abs(agent_direction_angle - lane_direction_angle)
        if diff_angle > 90:
            # If angle between agent trajectory and candidate is more than 90 degrees, drop the candidate
            continue

        filtered_cl_ids.append(lane_id)

    # Run dfs to get successor of given candidates (the closest lane ids)
    centerline_ids = \
        list(set(lists.flatten([lists.flatten(avm.dfs(lsi, city_name=city, threshold=radius)) for lsi in filtered_cl_ids])))
    centerlines = [avm.get_lane_segment_centerline(lsi, city_name=city)[:, :-1] for lsi in centerline_ids]
    n_centerlines = len(centerlines)

    if n_centerlines == 0:
        raise exceptions.NoCandidateCenterlinesWereFoundException('No centerline candidates were found!')

    n_found_centerlines = n_centerlines
    logger.debug(f'Found {n_found_centerlines} candidate centerlines.')

    # normalize centerline trajectories (direction and position)
    centerlines = [(c - center_point) for c in centerlines]
    centerlines = [trajectories.rotate_points(c, agent_angle) for c in centerlines]

    # pad centerline trajectories
    centerlines = [c[:trajectory_future_window_length] for c in centerlines]
    # pad_trajectory returns tuple (padded_trajectory, n_missing_points)
    centerlines = [trajectories.pad_trajectory(c, trajectory_future_window_length, trajectories.PadType.FUTURE)[0] for c in centerlines]

    centerlines = np.stack(centerlines)

    # Asserts
    expected_centerlines_shape = (n_found_centerlines, trajectory_future_window_length, 3)
    assert centerlines.shape == expected_centerlines_shape, f'Wrong shape: Expected {expected_centerlines_shape} but found {centerlines.shape}'

    return centerlines


def calculate_point_variance(
    agent_traj_hist: np.ndarray,
    objects_traj_hists: np.ndarray,
    lane_features: np.ndarray,
    centerline_candidate_features: np.ndarray
):
    """
    Calculates variance of distances of all points on scenario

    Args:
        agent_traj_hist: Agent Trajectory History
        objects_traj_hists: Objects Trajectory History
        lane_features: Lane segments
        centerline_candidate_features: Candidate lanes

    Returns: Variance of distances of all points on scenario
    """
    agent_points = agent_traj_hist[:, :2]
    objects_points = objects_traj_hists[:, :, :2].reshape(-1, 2)
    lane_points = lane_features[:, :, :2].reshape(-1, 2)
    centerline_points = centerline_candidate_features[:, :, :2].reshape(-1, 2)
    points = np.vstack([agent_points, objects_points, lane_points, centerline_points])
    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return np.var(distances)


class ArgoverseHDPipeline(pipeline.Pipeline):
    """
    Allows running HD map processing in parallel
    """
    def __init__(
        self,
        output_path: str,
        config: config_parser.GlobalConfig,
        argoverse_map: ArgoverseMap,
        completed_sequences: Optional[Union[Set[str], List[str]]] = None,
        visualize: bool = False
    ):
        """
        Args:
            output_path: Output path
            config: Configs
            argoverse_map: Argoverse Map API
            completed_sequences: List of completed sequences (to be skipped)
            visualize: Visualize data (or not)
        """
        super().__init__(output_path=output_path, visualize=visualize)

        self._avm = argoverse_map
        self._config = config
        self._parameters = config.data_process.parameters

        self._completed_sequences = completed_sequences
        if self._completed_sequences is None:
            self._completed_sequences = []
        self._completed_sequences = set(self._completed_sequences)

    def process(self, data: Any) -> Any:
        seq_df, sequence_name, city = data
        sequence_fullname = f'{city}_{sequence_name}'
        if sequence_fullname in self._completed_sequences:
            logger.debug(f'Sequence "{sequence_fullname}" already processed!')
            return

        logger.debug(f'Processing sequence "{sequence_fullname}"...')

        try:
            # Extract features
            agent_traj_hist, agent_traj_gt, center_point, base_agent_velocity, agent_y_angle, base_agent_traj_hist = process_agent_data(
                df=seq_df,
                trajectory_history_window_length=self._config.global_parameters.trajectory_history_window_length,
                trajectory_future_window_length=self._config.global_parameters.trajectory_future_window_length,
                trajectory_min_history_window_length=self._parameters.trajectory_min_history_window_length
            )

            objects_traj_hists, objects_traj_gts, objects_center_points = process_neighbors_data(
                df=seq_df,
                center_point=center_point,
                agent_angle=agent_y_angle,
                agent_traj_hist=agent_traj_hist,
                agent_traj_gt=agent_traj_gt,
                base_agent_velocity=base_agent_velocity,
                object_distance_threshold=self._parameters.object_distance_threshold,
                trajectory_history_window_length=self._config.global_parameters.trajectory_history_window_length,
                trajectory_future_window_length=self._config.global_parameters.trajectory_future_window_length,
                object_trajectory_min_history_window_length=self._parameters.object_trajectory_min_history_window_length,
                object_trajectory_min_future_window_length=self._parameters.object_trajectory_min_future_window_length
            )

            agent_traj_hist, agent_traj_gt = drop_agent_traj_timestamps(agent_traj_hist, agent_traj_gt)

            lane_features, lane_ids = process_lane_data(
                avm=self._avm,
                city=city,
                agent_center_point=center_point,
                agent_angle=agent_y_angle,
                agent_velocity=base_agent_velocity,
                radius_scale=self._parameters.lane_radius_scale,
                objects_center_points=objects_center_points,
                add_neighboring_lanes=self._parameters.add_neighboring_lanes
            )

            centerline_candidate_features = find_and_process_centerline_features(
                avm=self._avm,
                city=city,
                lane_features=lane_features,
                lane_ids=lane_ids,
                agent_traj_hist=agent_traj_hist,
                base_agent_traj_hist=base_agent_traj_hist,
                center_point=center_point,
                agent_angle=agent_y_angle,
                trajectory_future_window_length=self._config.global_parameters.trajectory_future_window_length,
                centerline_radius_scale=self._parameters.centerline_radius_scale,
                min_lane_radius=self._parameters.min_lane_radius
            )

            # Update variance stats
            variance = calculate_point_variance(agent_traj_hist, objects_traj_hists, lane_features, centerline_candidate_features)
            logger.debug(f'Variance: {variance:.2f}')

            scenario = ScenarioData(
                id=sequence_name,
                city=city,
                center_point=center_point,
                angle=agent_y_angle,
                agent_traj_hist=agent_traj_hist,
                agent_traj_gt=agent_traj_gt,
                objects_traj_hists=objects_traj_hists,
                objects_traj_gts=objects_traj_gts,
                lane_features=lane_features,
                centerline_candidate_features=centerline_candidate_features
            )

            return scenario

        except exceptions.DataProcessException as e:
            logger.warning(f'Skipped "{sequence_name}" sequence! Error: "{e}"')

    def save(self, data: ScenarioData) -> None:
        data.save(self._output_path)

    def visualize(self, data: Any) -> None:
        self._fig = data.visualize(self._fig)
        figpath = os.path.join(self._output_path, data.dirname, 'scenario.png')
        self._fig.savefig(figpath)


class ArgoverseForecastingLoaderWrapper:
    def __init__(self, avfl: ArgoverseForecastingLoader):
        self._avfl = avfl

    def __iter__(self):
        for data in self._avfl:
            sequence_name = os.path.basename(data.current_seq).split('.')[0]  # `path/.../1234.csv` -> `1234`
            yield data.seq_df, sequence_name, data.city

    def __len__(self):
        return len(self._avfl)


@time.timeit
def run(config: config_parser.GlobalConfig):
    """
    Processes rad HD map using ArgoVerse API

    Args:
        config: Configuration
    """
    avm = ArgoverseMap()
    datasets_path = os.path.join(config.global_path, config.data_process.input_path)
    outputs_path = os.path.join(config.global_path, config.data_process.output_path)

    assert set(conventions.SPLIT_NAMES).issubset(set(os.listdir(datasets_path))), f'Format is not valid. Required splits: {conventions.SPLIT_NAMES}'

    for split_name in conventions.SPLIT_NAMES:
        if config.data_process.skip is not None and split_name in config.data_process.skip:
            logger.info(f'Skipping "{split_name}" as defined in config!')
            continue

        logger.info(f'Processing split: {split_name}')
        ds_path = os.path.join(datasets_path, split_name)
        output_path = os.path.join(outputs_path, split_name)

        # noinspection PyTypeChecker
        avfl_dataloader: Collection = ArgoverseForecastingLoaderWrapper(ArgoverseForecastingLoader(ds_path))

        Path(output_path).mkdir(parents=True, exist_ok=True)
        completed_sequences = set(os.listdir(output_path))

        hd_pipeline = ArgoverseHDPipeline(
            output_path=output_path,
            config=config,
            argoverse_map=avm,
            completed_sequences=completed_sequences,
            visualize=config.data_process.visualize
        )
        pipeline.run_pipeline(pipeline=hd_pipeline, data_iterator=avfl_dataloader, n_processes=config.data_process.n_processes)


if __name__ == '__main__':
    run(config_parser.GlobalConfig.load(steps.get_config_path()))
