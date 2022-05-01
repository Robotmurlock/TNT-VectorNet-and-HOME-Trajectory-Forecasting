import os

import numpy as np
import pandas as pd

from utils import steps, trajectories, lists
from data_processing import exceptions
from data_processing.data import ScenarioData

import configparser
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from typing import Tuple, List, Optional


import logging


logger = logging.getLogger('DataProcess')


def approximate_agent_speed(agent_traj_hist: np.ndarray) -> np.ndarray:
    """
    Approximates agent speed from history trajectory

    Args:
        agent_traj_hist: Agent history trajectory

    Returns: Agent speed
    """
    next_obs = agent_traj_hist[1:, :2]
    prev_obs = agent_traj_hist[:-1, :2]
    diffs = next_obs - prev_obs
    speed = np.abs(np.sum(diffs, axis=0) / agent_traj_hist[:, 3].sum())  # Ignoring masked values

    assert speed.shape == (2,), f'Wrong shape: Expetced {(2,)} but found {speed.shape}'
    return speed


def process_agent_data(
    df: pd.DataFrame,
    trajectory_history_window_length: int,
    trajectory_future_window_length: int,
    trajectory_min_history_window_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts Agent trajectory:
        - history (features for predictions)
        - ground truth / future (data for evaluation)
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
        - Agent naive speed approximation
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

    # normalize trajectory by center coordite (all coordinates are relative to agent trajectory center)
    traj_center = agent_traj_hist[-1, :2].copy()  # center point (all values)
    agent_traj_gt[:, :2] -= traj_center  # Third dimension is timestamp
    agent_traj_hist[:, :2] -= traj_center  # Third dimension is timestamp
    logger.debug(f'Agent center point: ({traj_center[0]}, {traj_center[1]})')

    # add paddings to traj_hist
    agent_traj_hist, n_missing_points = trajectories.pad_trajectory(agent_traj_hist, trajectory_history_window_length,
                                                                    pad_type=trajectories.PadType.PAST)
    if n_missing_points > 0:
        logger.debug(f'Padded {n_missing_points} on agent history trajectory.')

    # approximate agent speed
    agent_speed = approximate_agent_speed(agent_traj_hist)
    logger.debug(f'Agent speed: ({agent_speed[0]}, {agent_speed[1]})')

    # Asserts
    expected_traj_hist_shape = (trajectory_history_window_length, 4)
    assert agent_traj_hist.shape == expected_traj_hist_shape, f'Wrong shape: Expected {expected_traj_hist_shape} but found {agent_traj_hist.shape}'
    expected_traj_future_shape = (trajectory_future_window_length, 3)
    assert agent_traj_gt.shape == expected_traj_future_shape, f'Wrong shape: Expetced {expected_traj_future_shape} but found {agent_traj_gt.shape}'

    return agent_traj_hist, agent_traj_gt, traj_center, agent_speed


def process_neighbors_data(
    df: pd.DataFrame,
    center_point: np.ndarray,
    agent_traj_hist: np.ndarray,
    agent_traj_gt: np.ndarray,
    agent_speed: np.ndarray,
    object_distance_threshold: Optional[float],
    trajectory_history_window_length: int,
    trajectory_future_window_length: int,
    object_trajectory_min_history_window_length: int,
    object_trajectory_min_future_window_length: int
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Processes neighbors data (similliar to process_agent_data)

    Args:
        df: Sequence DataFrame
        center_point: Agent center point
        agent_traj_hist: Agent trajectory history
        agent_traj_gt: Agent trajectory ground truth
        agent_speed: Approximated agent speed
        object_distance_threshold: Filter neighbor objects using agent speed (if not None)
            If objects are too far (compared to approximated agent speed) then they are ignored
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

        # normalize trajectories
        object_traj_hist -= center_point
        object_traj_gt -= center_point

        # pad trajectories
        object_traj_hist, n_hist_missing = trajectories.pad_trajectory(object_traj_hist, trajectory_history_window_length,
                                                                       pad_type=trajectories.PadType.PAST)
        object_traj_gt, n_gt_missing = trajectories.pad_trajectory(object_traj_gt, trajectory_future_window_length,
                                                                   pad_type=trajectories.PadType.FUTURE)
        logger.debug(f'Object "{object_id}" trajectory history padded points is: {n_hist_missing}.')
        logger.debug(f'Object "{object_id}" trajectory ground truth padded points is: {n_gt_missing}.')

        if object_distance_threshold is not None:
            object_center_point = object_traj_hist[-1, :2]  # mask values removed
            alpha_x = np.abs(object_center_point[0]) / agent_speed[0]  # coordinates are already normalized (relative to agent)
            alpha_y = np.abs(object_center_point[1]) / agent_speed[1]

            if alpha_x > object_distance_threshold:
                n_objects -= 1
                logger.debug(f'Object "{object_id} is too far on x axis: {alpha_x:.2f} > {object_distance_threshold}.')
                continue

            if alpha_y > object_distance_threshold:
                n_objects -= 1
                logger.debug(f'Object "{object_id} is too far on y axis: {alpha_y:.2f} > {object_distance_threshold}.')
                continue

        objects_traj_hist_list.append(object_traj_hist)
        objects_traj_gt_list.append(object_traj_gt)

    objects_traj_hists = np.stack(objects_traj_hist_list)
    objects_traj_gts = np.stack(objects_traj_gt_list)
    # mask feature is dropped, coordinates are denormalized
    objects_center_points = [oth[-1, :2] + center_point for oth in objects_traj_hist_list]

    logger.debug(f'Total number of neighbors is {n_objects}.')
    # Asserts
    expected_traj_hist_shape = (n_objects, trajectory_history_window_length, 3)
    assert objects_traj_hists.shape == expected_traj_hist_shape, \
        f'Wrong shape: Expected {expected_traj_hist_shape} but found {objects_traj_hists.shape}'
    expected_traj_future_shape = (n_objects, trajectory_future_window_length, 3)
    assert objects_traj_gts.shape == expected_traj_future_shape, \
        f'Wrong shape: Expetced {expected_traj_future_shape} but found {objects_traj_gts.shape}'

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
    assert agent_traj_gt.shape == expected_traj_future_shape, f'Wrong shape: Expetced {expected_traj_future_shape} but found {agent_traj_gt.shape}'

    return agent_traj_hist, agent_traj_gt


def process_lane_data(
    avm: ArgoverseMap,
    city: str,
    agent_center_point: np.ndarray,
    objects_center_points: List[np.ndarray],
    add_neighboring_lanes: bool
) -> np.ndarray:
    """
    Generate feature for lane data that are nearby to agent or some object

    Args:
        avm: ArgoverseMap API
        city: Map (Miami or Pittsburg)
        agent_center_point: Agent center point
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

    for center_point in center_points:
        cx, cy = center_point

        object_lane_segment_ids = avm.get_lane_segments_containing_xy(cx, cy, city)
        if add_neighboring_lanes:
            successor_lane_segment_ids = lists.flatten([avm.get_lane_segment_successor_ids(ls, city) for ls in object_lane_segment_ids])
            predecessor_lane_segment_ids = lists.flatten([avm.get_lane_segment_predecessor_ids(ls, city) for ls in object_lane_segment_ids])
            adjacent_lane_segment_ids = lists.flatten([avm.get_lane_segment_adjacent_ids(ls, city) for ls in object_lane_segment_ids])

            object_lane_segment_ids = object_lane_segment_ids + successor_lane_segment_ids + predecessor_lane_segment_ids + adjacent_lane_segment_ids
            object_lane_segment_ids = [lsi for lsi in object_lane_segment_ids if lsi is not None]  # filter `None` values
        lane_segment_id_list.append(object_lane_segment_ids)
    lane_segment_ids = lists.flatten(lane_segment_id_list)

    n_lsi = len(lane_segment_ids)
    logger.debug(f'Total number of lane segments is {n_lsi}.')

    lsi_data_list: List[np.ndarray] = []
    for lsi in lane_segment_ids:
        # extract polygon
        centerline = avm.get_lane_segment_centerline(lsi, city)[:, :-1]  # Ignoring height coordinate
        centerline -= agent_center_point  # normalize coordinates

        # extract metadata features
        is_intersetion = avm.lane_is_in_intersection(lsi, city)
        is_traffic_control = avm.lane_has_traffic_control_measure(lsi, city)
        direction = encode_direction(avm.get_lane_turn_direction(lsi, city))
        # noinspection PyTypeChecker
        metadata = np.array([is_intersetion, is_traffic_control] + direction, dtype=np.float)

        # stack all features into one vector
        tiled_metadata = np.tile(metadata, centerline.shape[0]).reshape(centerline.shape[0], metadata.shape[0])
        lsi_data = np.hstack([centerline, tiled_metadata])
        lsi_data_list.append(lsi_data)

    data = np.stack(lsi_data_list)

    # Asserts
    expected_data_shape = (n_lsi, 10, 7)  # polygon shape is 21, number of features is 7
    assert data.shape == expected_data_shape, f'Wrong shape: Expetced {expected_data_shape} but found {data.shape}'

    return data


def find_and_process_centerline_features(
    avm: ArgoverseMap,
    city: str,
    agent_traj_hist: np.ndarray,
    center_point: np.ndarray,
    trajectory_future_window_length: int,
    max_centerline_distance: float
) -> Optional[np.ndarray]:
    """

    Args:
        avm: ArgoverseMap API
        city: Map (Miami of Pittsburg)
        agent_traj_hist: Agent trajectory history (used to find candidate centerlines)
        center_point: Center point
        trajectory_future_window_length: Trajectory future window length (parameter)
        max_centerline_distance: Maximum centerline distance (parameter)

    Returns:
        - None if not centerlines are found
        - Centerlines with normalized x and y coordinates
    """
    agent_traj_hist_denormalized = agent_traj_hist[:, :2].copy() + center_point

    try:
        centerlines = avm.get_candidate_centerlines_for_traj(agent_traj_hist_denormalized, city, max_search_radius=max_centerline_distance)
    except AssertionError as e:
        raise exceptions.NoCandidateCenterlinesWereFoundException('No centerline candidates were found!') from e

    n_found_centerlines = len(centerlines)
    logger.debug(f'Found {n_found_centerlines} candidate centerlines.')

    # normalize centerline trajectories
    centerlines = [(c - center_point) for c in centerlines]

    # pad centerline trajectories
    centerlines = [c[:trajectory_future_window_length] for c in centerlines]
    # pad_trajectory returns tuple (padded_trajectory, n_missing_points)
    centerlines = [trajectories.pad_trajectory(c, trajectory_future_window_length, trajectories.PadType.FUTURE)[0] for c in centerlines]

    centerlines = np.stack(centerlines)

    # Asserts
    expected_centerlines_shape = (n_found_centerlines, trajectory_future_window_length, 3)
    assert centerlines.shape == expected_centerlines_shape, f'Wrong shape: Expetced {expected_centerlines_shape} but found {centerlines.shape}'

    return centerlines


def run(config: configparser.GlobalConfig):
    """
    Processes rad HD map using ArgoVerse API

    Args:
        config: Congiruation
    """
    parameters = config.data_process.parameters
    dataset_path = os.path.join(steps.SOURCE_PATH, config.data_process.input_path)
    output_path = os.path.join(steps.SOURCE_PATH, config.data_process.output_path)

    avfl = ArgoverseForecastingLoader(dataset_path)
    avm = ArgoverseMap()
    fig = None  # figure for visualization (optional usage)

    logger.info('Started data processing.')
    for data in avfl:
        sequence_name = os.path.basename(data.current_seq).split('.')[0]  # `path/.../1234.csv` -> `1234`
        logger.info(f'Processing sqeuence "{sequence_name}"...')

        try:
            agent_traj_hist, agent_traj_gt, center_point, agent_speed = process_agent_data(
                df=data.seq_df,
                trajectory_history_window_length=parameters.trajectory_history_window_length,
                trajectory_future_window_length=parameters.trajectory_future_window_length,
                trajectory_min_history_window_length=parameters.trajectory_min_history_window_length
            )

            objects_traj_hists, objects_traj_gts, objects_center_points = process_neighbors_data(
                df=data.seq_df,
                center_point=center_point,
                agent_traj_hist=agent_traj_hist,
                agent_traj_gt=agent_traj_gt,
                agent_speed=agent_speed,
                object_distance_threshold=parameters.object_distance_threshold,
                trajectory_history_window_length=parameters.trajectory_history_window_length,
                trajectory_future_window_length=parameters.trajectory_future_window_length,
                object_trajectory_min_history_window_length=parameters.object_trajectory_min_history_window_length,
                object_trajectory_min_future_window_length=parameters.object_trajectory_min_future_window_length
            )

            agent_traj_hist, agent_traj_gt = drop_agent_traj_timestamps(agent_traj_hist, agent_traj_gt)

            lane_features = process_lane_data(
                avm=avm,
                city=data.city,
                agent_center_point=center_point,
                objects_center_points=objects_center_points,
                add_neighboring_lanes=parameters.add_neighboring_lanes
            )

            centerline_candidate_features = find_and_process_centerline_features(
                avm=avm,
                city=data.city,
                agent_traj_hist=agent_traj_hist,
                center_point=center_point,
                trajectory_future_window_length=parameters.trajectory_future_window_length,
                max_centerline_distance=parameters.max_centerline_distance
            )

            scenario = ScenarioData(
                id=sequence_name,
                center_point=center_point,
                agent_traj_hist=agent_traj_hist,
                agent_traj_gt=agent_traj_gt,
                objects_traj_hists=objects_traj_hists,
                objects_traj_gts=objects_traj_gts,
                lane_features=lane_features,
                centerline_candidate_features=centerline_candidate_features
            )

            scenario.save(output_path)
            if config.data_process.visualize:
                fig = scenario.visualize(fig)
                figpath = os.path.join(output_path, scenario.id, 'scenario.png')
                fig.savefig(figpath)

        except exceptions.DataProcessException:
            logger.warning(f'Skipped "{sequence_name}" sequence!')


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
