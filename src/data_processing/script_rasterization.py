import os
import logging
from typing import Dict, List, Tuple
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap

from utils import steps, image_processing
import configparser
from datasets.data_models import ScenarioData, RectangleBox, RasterScenarioData


logger = logging.getLogger('DataRasterization')


def rasterize_agent_trajectory(
    agent_traj_hist: np.ndarray,
    view: RectangleBox,
    object_shape: List[int]
) -> np.ndarray:
    """
    Input - Agent trajectory in format (Nh, 3) where Nh is trajectory length
    Output - Rasterized agent trajectory in format (20, view_height, view_width)

    Args:
        agent_traj_hist: Agent trajectory
        view: view bounding box
        object_shape: Agent object shape (width and height)

    Returns: Rasterized agent trajectory
    """
    rasterized_agent_trajectory = np.zeros(shape=(agent_traj_hist.shape[0], view.height, view.width))
    o_halfheight, o_halfwidth = object_shape[0] // 2, object_shape[1] // 2  # distance between object center and its borders
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2  # distance between view center and its borders

    for timestamp_index in range(agent_traj_hist.shape[0]):
        y, x, mask = (int(x) for x in agent_traj_hist[timestamp_index])
        if mask == 0:
            # Skip masked values (padded values)
            continue

        if not view.contains(y, x):
            # Objects is not in viewpoint
            continue

        # Agent center position is at view center position (v_halfheight, v_halfwidth)
        # Agent object shape is (o_halfheight, o_halfwidth)
        # All coordinates are relative to agent center (view center) position
        up, bottom = v_halfheight + x - o_halfheight, v_halfheight + x + o_halfheight
        left, right = v_halfwidth + y - o_halfwidth, v_halfwidth + y + o_halfwidth
        rasterized_agent_trajectory[timestamp_index, up:bottom, left:right] = 1

    expected_shape = (agent_traj_hist.shape[0], view.height, view.width)
    assert expected_shape == rasterized_agent_trajectory.shape, \
        f'Wrong shape: Expetced {expected_shape} but found {rasterized_agent_trajectory.shape}'

    return rasterized_agent_trajectory


def rasterize_object_trajectories(
    objects_traj_hists: np.ndarray,
    view: RectangleBox,
    object_shape: List[int]
) -> np.ndarray:
    """
    Input - Objects trajectory in format (Nobj, Nh, 3) where Nobj is number of objects and Nh is trajectory length
    Output - Rasterized object trajectories in format (20, view_height, view_width)

    Args:
        objects_traj_hists: Objects trajectories trajectory
        view: view bounding box
        object_shape: Object shape (width and height)

    Returns: Rasterized object trajectoris
    """
    rasterized_objects_trajectory = np.zeros(shape=(objects_traj_hists.shape[1], view.height, view.width))
    o_halfheight, o_halfwidth = object_shape[0] // 2, object_shape[1] // 2  # distance between object center and its borders
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2  # distance between view center and its borders

    for object_index in range(objects_traj_hists.shape[0]):
        for timestamp_index in range(objects_traj_hists.shape[1]):
            y, x, mask = (int(x) for x in objects_traj_hists[object_index, timestamp_index])
            if mask == 0:
                # Skip masked values
                continue

            if not view.contains(y, x):
                # Objects is not in viewpoint
                continue

            # Same as for `rasterize_object_trajectories`
            up, bottom = v_halfheight + x - o_halfheight, v_halfheight + x + o_halfheight
            left, right = v_halfwidth + y - o_halfwidth, v_halfwidth + y + o_halfwidth
            rasterized_objects_trajectory[timestamp_index, up:bottom, left:right] = 1

    expected_shape = (objects_traj_hists.shape[1], view.height, view.width)
    assert expected_shape == rasterized_objects_trajectory.shape, \
        f'Wrong shape: Expetced {expected_shape} but found {rasterized_objects_trajectory.shape}'

    return rasterized_objects_trajectory


def rasterize_lanes(lane_features: np.ndarray, view: RectangleBox, centerline_point_shape: List[int]) -> np.ndarray:
    """
    Rasterizes agent and neighbours nearby lane centerlines
    Input - Lane centerlines with features (Nl, Ll, 7) where Nl is number of lines, Ll is line length and each instance has
        7 features (x, y, is_intersection, is_traffic_control, is_no_direction, is_direction_left, is_direction_right)

    Args:
        lane_features: Lane features
        view: view bounding box
        centerline_point_shape: Shape for each point extracted from centerline features

    Returns: Rasterized lane centerlines
    """
    rasterized_lanes = np.zeros(shape=(lane_features.shape[2]-1, view.height, view.width))
    clp_halfheight, clp_halfwidth = centerline_point_shape[0] // 2, centerline_point_shape[1] // 2
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2

    for lane_index in range(lane_features.shape[0]):
        for lane_point_index in range(lane_features.shape[1]):
            features = [int(x) for x in lane_features[lane_index, lane_point_index]]
            y, x = features[:2]
            if not view.contains(x, y):
                # Lane point is not in viewpoint
                continue

            up, bottom = v_halfheight + x - clp_halfheight, v_halfheight + x + clp_halfwidth
            left, right = v_halfwidth + y - clp_halfheight, v_halfwidth + y + clp_halfwidth
            rasterized_lanes[0, up:bottom, left:right] = 1
            for feat_index in range(2, lane_features.shape[2]):
                rasterized_lanes[feat_index-1, up:bottom, left:right] = features[feat_index]

    expected_shape = (lane_features.shape[2]-1, view.height, view.width)
    assert expected_shape == rasterized_lanes.shape, \
        f'Wrong shape: Expected {expected_shape} but found {rasterized_lanes.shape}'

    return rasterized_lanes


def rasterize_candidate_centerlines(
    centerline_candidate_features: np.ndarray,
    view: RectangleBox,
    centerline_point_shape: List[int]
) -> np.ndarray:
    """
    Rasterizes candidate lane centerlines
    Input - Lane centerlines with features (Ncl, Lcl, 2) where Nl is number of centerlines, Ll is centerline length and each instance has
        2 features (x, y)

    Args:
        centerline_candidate_features: Candidate centerlines extracted from agent history
        view: view bounding box
        centerline_point_shape: Shape for each point extracted from centerline features

    Returns: Rasterized candidate lane centerlines

    """
    # TODO: Add centerline rasterization for other objects?
    rasterized_candidate_centerlines = np.zeros(shape=(1, view.height, view.width))
    clp_halfheight, clp_halfwidth = centerline_point_shape[0] // 2, centerline_point_shape[1] // 2
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2

    for cc_index in range(centerline_candidate_features.shape[0]):
        y, x, mask = [int(x) for x in centerline_candidate_features[0, cc_index]]
        if mask == 0:
            continue

        if not view.contains(x, y):
            # Lane point is not in viewpoint
            continue

        up, bottom = v_halfheight + x - clp_halfheight, v_halfheight + x + clp_halfwidth
        left, right = v_halfwidth + y - clp_halfheight, v_halfwidth + y + clp_halfwidth
        rasterized_candidate_centerlines[0, up:bottom, left:right] = 1

    expected_shape = (1, view.height, view.width)
    assert expected_shape == rasterized_candidate_centerlines.shape, \
        f'Wrong shape: Expected {expected_shape} but found {rasterized_candidate_centerlines.shape}'

    return rasterized_candidate_centerlines


def create_heatmap(agent_traj_gt: np.ndarray, view: RectangleBox, kernel_size: int, sigma: int) -> np.ndarray:
    """
    Creates ground truth heatmap by encoding by applying guass kernerl on ground truth location of last agent point in
    ground truth trajectory

    Args:
        agent_traj_gt: Ground truth trajectory
        view: view bounding box
        kernel_size: Gauss kernel size
        sigma: Gauss kernel sigma

    Returns: Ground Truth Heatmap
    """
    heatmap = np.zeros(shape=(view.height, view.width))
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2
    gaussian_filter = image_processing.create_gauss_kernel(kernel_size, sigma=sigma)
    kernel_halfsize = kernel_size // 2
    gaussian_filter = gaussian_filter * (1 / gaussian_filter[kernel_halfsize, kernel_halfsize])

    y, x = [int(x) for x in agent_traj_gt[-1]]
    y_center, x_center = v_halfheight + y, v_halfwidth + x
    up, bottom = x_center - kernel_halfsize, x_center + kernel_halfsize + 1
    left, right = y_center - kernel_halfsize, y_center + kernel_halfsize + 1

    heatmap[up:bottom, left:right] = np.maximum(heatmap[up:bottom, left:right], gaussian_filter)

    return heatmap


def run(config: configparser.GlobalConfig):
    scenario_path = os.path.join(steps.SOURCE_PATH, config.data_process_rasterization.input_path)
    scenario_paths = [os.path.join(scenario_path, dirname) for dirname in os.listdir(scenario_path)]
    output_path = os.path.join(steps.SOURCE_PATH, config.data_process_rasterization.output_path)

    scenarios = [ScenarioData.load(path) for path in scenario_paths]
    logger.info(f'Found {len(scenarios)} scenarios.')
    dpr_config = config.data_process_rasterization

    avm = ArgoverseMap()
    city_da_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    agent_window_halfsize = dpr_config.parameters.agent_view_window_size // 2
    fig = None

    logger.info('Started datasets processing.')
    for scenario in scenarios:
        if scenario.city not in city_da_cache:
            logging.debug(f'Loaded drivable area for "{scenario.city}".')
            city_da_cache[scenario.city] = avm.get_rasterized_driveable_area(scenario.city)

        c_y, c_x = int(scenario.center_point[0]), int(scenario.center_point[1])
        c_y = int(c_y + city_da_cache[scenario.city][1][0, 2])
        c_x = int(c_x + city_da_cache[scenario.city][1][1, 2])

        view = RectangleBox(
            up=c_y - agent_window_halfsize,
            left=c_x - agent_window_halfsize,
            bottom=c_y + agent_window_halfsize,
            right=c_x + agent_window_halfsize,
        )
        view_normalized = view.move(-c_y, -c_x)

        da_raster = city_da_cache[scenario.city][0][np.newaxis, view.left:view.right, view.up:view.bottom]  # driveable area

        agent_trajectory_raster = rasterize_agent_trajectory(
            agent_traj_hist=scenario.agent_traj_hist,
            view=view_normalized,
            object_shape=dpr_config.parameters.object_shape)
        objects_trajectory_raster = rasterize_object_trajectories(
            objects_traj_hists=scenario.objects_traj_hists,
            view=view_normalized,
            object_shape=dpr_config.parameters.object_shape)
        lane_raster = rasterize_lanes(
            lane_features=scenario.lane_features,
            view=view_normalized,
            centerline_point_shape=dpr_config.parameters.centerline_point_shape)
        candidate_centerlines_raster = rasterize_candidate_centerlines(
            centerline_candidate_features=scenario.centerline_candidate_features,
            view=view_normalized,
            centerline_point_shape=dpr_config.parameters.centerline_point_shape)

        rasterized_features = np.vstack([
            da_raster,
            agent_trajectory_raster,
            objects_trajectory_raster,
            lane_raster,
            candidate_centerlines_raster
        ])

        heatmap = create_heatmap(
            agent_traj_gt=scenario.agent_traj_gt,
            view=view_normalized,
            kernel_size=dpr_config.parameters.gauss_kernel_size,
            sigma=dpr_config.parameters.gauss_kernel_sigma)

        raster_scenario = RasterScenarioData(
            id=scenario.id,
            city=scenario.city,
            center_point=scenario.center_point,
            agent_traj_hist=scenario.agent_traj_hist,
            agent_traj_gt=scenario.agent_traj_gt,
            objects_traj_hists=scenario.objects_traj_hists,
            objects_traj_gts=scenario.objects_traj_gts,
            raster_features=rasterized_features,
            heatmap=heatmap)

        if dpr_config.visualize:
            logger.debug(f'Visualizing data for scenarion "{scenario.dirname}"')
            raster_scenario.save(output_path)
            fig = raster_scenario.visualize(fig)
            fig.savefig(os.path.join(output_path, raster_scenario.dirname, 'rasterized_scenario.png'))

    logger.info('Finished rasterization!')


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
