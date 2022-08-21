import os
import logging
from typing import List, Optional, Tuple
from tqdm import tqdm
import cv2
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
from pathlib import Path

from utils import image_processing, time, steps
import configparser
from datasets.data_models import ScenarioData, RectangleBox, RasterScenarioData
import conventions


logger = logging.getLogger('DataRasterization')


def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def form_driveable_area_raster(da_map: np.ndarray, view: RectangleBox, angle: float, size: int = 224) -> np.ndarray:
    """
    Pads driveable area by view

    Args:
        da_map: Driveable Area Map
        view: Agent View
        angle: Scene rotation angle
        size: 224

    Returns: Padded Agent driveable area (by view)
    """
    # TODO: fix left-up, right-bottom mix
    da_raster = da_map[max(0, view.left):view.right, max(0, view.up):view.bottom].copy()
    padsize_up = max(-view.left, 0)
    if padsize_up > 0:
        pad_up = np.zeros(shape=(padsize_up, da_raster.shape[1]))
        da_raster = np.concatenate([pad_up, da_raster], axis=0)
    padsize_down = max(view.right-da_map.shape[0], 0)
    if padsize_down > 0:
        pad_down = np.zeros(shape=(padsize_down, da_raster.shape[1]))
        da_raster = np.concatenate([da_raster, pad_down], axis=0)
    padsize_left = max(-view.up, 0)
    if padsize_left > 0:
        pad_left = np.zeros(shape=(da_raster.shape[0], padsize_left))
        da_raster = np.concatenate([pad_left, da_raster], axis=1)
    padsize_right = max(view.bottom-da_map.shape[1], 0)
    if padsize_right > 0:
        pad_right = np.zeros(shape=(da_raster.shape[0], padsize_right))
        da_raster = np.concatenate([da_raster, pad_right], axis=1)

    da_raster = rotate_image(da_raster, -(180 / np.pi) * angle)
    da_raster = da_raster.reshape(1, *da_raster.shape)
    assert da_raster.shape == (1, size, size), f'Wrong DA raster shape: Expected {(1, size, size)} but found {da_raster.shape}! ' \
        f'view: {view}, DA map shape: {da_map.shape}'
    return da_raster


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
    rasterized_agent_trajectory = np.zeros(shape=(1, view.height, view.width))
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
        rasterized_agent_trajectory[0, up:bottom, left:right] = 1

    expected_shape = (1, view.height, view.width)
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
    rasterized_objects_trajectory = np.zeros(shape=(1, view.height, view.width))
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
            rasterized_objects_trajectory[0, up:bottom, left:right] = 1

    expected_shape = (1, view.height, view.width)
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
    rasterized_lanes = np.zeros(shape=(lane_features.shape[2]-2, view.height, view.width))
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
                rasterized_lanes[feat_index-2, up:bottom, left:right] = features[feat_index]

    expected_shape = (lane_features.shape[2]-2, view.height, view.width)
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
    rasterized_candidate_centerlines = np.zeros(shape=(1, view.height, view.width))
    clp_halfheight, clp_halfwidth = centerline_point_shape[0] // 2, centerline_point_shape[1] // 2
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2

    for cc_index in range(centerline_candidate_features.shape[0]):
        for coord_index in range(centerline_candidate_features.shape[1]):
            y, x, mask = [int(x) for x in centerline_candidate_features[cc_index, coord_index]]
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


def create_heatmap(
    agent_traj_gt: np.ndarray,
    driveable_area: np.ndarray,
    view: RectangleBox,
    kernel_size: int,
    sigma: int,
    object_shape: List[int],
    size: int = 224
) -> np.ndarray:
    """
    Creates ground truth heatmap by encoding by applying guass kernerl on ground truth location of last agent point in
    ground truth trajectory

    Args:
        agent_traj_gt: Ground truth trajectory
        driveable_area: Driveable area (probability is 0 in non-driveable areas)
        view: view bounding box
        kernel_size: Gauss kernel size
        sigma: Gauss kernel sigma
        object_shape: Agent object shape
        size:

    Returns: Ground Truth Heatmap
    """
    heatmap = np.zeros(shape=(view.height, view.width))
    v_halfheight, v_halfwidth = view.height // 2, view.width // 2
    o_halfheight, o_halfwidth = object_shape[0] // 2, object_shape[1] // 2  # distance between object center and its borders

    # Add agent rectangle at position
    y, x = [int(x) for x in agent_traj_gt[-1]]
    y_center, x_center = min(size-1, max(0, v_halfheight + y)), min(size-1, max(0, v_halfwidth + x))
    up, bottom = x_center - o_halfheight, x_center + o_halfheight + 1
    left, right = y_center - o_halfwidth, y_center + o_halfwidth + 1
    heatmap[up:bottom, left:right] = 1

    # Apply gaussian kernel
    gaussian_filter = image_processing.create_gauss_kernel(kernel_size, sigma=sigma)
    heatmap = cv2.filter2D(heatmap, -1, gaussian_filter)
    heatmap[x_center, y_center] = 1.0

    heatmap = heatmap * driveable_area[0]  # Set probability to 0 on in non-driveable area

    return heatmap


def plot_all_feature_maps(rasterized_features: np.ndarray, path: str, fig: Optional[plt.Figure] = None) -> None:
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    else:
        fig.clf()

    Path(path).mkdir(parents=True, exist_ok=True)
    for feature_index in range(rasterized_features.shape[0]):
        plt.imshow(rasterized_features[feature_index], origin='lower', cmap='gray')
        fig.savefig(os.path.join(path, f'{feature_index:02d}.png'))


def pad_objects_trajectories(
    center_points:
    np.ndarray,
    objects_traj_hists: np.ndarray,
    objects_traj_gts: np.ndarray,
    max_neighbours: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    
    Args:
        center_points: 
        objects_traj_hists: 
        objects_traj_gts: 
        max_neighbours: 

    Returns:

    """
    if objects_traj_hists.shape[0] > max_neighbours:
        object_traj_points = objects_traj_hists[:, -1]
        distances = (object_traj_points[:, 0] - center_points[0]) ** 2 + (object_traj_points[:, 1] - center_points[1]) ** 2
        indices = np.argsort(distances)[:max_neighbours]
        objects_traj_hists = objects_traj_hists[indices]
        objects_traj_gts = objects_traj_gts[indices]

    if objects_traj_hists.shape[0] < max_neighbours:
        hists_padding = np.zeros(shape=(max_neighbours - objects_traj_hists.shape[0], objects_traj_hists.shape[1], 3))
        objects_traj_hists = np.concatenate([objects_traj_hists, hists_padding], axis=0)
        gts_padding = np.zeros(shape=(max_neighbours - objects_traj_hists.shape[0], objects_traj_gts.shape[1], 3))
        objects_traj_gts = np.concatenate([objects_traj_gts, gts_padding], axis=0)

    assert objects_traj_hists.shape == (max_neighbours, 20, 3)

    return objects_traj_hists, objects_traj_gts


class ScenarioRasterPreprocess:
    def __init__(self, config: configparser.GlobalConfig, disable_visualization: bool = False):
        """
        TODO

        Args:
            config:
            disable_visualization:
        """
        # configs
        self._config = config
        self._params = config.raster.data_process.parameters
        self._disable_visualization = disable_visualization

        # state
        self._avm = ArgoverseMap()
        self._city_cache = {}

    def process(self, path: str) -> RasterScenarioData:
        scenario = ScenarioData.load(path)
        if scenario.city not in self._city_cache:
            self._city_cache[scenario.city] = self._avm.get_rasterized_driveable_area(scenario.city)

        c_y, c_x = int(scenario.center_point[0]), int(scenario.center_point[1])
        c_y = int(c_y + self._city_cache[scenario.city][1][0, 2])
        c_x = int(c_x + self._city_cache[scenario.city][1][1, 2])

        view = RectangleBox(
            up=c_y - self._params.agent_view_window_halfize,
            left=c_x - self._params.agent_view_window_halfize,
            bottom=c_y + self._params.agent_view_window_halfize,
            right=c_x + self._params.agent_view_window_halfize,
        )
        view_normalized = view.move(-c_y, -c_x)

        da_raster = form_driveable_area_raster(da_map=self._city_cache[scenario.city][0], view=view, angle=scenario.angle)

        agent_trajectory_raster = rasterize_agent_trajectory(
            agent_traj_hist=scenario.agent_traj_hist,
            view=view_normalized,
            object_shape=self._params.object_shape)
        objects_trajectory_raster = rasterize_object_trajectories(
            objects_traj_hists=scenario.objects_traj_hists,
            view=view_normalized,
            object_shape=self._params.object_shape)
        lane_raster = rasterize_lanes(
            lane_features=scenario.lane_features,
            view=view_normalized,
            centerline_point_shape=self._params.centerline_point_shape)
        candidate_centerlines_raster = rasterize_candidate_centerlines(
            centerline_candidate_features=scenario.centerline_candidate_features,
            view=view_normalized,
            centerline_point_shape=self._params.centerline_point_shape)

        rasterized_features = np.vstack([
            da_raster,
            agent_trajectory_raster,
            objects_trajectory_raster,
            lane_raster,
            candidate_centerlines_raster
        ])

        objects_traj_hists, objects_traj_gts = pad_objects_trajectories(
            scenario.center_point,
            scenario.objects_traj_hists,
            scenario.objects_traj_gts,
            self._params.max_neighbours
        )

        heatmap = create_heatmap(
            agent_traj_gt=scenario.agent_traj_gt,
            driveable_area=da_raster,
            view=view_normalized,
            kernel_size=self._params.gauss_kernel_size,
            sigma=self._params.gauss_kernel_sigma,
            object_shape=self._params.object_shape)

        raster_scenario = RasterScenarioData(
            id=scenario.id,
            city=scenario.city,
            center_point=scenario.center_point,
            agent_traj_hist=scenario.agent_traj_hist,
            agent_traj_gt=scenario.agent_traj_gt,
            objects_traj_hists=objects_traj_hists,
            objects_traj_gts=objects_traj_gts,
            raster_features=rasterized_features,
            heatmap=heatmap,
            angle=scenario.angle
        )

        return raster_scenario


@time.timeit
def run(config: configparser.GlobalConfig):
    """
    Converts vectorized structured data to images
    Args:
        config: Config
    """
    for split in conventions.SPLIT_NAMES:
        input_path = os.path.join(config.global_path, config.raster.data_process.input_path, split)
        output_path = os.path.join(config.global_path, config.raster.data_process.output_path, split)
        scenario_names = os.listdir(input_path)

        preprocessor = ScenarioRasterPreprocess(config)

        fig = None
        for scenario_name in tqdm(scenario_names):
            scenario_inpath = os.path.join(input_path, scenario_name)
            scenario_outpath = os.path.join(output_path, scenario_name)
            Path(scenario_inpath).mkdir(exist_ok=True, parents=True)
            Path(scenario_outpath).mkdir(exist_ok=True, parents=True)

            data = preprocessor.process(scenario_inpath)
            fig = data.visualize_heatmap(fig)
            fig.savefig(os.path.join(scenario_outpath, 'heatmap.png'))

            fig = data.visualize_raster(fig)
            fig.savefig(os.path.join(scenario_outpath, 'raster.png'))


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
