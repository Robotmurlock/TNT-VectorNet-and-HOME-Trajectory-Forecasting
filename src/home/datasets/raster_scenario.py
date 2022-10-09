"""
Raster scenario
"""
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from typing import Optional
import matplotlib.pyplot as plt
import torch


from library.datasets.data_models import constants


@dataclass
class RasterScenarioData:
    id: str
    city: str
    center_point: np.ndarray
    agent_traj_hist: np.ndarray
    agent_traj_gt: np.ndarray
    objects_traj_hists: np.ndarray
    objects_traj_gts: np.ndarray
    raster_features: np.ndarray
    heatmap: np.ndarray
    angle: float

    def flip(self):
        """
        Flip scenario by y-axis
        """
        self.agent_traj_hist[:, 0] *= -1
        self.agent_traj_gt[:, 0] *= -1
        self.objects_traj_hists[:, :, 0] *= -1
        self.objects_traj_gts[:, :, 0] *= -1
        self.raster_features = np.flip(self.raster_features, axis=2).copy()
        self.heatmap = np.flip(self.heatmap, axis=1).copy()

    @property
    def ground_truth_trajectory_difference(self) -> torch.Tensor:
        diffs = self.agent_traj_gt.copy()
        diffs[1:, :] = diffs[1:, :] - diffs[:-1, :]
        return torch.tensor(diffs, dtype=torch.float32)

    @property
    def dirname(self) -> str:
        """
        Returns: Scenario dirname
        """
        return f'{self.city}_{self.id}'

    @property
    def window_size(self) -> int:
        return self.raster_features.shape[1]

    def save(self, path: str) -> None:
        """
        Saves Scenario data_models at given path

        Args:
            path: Path to save object
        """
        scenario_path = os.path.join(path, self.dirname)
        Path(scenario_path).mkdir(parents=True, exist_ok=True)

        catalog = {
            'center_point.npy': self.center_point,
            'agent_traj_hist.npy': self.agent_traj_hist,
            'agent_traj_gt.npy': self.agent_traj_gt,
            'objects_traj_hists.npy': self.objects_traj_hists,
            'objects_traj_gts.npy': self.objects_traj_gts,
            'raster_features.npy': self.raster_features,
            'heatmap.npy': self.heatmap
        }

        for filename, data in catalog.items():
            filepath = os.path.join(scenario_path, filename)
            np.save(filepath, data)

    @classmethod
    def load(cls, path: str) -> 'RasterScenarioData':
        """
        Loads Scenario Data from given path

        Args:
            path: Path to load object from

        Returns: RasterizedScenarioData
        """
        objects_to_load = ['center_point', 'agent_traj_hist', 'agent_traj_gt', 'objects_traj_hists',
                           'objects_traj_gts', 'raster_features', 'heatmap']
        filename = os.path.basename(path)
        sequence_city, sequence_id = filename.split('_')
        catalog = {'city': sequence_city, 'id': sequence_id}

        for object_name in objects_to_load:
            object_path = os.path.join(path, f'{object_name}.npy')
            catalog[object_name] = np.load(object_path)

        return cls(**catalog)

    def visualize_raster(self, fig: Optional[plt.Figure] = None):
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        else:
            fig.clf()

        feature_names = [
            'Drivable Area', 'Agent Trajectory', 'Neighbor Trajectory', 'Line Segment Metadata: is_intersection',
            'Line Segment Metadata: is_traffict_control', 'Line Segment Metadata: No direction',
            'Line Segment Metadata: Direction Left', 'Line Segment Metadata: Direction Right',
            'Candidate Line Segment'
        ]

        for feature_index, feature_name in enumerate(feature_names):
            plt.subplot(3, 3, feature_index+1)
            plt.imshow(self.raster_features[feature_index, :, :], cmap='gray')
            plt.axis('off')
            plt.title(feature_name, **constants.SMALL_TITLE_FONT)

        return fig

    def visualize_heatmap(self, fig: Optional[plt.Figure] = None, targets: Optional[np.ndarray] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        image = np.zeros(shape=(self.heatmap.shape[0], self.heatmap.shape[1], 3))

        # plot drivable area
        for i in range(3):
            image[:, :, i] = 0.5*self.raster_features[0]

        # plot ground truth (heatmap)
        # Scaling values of heatmap for better visualization
        gt_point_col, gt_point_row = [int(x) + self.heatmap.shape[0] // 2 for x in self.agent_traj_gt[-1]]
        gt_point_intensity = np.array([self.heatmap[gt_point_row+r_offset, gt_point_col+c_offset]
                                       for r_offset, c_offset in [(-1, 0), (0, -1), (1, 0), (0, -1)]]).mean()
        heatmap = np.minimum(1.0, self.heatmap / gt_point_intensity)
        for i in range(2):
            image[:, :, i] = np.maximum(image[:, :, i], heatmap)

        # Show image
        plt.imshow(image, cmap='YlOrRd')

        # plot agent trajectory
        agent_traj_hist = self.agent_traj_hist[:, :2] + self.window_size // 2
        agent_traj_gt = self.agent_traj_gt + self.window_size // 2
        plt.plot(agent_traj_hist[:, 0], agent_traj_hist[:, 1], '--', color='blue', linewidth=7)
        plt.plot(agent_traj_gt[:, 0], agent_traj_gt[:, 1], '--', color='green', linewidth=7)

        if targets is not None:
            plt.scatter(targets[:, 0], targets[:, 1], color='red', s=100, label='sampled targets')

        # set title and axis info
        plt.title(f'Heatmap {self.id}', **constants.TITLE_FONT)
        plt.xlabel('X', **constants.AXIS_FONT)
        plt.ylabel('Y', **constants.AXIS_FONT)
        plt.xticks(**constants.TICK_FONT)
        plt.yticks(**constants.TICK_FONT)

        return fig

    def visualize(
        self,
        fig: Optional[plt.Figure] = None,
        targets: Optional[np.ndarray] = None,
        agent_forecast: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None,
        map_radius: Optional[int] = None
    ) -> plt.Figure:
        """
        Visualizes scenario with:
            - agent
            - neighbor objects
            - nearby centerlines
            - candidate centerlines
            - agent forecast (optional)
            - objects (neighbors) forecasts (optional)

        Args:
            fig: Figure (if None then new one is created otherwise old one is cleared)
                Make sure to always use same figure instead of creating multiple ones
            targets:
            heatmap:
            agent_forecast: Agent forecast trajectory (optional)
            map_radius:

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        map_radius = self.heatmap.shape[0] // 2 if map_radius is None else map_radius

        if heatmap is not None:
            img = np.zeros(shape=(2*map_radius, 2*map_radius, 3))
            start_index = self.heatmap.shape[0] // 2 - map_radius
            end_index = self.heatmap.shape[1] // 2 + map_radius
            # noinspection PyArgumentList
            gt_heatmap = (self.heatmap / self.heatmap.max())[start_index:end_index, start_index:end_index]
            pred_heatmap = (heatmap / np.max(heatmap))[start_index:end_index, start_index:end_index]
            da_area = 0.25*self.raster_features[0, start_index:end_index, start_index:end_index]

            img[:, :, 1] = gt_heatmap
            img[:, :, 2] = pred_heatmap
            for i in range(3):
                img[:, :, i] = np.maximum(img[:, :, i], da_area)
            plt.imshow(img, cmap='gray')

        # plot agent
        agent_traj_hist = self.agent_traj_hist + map_radius
        agent_traj_gt = self.agent_traj_gt + map_radius
        plt.plot(agent_traj_hist[:, 0], agent_traj_hist[:, 1], color='darkgreen', linewidth=7, label='Agent history')
        plt.plot(agent_traj_gt[:, 0], agent_traj_gt[:, 1], color='yellow', linewidth=7, label='Agent Ground Truth')
        if agent_forecast is not None:
            if len(agent_forecast.shape) == 2:
                # In case of single forecasts, reshape it as (1, traj_length, 2)
                agent_forecast = agent_forecast.reshape(1, *agent_forecast.shape)
            assert len(agent_forecast.shape) == 3, 'Invalid agent forecast shape!'
            agent_forecast_local = agent_forecast.copy() + map_radius

            n_forecasts = agent_forecast.shape[0]
            for f_index in range(n_forecasts):
                plt.plot(agent_forecast_local[f_index, :, 0], agent_forecast_local[f_index, :, 1],
                         color='orange', linewidth=3, label='Agent Forecast')
                plt.scatter(agent_forecast_local[:, -1, 0], agent_forecast_local[:, -1, 1], color='orange', s=100)

        if targets is not None:
            targets = targets - self.heatmap.shape[0] // 2 + map_radius
            plt.scatter(targets[:, 0], targets[:, 1], color='red', s=200)

        plt.scatter(agent_traj_hist[-1:, 0], agent_traj_hist[-1:, 1], color='teal', s=200)

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', prop=constants.LEGEND_FONT)

        # set title and axis info
        plt.title(f'Scenario {self.id}', **constants.TITLE_FONT)
        plt.xlabel('X', **constants.AXIS_FONT)
        plt.ylabel('Y', **constants.AXIS_FONT)
        plt.xticks(**constants.TICK_FONT)
        plt.yticks(**constants.TICK_FONT)

        return fig

