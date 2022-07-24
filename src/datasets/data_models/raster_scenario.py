from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from typing import Optional
import matplotlib.pyplot as plt
import torch


from utils import trajectories


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
        return self.raster_features.shape[0]

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

    def visualize_heatmap(
        self,
        fig: Optional[plt.Figure] = None
    ) -> plt.Figure:
        """
        Visualizes scenario with:
            - agent
            - neighbor objects
            - nearby centerlines
            - candidate centerlines

        Args:
            fig: Figure (if None then new one is created otherwise old one is cleared)
                Make sure to always use same figure instead of creating multiple ones

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        image = np.zeros(shape=(self.heatmap.shape[0], self.heatmap.shape[1], 3))

        # plot drivable area
        for i in range(3):
            image[:, :, i] = 0.5*self.raster_features[0]

        # plot agent and other objects
        image[:, :, 1] = np.maximum(image[:, :, 1], self.raster_features[1])
        image[:, :, 0] = np.maximum(image[:, :, 0], 0.7*self.raster_features[2])

        # plot ground truth (heatmap)
        for i in range(3):
            image[:, :, i] = np.maximum(image[:, :, i], self.heatmap)

        # Show image
        plt.imshow(image, origin='lower', cmap='YlOrRd')

        # set title and axis info
        plt.title(f'Rasterized Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        return fig

    def visualize(
        self,
        fig: Optional[plt.Figure] = None,
        targets: Optional[np.ndarray] = None,
        agent_forecast: Optional[np.ndarray] = None,
        heatmap: Optional[np.ndarray] = None
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

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()


        if heatmap is not None:
            # heatmap = np.maximum(heatmap, self.heatmap)
            plt.imshow(heatmap, origin='lower', cmap='gray')

        # plot agent
        agent_traj_hist = trajectories.rotate_points(self.agent_traj_hist, -self.angle) + 112
        agent_traj_gt = trajectories.rotate_points(self.agent_traj_gt, -self.angle) + 112
        plt.plot(agent_traj_hist[:, 0], agent_traj_hist[:, 1], color='green', linewidth=7, label='Agent history')
        plt.plot(agent_traj_gt[:, 0], agent_traj_gt[:, 1], color='lightgreen', linewidth=7, label='Agent Ground Truth')
        if agent_forecast is not None:
            if len(agent_forecast.shape) == 2:
                # In case of single forecasts, reshape it as (1, traj_length, 2)
                agent_forecast = agent_forecast.reshape(1, *agent_forecast.shape)
            assert len(agent_forecast.shape) == 3, 'Invalid agent forecast shape!'

            n_forecasts = agent_forecast.shape[0]
            for f_index in range(n_forecasts):
                agent_forecast[f_index, :] = trajectories.rotate_points(agent_forecast[f_index, :], -self.angle) + 112
                plt.plot(agent_forecast[f_index, :, 0], agent_forecast[f_index, :, 1],
                         color='turquoise', linewidth=3, label='Agent Forecast')

        if targets is not None:
            targets = trajectories.rotate_points(targets - 112, -self.angle) + 112
            plt.scatter(targets[:, 0], targets[:, 1], color='red', s=200)

        plt.scatter(agent_traj_hist[-1:, 0], agent_traj_hist[-1:, 1], color='teal', s=200)

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        # set title and axis info
        plt.title(f'Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        return fig

