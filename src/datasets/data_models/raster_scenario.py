from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch


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

    @property
    def dirname(self) -> str:
        """
        Returns: Scenario dirname
        """
        return f'{self.city}_{self.id}'

    @property
    def features(self) -> Tuple[torch.Tensor, ...]:
        """
        Returns: Input features for PyTorch models
        """
        items = (self.agent_traj_hist, self.objects_traj_hists, self.raster_features)
        return tuple(torch.tensor(item, dtype=torch.float32) for item in items)

    @property
    def ground_truth(self) -> Tuple[torch.Tensor, ...]:
        """
        Returns: Ground truth for PyTorch models
        """
        items = (self.agent_traj_gt, self.objects_traj_gts, self.heatmap)
        return tuple(torch.tensor(item, dtype=torch.float32) for item in items)

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

    def visualize(
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

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        image = np.zeros(shape=(self.heatmap.shape[0], self.heatmap.shape[1], 3))

        # plot drivable area
        for i in range(3):
            image[:, :, i] = 0.5*self.raster_features[0]

        # plot agent and other objects
        for timestamp_index in range(self.agent_traj_hist.shape[0]):
            image[:, :, 1] = np.maximum(image[:, :, 1], self.raster_features[1+timestamp_index])
            image[:, :, 0] = np.maximum(image[:, :, 0], self.raster_features[1+self.agent_traj_hist.shape[0] + timestamp_index])

        # plot ground truth (heatmap)
        for i in range(3):
            image[:, :, i] = np.maximum(image[:, :, i], self.heatmap)

        # Show image
        plt.imshow(image, origin='lower')

        # set title and axis info
        plt.title(f'Rasterized Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        return fig
