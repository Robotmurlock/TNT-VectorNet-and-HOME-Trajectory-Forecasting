"""
Base dataclass for processed HD maps
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from library.datasets.data_models import constants


@dataclass
class ScenarioData:
    id: str
    city: str
    center_point: np.ndarray
    angle: float
    agent_traj_hist: np.ndarray
    agent_traj_gt: np.ndarray
    objects_traj_hists: np.ndarray
    objects_traj_gts: np.ndarray
    lane_features: np.ndarray
    centerline_candidate_features: np.ndarray

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
    def features(self) -> Tuple[torch.Tensor, ...]:
        """
        Returns: Input features for PyTorch models
        """
        items = (self.agent_traj_hist, self.objects_traj_hists, self.lane_features, self.centerline_candidate_features)
        return tuple(torch.tensor(item, dtype=torch.float32) for item in items)

    @property
    def ground_truth(self) -> torch.Tensor:
        """
        Returns: Ground truth for PyTorch models
        """
        return torch.tensor(self.agent_traj_gt, dtype=torch.float32)

    @property
    def final_point_gt(self) -> torch.Tensor:
        return torch.tensor(self.agent_traj_gt[-1, :], dtype=torch.float32)

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
            'angle.npy': np.array(self.angle),
            'agent_traj_hist.npy': self.agent_traj_hist,
            'agent_traj_gt.npy': self.agent_traj_gt,
            'objects_traj_hists.npy': self.objects_traj_hists,
            'objects_traj_gts.npy': self.objects_traj_gts,
            'lane_features.npy': self.lane_features,
            'centerline_candidate_features.npy': self.centerline_candidate_features
        }

        for filename, data in catalog.items():
            filepath = os.path.join(scenario_path, filename)
            np.save(filepath, data)

    @classmethod
    def load(cls, path: str) -> 'ScenarioData':
        """
        Loads Scenario Data from given path

        Args:
            path: Path to load object from

        Returns: ScenarioData
        """
        objects_to_load = ['center_point', 'angle', 'agent_traj_hist', 'agent_traj_gt', 'objects_traj_hists',
                           'objects_traj_gts', 'lane_features', 'centerline_candidate_features']
        filename = os.path.basename(path)
        sequence_city, sequence_id = filename.split('_')
        catalog = {'city': sequence_city, 'id': sequence_id}

        for object_name in objects_to_load:
            object_path = os.path.join(path, f'{object_name}.npy')
            catalog[object_name] = np.load(object_path)

        catalog['angle'] = float(catalog['angle'])  # angle is float saves as numpy array

        return cls(**catalog)

    def visualize(
        self,
        fig: Optional[plt.Figure] = None,
        agent_forecast: Optional[np.ndarray] = None,
        objects_forecast: Optional[np.ndarray] = None
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
            agent_forecast: Agent forecast trajectory (optional)
            objects_forecast: Objects forecast trajectories (optional)

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        # plot lines
        for lane_index in range(self.lane_features.shape[0]):
            lane_polygon = self.lane_features[lane_index, :21, :2]
            plt.plot(lane_polygon[:, 0], lane_polygon[:, 1], color='lightgray', linewidth=3, label='centerline')

        # plot candidate centerlines
        for c_index in range(self.centerline_candidate_features.shape[0]):
            plt.plot(self.centerline_candidate_features[c_index, :, 0], self.centerline_candidate_features[c_index, :, 1],
                     color='k', linewidth=5, linestyle='dashed', label='candidate centerline')

        # plot agent
        plt.plot(self.agent_traj_hist[:, 0], self.agent_traj_hist[:, 1], color='green', linewidth=7, label='Agent history')
        plt.plot(self.agent_traj_gt[:, 0], self.agent_traj_gt[:, 1], color='lightgreen', linewidth=7, label='Agent Ground Truth')
        if agent_forecast is not None:
            if len(agent_forecast.shape) == 2:
                # In case of single forecasts, reshape it as (1, traj_length, 2)
                agent_forecast = agent_forecast.reshape(1, *agent_forecast.shape)
            assert len(agent_forecast.shape) == 3, 'Invalid agent forecast shape!'

            n_forecasts = agent_forecast.shape[0]
            for f_index in range(n_forecasts):
                plt.plot(agent_forecast[f_index, :, 0], agent_forecast[f_index, :, 1],
                         color='turquoise', linewidth=7, label='Agent Forecast', linestyle='dashed')

        plt.scatter(self.agent_traj_hist[-1:, 0], self.agent_traj_hist[-1:, 1], color='teal', s=200)

        # plot objects
        for obj_index in range(self.objects_traj_hists.shape[0]):
            plt.plot(self.objects_traj_hists[obj_index, :, 0], self.objects_traj_hists[obj_index, :, 1], color='orange', linewidth=4,
                     label='Neighbor history')
            plt.plot(self.objects_traj_gts[obj_index, :, 0], self.objects_traj_gts[obj_index, :, 1], color='yellow', linewidth=4,
                     label='Neighbor Ground Truth')
            if objects_forecast is not None:
                plt.plot(objects_forecast[obj_index, :, 0], objects_forecast[obj_index, :, 1], color='olive', linewidth=4,
                         label='Neighbor Forecast', linestyle='dashed')

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop=constants.LEGEND_FONT)

        # set title and axis info
        plt.title(f'Scenario {self.id}', **constants.TITLE_FONT)
        plt.xlabel('X', **constants.AXIS_FONT)
        plt.ylabel('Y', **constants.AXIS_FONT)
        plt.xticks(**constants.TICK_FONT)
        plt.yticks(**constants.TICK_FONT)

        return fig
