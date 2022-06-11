from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from typing import List, Optional
import torch
import matplotlib.pyplot as plt


from datasets.data_models.types import ObjectType


@dataclass
class GraphScenarioData:
    id: str
    city: str
    center_point: np.ndarray
    polylines: List[np.ndarray]
    agent_traj_gt: np.ndarray
    objects_traj_gts: np.ndarray
    anchors: np.ndarray
    ground_truth_point: np.ndarray

    @property
    def dirname(self) -> str:
        return f'{self.city}_{self.id}'

    @property
    def inputs(self) -> torch.Tensor:
        return torch.tensor(self.polylines, dtype=torch.float32)

    @property
    def ground_truth_trajectory(self) -> torch.Tensor:
        return torch.tensor(self.agent_traj_gt, dtype=torch.float32)

    @property
    def ground_truth_trajectory_difference(self) -> torch.Tensor:
        diffs = self.agent_traj_gt.copy()
        diffs[1:, :] = diffs[1:, :] - diffs[:-1, :]
        return torch.tensor(diffs, dtype=torch.float32)

    @property
    def target_proposals(self) -> torch.Tensor:
        return torch.tensor(self.anchors, dtype=torch.float32)

    @property
    def target_ground_truth(self) -> torch.Tensor:
        return torch.tensor(self.ground_truth_point, dtype=torch.float32)

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
            'agent_traj_gt.npy': self.agent_traj_gt,
            'objects_traj_gts.npy': self.objects_traj_gts,
            'anchors.npy': self.anchors,
            'ground_truth_point.npy': self.ground_truth_point,
            'polylines.npy': self.polylines
        }

        for filename, data in catalog.items():
            filepath = os.path.join(scenario_path, filename)
            np.save(filepath, data)

    @classmethod
    def load(cls, path: str) -> 'GraphScenarioData':
        """
        Loads Scenario Data from given path

        Args:
            path: Path to load object from

        Returns: RasterizedScenarioData
        """
        objects_to_load = ['center_point',  'agent_traj_gt', 'objects_traj_gts', 'anchors', 'ground_truth_point', 'polylines']
        filename = os.path.basename(path)
        sequence_city, sequence_id = filename.split('_')
        catalog = {'city': sequence_city, 'id': sequence_id}

        for object_name in objects_to_load:
            object_path = os.path.join(path, f'{object_name}.npy')
            catalog[object_name] = np.load(object_path)

        return cls(**catalog)

    def visualize(
        self,
        fig: Optional[plt.Figure] = None,
        targets_prediction: Optional[np.ndarray] = None,
        agent_traj_forecast: Optional[np.ndarray] = None,
        scale: Optional[float] = None
    ) -> plt.Figure:
        """
        Visualizes scenario with:
            - list of polylines
            - Initial anchors
            - Corrected anchors
            - Ground truth point
            - Trajectories (history and ground truth)

        Args:
            fig: Figure (if None then new one is created otherwise old one is cleared)
                Make sure to always use same figure instead of creating multiple ones
            targets_prediction: Targets prediction
            agent_traj_forecast: Forecast trajectories
            scale: Scale map

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        # Plot polylines
        sorted_polylines = sorted(self.polylines, key=lambda x: ObjectType.from_one_hot(x[0, 4:]).value, reverse=True)
        sorted_polylines = [p * scale for p in sorted_polylines] if scale is not None else sorted_polylines
        for polyline in sorted_polylines:
            for point in polyline:
                object_type = ObjectType.from_one_hot(point[4:])
                plt.arrow(x=point[0], y=point[1], dx=point[2]-point[0], dy=point[3]-point[1],
                          color=object_type.color, label=object_type.label, length_includes_head=True,
                          head_width=0.02, head_length=0.02, zorder=5)

        if agent_traj_forecast is not None:
            if len(agent_traj_forecast.shape) == 2:
                # In case of single forecasts, reshape it as (1, traj_length, 2)
                agent_traj_forecast = agent_traj_forecast.reshape(1, *agent_traj_forecast.shape)
            assert len(agent_traj_forecast.shape) == 3, 'Invalid agent forecast shape!'
            agent_traj_forecast = agent_traj_forecast * scale if scale is not None else agent_traj_forecast

            n_forecasts = agent_traj_forecast.shape[0]
            for f_index in range(n_forecasts):
                plt.plot(agent_traj_forecast[f_index, :, 0], agent_traj_forecast[f_index, :, 1],
                         color='lime', linewidth=5, label='forecast', zorder=10)

        agent_traj_gt = self.agent_traj_gt * scale if scale is not None else self.agent_traj_gt
        plt.plot(agent_traj_gt[:, 0], agent_traj_gt[:, 1], color='darkgreen', linewidth=5, label='ground truth', zorder=10)

        if targets_prediction is not None:
            # Plot prediction target points
            targets_prediction = targets_prediction * scale if scale is not None else targets_prediction
            plt.scatter(targets_prediction[:, 0], targets_prediction[:, 1],
                        color='slateblue', label='target predictions', s=200, zorder=10)

        # Plot anchor points and ground truth target point
        anchors = self.anchors * scale if scale is not None else self.anchors
        ground_truth_point = self.ground_truth_point * scale if scale is not None else self.ground_truth_point

        plt.scatter(anchors[:, 0], anchors[:, 1], color='purple', label='anchors', zorder=20)
        plt.scatter([ground_truth_point[0]], [ground_truth_point[1]],
                    color='pink', label='ground truth target', s=200, zorder=20)

        # set title and axis info
        plt.title(f'Graph Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        return fig

