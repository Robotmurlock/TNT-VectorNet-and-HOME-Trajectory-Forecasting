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

    @property
    def dirname(self) -> str:
        """
        Returns: Scenario dirname
        """
        return f'{self.city}_{self.id}'

    @property
    def inputs(self) -> List[torch.Tensor]:
        return [torch.tensor(polyline, dtype=torch.float32) for polyline in self.polylines]

    @property
    def ground_truth(self) -> torch.Tensor:
        return torch.tensor(self.agent_traj_gt, dtype=torch.float32)

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
        }

        for filename, data in catalog.items():
            filepath = os.path.join(scenario_path, filename)
            np.save(filepath, data)

        polylines_path = os.path.join(scenario_path, 'polylines')
        Path(polylines_path).mkdir(parents=True, exist_ok=True)
        for index, p in enumerate(self.polylines):
            filepath = os.path.join(polylines_path, f'polyline_{index}.npy')
            np.save(filepath, p)

    @classmethod
    def load(cls, path: str) -> 'GraphScenarioData':
        """
        Loads Scenario Data from given path

        Args:
            path: Path to load object from

        Returns: RasterizedScenarioData
        """
        objects_to_load = ['center_point',  'agent_traj_gt', 'objects_traj_gts']
        filename = os.path.basename(path)
        sequence_city, sequence_id = filename.split('_')
        catalog = {'city': sequence_city, 'id': sequence_id}

        for object_name in objects_to_load:
            object_path = os.path.join(path, f'{object_name}.npy')
            catalog[object_name] = np.load(object_path)

        polylines_path = os.path.join(path, 'polylines')
        polylines_filepaths = os.listdir(polylines_path)
        catalog['polylines'] = [np.load(os.path.join(polylines_path, pfp)) for pfp in polylines_filepaths]

        return cls(**catalog)

    def visualize(
        self,
        fig: Optional[plt.Figure] = None,
        agent_traj_forecast: Optional[np.ndarray] = None
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
            agent_traj_forecast: Forecast

        Returns: Figure
        """
        if fig is None:
            fig = plt.figure(figsize=(20, 14))
        else:
            fig.clf()

        sorted_polylines = sorted(self.polylines, key=lambda x: ObjectType.from_one_hot(x[0, 4:]).value, reverse=True)
        for polyline in sorted_polylines:
            for point in polyline:
                object_type = ObjectType.from_one_hot(point[4:])
                plt.arrow(x=point[0], y=point[1], dx=point[2]-point[0], dy=point[3]-point[1],
                          color=object_type.color, label=object_type.label, length_includes_head=True,
                          head_width=0.05, head_length=0.05)

        if agent_traj_forecast is not None:
            plt.plot(agent_traj_forecast[:, 0], agent_traj_forecast[:, 1], color='lime', linewidth=5, label='forecast')
            plt.plot(self.agent_traj_gt[:, 0], self.agent_traj_gt[:, 1], color='darkgreen', linewidth=5, label='ground truth')

        # set title and axis info
        plt.title(f'Graph Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        return fig

