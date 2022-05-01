from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from typing import Optional
import matplotlib.pyplot as plt


@dataclass
class ScenarioData:
    id: str
    center_point: np.ndarray
    agent_traj_hist: np.ndarray
    agent_traj_gt: np.ndarray
    objects_traj_hists: np.ndarray
    objects_traj_gts: np.ndarray
    lane_features: np.ndarray
    centerline_candidate_features: np.ndarray

    def save(self, path: str) -> None:
        scenario_path = os.path.join(path, self.id)
        Path(path).mkdir(parents=True, exist_ok=True)

        catalog = {
            'center_point.npy': self.center_point,
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
        objects_to_load = ['center_point', 'agent_traj_hist', 'agent_traj_gt', 'objects_traj_hists'
                           'objects_traj_gts' 'lane_features' 'centerline_candidate_features']
        catalog = {'id': os.path.basename(path)}

        for object_name in objects_to_load:
            object_path = os.path.join(path, f'{object_name}.npy')
            catalog[object_name] = np.load(object_path)

        return cls(**catalog)

    def visualize(self, fig: Optional[plt.Figure] = None) -> plt.Figure:
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
                     color='k', linestyle='dashed', label='candidate centerline')

        # plot agent
        plt.plot(self.agent_traj_hist[:, 0], self.agent_traj_hist[:, 1], color='green', linewidth=7, label='Agent history')
        plt.plot(self.agent_traj_gt[:, 0], self.agent_traj_gt[:, 1], color='lightgreen', linewidth=7, label='Agent Ground Truth')

        # plot objects
        for obj_index in range(self.objects_traj_hists.shape[0]):
            plt.plot(self.objects_traj_hists[obj_index, :, 0], self.objects_traj_hists[obj_index, :, 1], color='orange', linewidth=4,
                     label='Neighbor history')
            plt.plot(self.objects_traj_gts[obj_index, :, 0], self.objects_traj_gts[obj_index, :, 1], color='yellow', linewidth=4,
                     label='Neighbor Ground Truth')

        # remove duplicated labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        # set title and axis info
        plt.title(f'Scenario {self.id}')
        plt.xlabel('X')
        plt.ylabel('Y')

        return fig
