import torch
from typing import List, Tuple

from architectures.base import BaseModel


class ConstantVelocityModel(BaseModel):
    """
    Baseline model - Calculates average velocity and propagates it to generate forecast
    """
    def __init__(self, *args, **kwargs):
        super(ConstantVelocityModel, self).__init__(*args, **kwargs)

    @staticmethod
    def object_velocity(trajectory: torch.Tensor) -> torch.Tensor:
        """
        Approximates trajectory velocity

        Args:
            trajectory: Trajectory (history)

        Returns: Approximated trajectory speed vector
        """
        next_obs = trajectory[1:, :2]  # ignore mask feature
        prev_obs = trajectory[:-1, :2]
        diffs = next_obs - prev_obs
        speed = torch.sum(diffs, dim=0) / trajectory[:, 2].sum()  # Ignoring masked values
        return speed

    def _trajectory_forecast(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Generates forecast for input trajectory

        Args:
            trajectory: Trajectory history

        Returns: Trajectory forecast
        """
        speed = self.object_velocity(trajectory)
        last_obs = trajectory[-1, :2]  # ignore mask feature

        prediction_list: List[torch.Tensor] = []
        for _ in range(self._prediction_length):
            last_obs = last_obs + speed
            prediction_list.append(last_obs)

        return torch.stack(prediction_list)

    def _predict(self, agent_features: torch.Tensor, object_features: torch.Tensor):
        """
        Predicts trajectories for one scenario (agents and objects)

        Args:
            agent_features: Agent features (trajectory)
            object_features: Multiple objects with features (trajectories)

        Returns: Forecasts for agent and all objects (neighbors)
        """
        agent_prediction = self._trajectory_forecast(agent_features)

        object_prediction_list: List[torch.Tensor] = []
        for object_index in range(object_features.shape[0]):
            object_prediction = self._trajectory_forecast(object_features[object_index, :, :])
            object_prediction_list.append(object_prediction)

        return agent_prediction, torch.stack(object_prediction_list)

    def forward(self, features: Tuple[torch.Tensor, ...], is_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_features, object_features, _, _ = features

        if is_batch:
            agent_prediction_list, objects_prediction_list = [], []
            for batch_index in range(agent_features.shape[0]):
                agent_pred, objects_pred = self._predict(agent_features[batch_index], object_features[batch_index])
                agent_prediction_list.append(agent_pred)
                objects_prediction_list.append(objects_pred)

            return torch.stack(agent_prediction_list), torch.stack(objects_prediction_list)

        return self._predict(agent_features, object_features)

    def forecast(self, features: Tuple[torch.Tensor, ...], is_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(features, is_batch=is_batch)
