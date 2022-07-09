import os

import configparser
from utils import steps
from architectures.heatmap import HeatmapTrajectoryForecaster


def run(config: configparser.GlobalConfig):
    model = HeatmapTrajectoryForecaster(
        encoder_input_shape=(10, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=config.global_parameters.trajectory_history_window_length,

        n_targets=6,
        radius=2
    )
    model.load_weights(
        heatmap_estimator_path=os.path.join(config.global_path, 'model_storage', 'heatmap_targets', 'last.ckpt'),
        trajectory_forecater_path=os.path.join(config.global_path, 'model_storage', 'heatmap_tf', 'last.ckpt')
    )
    model.eval()


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
