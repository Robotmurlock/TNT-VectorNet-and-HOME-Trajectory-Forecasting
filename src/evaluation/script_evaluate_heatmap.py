import os

import configparser
from utils import steps
from architectures.heatmap import HeatmapTrajectoryForecaster
from architectures.heatmap.loss import PixelFocalLoss
import conventions
from evaluation import eval_home

from datasets.heatmap_dataset import HeatmapOutputRasterScenarioDataset


def run(config: configparser.GlobalConfig):
    heatmap_model_name = config.raster.train_heatmap.model_name
    forecaster_model_name = config.raster.train_tf.model_name

    model = HeatmapTrajectoryForecaster(
        encoder_input_shape=(9, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        trajectory_history_window_length=config.global_parameters.trajectory_history_window_length,
        trajectory_future_window_length=config.global_parameters.trajectory_future_window_length,

        sampler_targets=6,
        sampler_radius=1.5,
        sampler_upscale=2,

        heatmap_estimator_path=os.path.join(config.model_storage_path, 'home', 'heatmap_targets', heatmap_model_name, 'epoch=4-step=64285.ckpt'),
        trajectory_forecaster_path=os.path.join(config.model_storage_path, 'home', 'forecaster', forecaster_model_name, 'last.ckpt')
    )
    model.eval()

    outputs_path = os.path.join(config.global_path, 'raster_result')
    for split_name in conventions.SPLIT_NAMES:
        dataset_heatmap = HeatmapOutputRasterScenarioDataset(config, split=split_name)
        eval_home.evaluate(
            model=model,
            loss=PixelFocalLoss(),
            dataset=dataset_heatmap,
            output_path=os.path.join(outputs_path, split_name),
            device='cuda',
            visualize=True,
            scale=config.graph.data_process.normalization_parameter,
            title=split_name
        )


if __name__ == '__main__':
    run(configparser.config_from_yaml(steps.get_config_path()))
