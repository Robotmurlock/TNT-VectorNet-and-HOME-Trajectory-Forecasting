import os

import configparser
from utils import steps
from architectures.heatmap import HeatmapTrajectoryForecaster
from architectures.heatmap.loss import PixelFocalLoss
import conventions
from evaluation import eval_tmp

from datasets.heatmap_dataset import HeatmapOutputRasterScenarioDataset


def run(config: configparser.GlobalConfig):
    model = HeatmapTrajectoryForecaster(
        encoder_input_shape=(9, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=config.global_parameters.trajectory_history_window_length,

        n_targets=6,
        radius=2,
        device='cuda:0'
    )
    model.load_weights(
        heatmap_estimator_path=os.path.join(config.global_path, 'model_storage', 'heatmap_targets', 'last.ckpt'),
        trajectory_forecater_path=os.path.join(config.global_path, 'model_storage', 'heatmap_tf', 'last.ckpt')
    )
    model.eval()

    outputs_path = os.path.join(config.global_path, 'test_data/raster_result')
    for split_name in conventions.SPLIT_NAMES:
        dataset_heatmap = HeatmapOutputRasterScenarioDataset(config, split=split_name)
        eval_tmp.evaluate(
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
