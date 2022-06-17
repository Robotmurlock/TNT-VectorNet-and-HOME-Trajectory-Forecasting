import os

import configparser
from utils import steps
from evaluation import eval
from datasets.vectornet_dataset import VectorNetScenarioDataset
from architectures.vectornet import TargetDrivenForecaster
import conventions


def run():
    config = configparser.config_from_yaml(steps.get_config_path())

    model = TargetDrivenForecaster.load_from_checkpoint(
        checkpoint_path=os.path.join(steps.SOURCE_PATH, config.evaluation.model_path, 'last.ckpt'),
        cluster_size=20,
        trajectory_length=30,
        polyline_features=14,
        n_targets=6
    )

    datasets_path = os.path.join(steps.SOURCE_PATH, config.evaluation.input_path)
    outputs_path = os.path.join(steps.SOURCE_PATH, config.evaluation.output_path)
    for split_name in conventions.SPLIT_NAMES:
        ds_path = os.path.join(datasets_path, split_name)

        dataset = VectorNetScenarioDataset(ds_path)
        eval.evaluate(
            model=model,
            dataset=dataset,
            output_path=os.path.join(outputs_path, split_name),
            device='cuda',
            visualize=config.evaluation.visualize,
            scale=config.graph.data_process.normalization_parameter,
            title=split_name
        )


if __name__ == '__main__':
    run()
