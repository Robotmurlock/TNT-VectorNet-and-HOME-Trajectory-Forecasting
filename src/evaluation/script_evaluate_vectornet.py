import os
import yaml

import configparser
from utils import steps
from evaluation import eval
from datasets.vectornet_dataset import VectorNetScenarioDataset
from architectures.factory import model_factory
import conventions


def run():
    config = configparser.config_from_yaml(steps.get_config_path())

    with open(os.path.join(steps.SOURCE_PATH, config.model.config_path), 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    model = model_factory(config.model.name, params=params)
    model.load_state(os.path.join(steps.SOURCE_PATH, config.evaluation.model_path))

    datasets_path = os.path.join(steps.SOURCE_PATH, config.evaluation.input_path)
    outputs_path = os.path.join(steps.SOURCE_PATH, config.evaluation.output_path)
    for split_name in conventions.SPLIT_NAMES:
        ds_path = os.path.join(datasets_path, split_name)

        dataset = VectorNetScenarioDataset(ds_path, device='cuda')
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
