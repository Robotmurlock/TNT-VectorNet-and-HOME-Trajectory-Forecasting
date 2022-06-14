import os
import yaml

import configparser
from utils import steps
from evaluation import eval
from datasets.vectornet_dataset import VectorNetScenarioDataset
from architectures.factory import model_factory


def run():
    config = configparser.config_from_yaml(steps.get_config_path())

    with open(os.path.join(steps.SOURCE_PATH, config.model.config_path), 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    model = model_factory(config.model.name, params=params)
    model.load_state(os.path.join(steps.SOURCE_PATH, config.evaluation.model_path))
    dataset = VectorNetScenarioDataset(os.path.join(steps.SOURCE_PATH, config.evaluation.input_path), device='cuda')
    eval.evaluate(
        model=model,
        dataset=dataset,
        output_path=os.path.join(steps.SOURCE_PATH, config.evaluation.output_path),
        device='cuda',
        visualize=config.evaluation.visualize,
        scale=25.0
    )


if __name__ == '__main__':
    run()
