import os
import yaml

import configparser
from utils import steps
from evaluation import eval
from datasets.dataset import ScenarioDataset
from architectures.factory import model_factory


def run():
    config = configparser.config_from_yaml(steps.get_config_path())

    with open(os.path.join(steps.SOURCE_PATH, config.model.config_path), 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    model = model_factory(config.model.name, params=params)
    dataset = ScenarioDataset(os.path.join(steps.SOURCE_PATH, config.evaluation.input_path))
    eval.evaluate(model, dataset, os.path.join(steps.SOURCE_PATH, config.evaluation.output_path), visualize=config.evaluation.visualize)


if __name__ == '__main__':
    run()
