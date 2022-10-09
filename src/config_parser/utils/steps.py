
import argparse
import os


SOURCE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
CONFIG_PATH = os.path.join(SOURCE_PATH, 'configs')
MODEL_PATH = os.path.join(SOURCE_PATH, 'model_storage')


def get_config_path() -> str:
    """
    Loads config path as command line argument
    Returns: config path
    """
    parser = argparse.ArgumentParser(description='Load Global Config')
    parser.add_argument('--cfg', default='configs/test.yaml', type=str, help='config_path')

    args = parser.parse_args()
    return os.path.join(SOURCE_PATH, args.cfg)


