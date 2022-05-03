import argparse
import os


SOURCE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def get_config_path() -> str:
    """
    Loads config path as command line argument
    Returns: config path
    """
    parser = argparse.ArgumentParser(description='Load Global Config')
    parser.add_argument('--config', default='configs/test.yaml', type=str, help='config_path')

    args = parser.parse_args()
    return os.path.join(SOURCE_PATH, args.config)


