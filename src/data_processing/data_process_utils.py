import numpy as np


def approximate_agent_speed(agent_traj_hist: np.ndarray) -> np.ndarray:
    """
    Approximates agent speed from history trajectory

    Args:
        agent_traj_hist: Agent history trajectory

    Returns: Agent speed
    """
    next_obs = agent_traj_hist[1:, :2]
    prev_obs = agent_traj_hist[:-1, :2]
    diffs = next_obs - prev_obs
    speed = np.abs(np.sum(diffs, axis=0) / agent_traj_hist[:, 3].sum())  # Ignoring masked values

    assert speed.shape == (2,), f'Wrong shape: Expetced {(2,)} but found {speed.shape}'
    return speed
