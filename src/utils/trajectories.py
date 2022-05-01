import numpy as np
import enum
from typing import Tuple


class PadType(enum.Enum):
    """
    PAST - pad left side
    FUTURE - pad right side
    """
    PAST = 0
    FUTURE = 1


def pad_trajectory(traj: np.ndarray, length: int, pad_type: PadType) -> Tuple[np.ndarray, int]:
    n_missing_points = length - traj.shape[0]

    if pad_type == PadType.PAST:
        traj = np.vstack([np.tile(traj[:1, :], (n_missing_points, 1)), traj])
        padding_mask = np.array([([0] * n_missing_points) + ([1] * (length - n_missing_points))]).T
    elif pad_type == PadType.FUTURE:
        traj = np.vstack([traj, np.tile(traj[-1:, :], (n_missing_points, 1))])
        padding_mask = np.array([([1] * (length - n_missing_points)) + ([0] * n_missing_points)]).T
    else:
        assert False, 'Invalid program state!'

    return np.hstack([traj, padding_mask]), n_missing_points
