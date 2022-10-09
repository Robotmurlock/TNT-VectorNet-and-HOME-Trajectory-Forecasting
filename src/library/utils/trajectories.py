"""
Trajectory transformation functions
"""
import enum
from typing import Tuple, Optional, List

import numpy as np


class PadType(enum.Enum):
    """
    PAST - pad left/start side
    FUTURE - pad right/end side
    """
    PAST = 0
    FUTURE = 1


def pad_trajectory(traj: np.ndarray, length: int, pad_type: PadType) -> Tuple[np.ndarray, int]:
    """
    Pads trajectory with zeros (at start or end

    Args:
        traj: Input trajectory (any length)
        length: Final length of trajectory
        pad_type: Pad at start or end

    Returns:

    """
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


def normalize_polyline_log1p(polyline: np.ndarray, last_index: int) -> np.ndarray:
    """
    Normalizes trajectory using formula: sgn(x) * log(1+|x|)

    Args:
        polyline: Polyline
        last_index: Last index to normalize

    Returns: Normalized polyline
    """
    signs = np.sign(polyline[..., :last_index])
    polyline[..., :last_index] = signs * np.log2(1 + np.abs(polyline[..., :last_index]))
    return polyline


def denormalize_polyline_log1p(polyline: np.ndarray, last_index: int) -> np.ndarray:
    """
    Denormalizes trajectory using formula: sgn(x) * (e^|x|-1)

    Args:
        polyline: Normalized polyline
        last_index: Last index to denormalize

    Returns: Denormalized polyline
    """
    signs = np.sign(polyline[..., :last_index])
    polyline[..., :last_index] = signs * (np.exp(np.abs(polyline[..., :last_index])) - 1)
    return polyline


def normalize_polyline(polyline: np.ndarray, last_index: int, sigma: float) -> np.ndarray:
    """
    Normalizes trajectory using formula: x / sigma

    Args:
        polyline: Polyline
        last_index: Last index to normalize
        sigma: Sigma value

    Returns: Normalized polyline
    """
    polyline[..., :last_index] = polyline[..., :last_index] / sigma
    return polyline


def denormalize_polyline(polyline: np.ndarray, last_index: int, sigma: float) -> np.ndarray:
    """
    Denormalizes trajectory using formula: x * sigma

    Args:
        polyline: Normalized polyline
        last_index: Last index to denormalize
        sigma: Sigma value

    Returns: Denormalized polyline
    """
    polyline[..., :last_index] = polyline[..., :last_index] * sigma
    return polyline


def approximate_trajectory_velocity(trajectory: np.ndarray, absolute: bool = True, mask_index: Optional[int] = None) -> np.ndarray:
    """
    Approximates trajectory velocity from history trajectory

    Args:
        trajectory: history trajectory
        absolute: Return absolute value
        mask_index: Mask index position

    Returns: speed
    """
    next_obs = trajectory[1:, :2]
    prev_obs = trajectory[:-1, :2]
    diffs = next_obs - prev_obs
    n_points = trajectory.shape[0] if mask_index is None else trajectory[:, mask_index].sum()
    speed = np.sum(diffs, axis=0) / n_points
    speed = speed if not absolute else np.abs(speed)

    assert speed.shape == (2,), f'Wrong shape: Expetced {(2,)} but found {speed.shape}'
    return speed


def sample_velocities(
    raw_velocity: np.ndarray,
    intensity_mult: List[float],
    rotations: List[float],
    intensity_add: Optional[List[float]] = None
) -> List[np.ndarray]:
    """
    Sample velocitys from given intensity multipliers and agent angle rotations

    velocity sample formula
    |cos(r) -sin(r)| * t * |Vx|
    |sin(r) cos(r) |       |Vy|
    where r is rotation, t is time elapsed from last point in history trajectory
    and Vx and Vy are Velocity projections


    Args:
        raw_velocity: Velocity at last position in history trajectory
        intensity_mult: List of intensity multipliers
        rotations: List of rotations
        intensity_add: Adds base value to intensity

    Returns: List of sampled velocities
    """
    if intensity_add is None:
        intensity_add = [0.0]

    velocities = []
    for intmul in intensity_mult:
        for intadd in intensity_add:
            for r in rotations:
                theta = np.arctan2(raw_velocity[1], raw_velocity[0])  # direction angle
                x = raw_velocity[0] + np.cos(theta) * intadd  # add intadd proportional by angle
                y = raw_velocity[1] + np.sin(theta) * intadd
                velocity = intmul * np.array([
                    x * np.cos(r) - y * np.sin(r),
                    x * np.sin(r) + y * np.cos(r)
                ])
                velocities.append(velocity)

    return velocities


def calc_angle_to_y_axis(point: np.ndarray) -> float:
    """
    Calculates angle relative to positive y-axis

    Args:
        point: Point

    Returns: Angle
    """
    return np.pi / 2 - np.arctan2(point[1], point[0])


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate set of points by given angle

    Args:
        points: Points
        angle: Angle

    Returns: Rotated points
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    points[..., :2] = (rotation_matrix @ points[..., :2].T).T
    return points
