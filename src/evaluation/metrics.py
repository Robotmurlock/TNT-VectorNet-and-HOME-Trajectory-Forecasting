import torch
from typing import Tuple


LOW_PROB_THRESHOLD_FOR_METRICS = torch.tensor(0.05, dtype=torch.float32)


def ADE(forecast: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error (ADE) is average L2 error of each step

    Args:
        forecast: Forecast trajectory
        ground_truth: Ground Truth Trajectory

    Returns: ADE
    """
    return torch.mean(
        torch.sqrt(
            torch.float_power(ground_truth[..., :, 0] - forecast[..., :, 0], 2)
            + torch.float_power(ground_truth[..., :, 1] - forecast[..., :, 1], 2)
        )
    )


def FDE(forecast: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error (FDE) is L2 error of last step

    Args:
        forecast: Forecast trajectory
        ground_truth: Ground Truth Trajectory

    Returns: FDE
    """
    return torch.mean(
        torch.sqrt(
            torch.float_power(ground_truth[..., -1, 0] - forecast[..., -1, 0], 2)
            + torch.float_power(ground_truth[..., -1, 1] - forecast[..., -1, 1], 2)
        )
    )


def minFDE(forecasts: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimum Final Displacement Error (minFDE) is minimum FDE over all forecasts

    Args:
        forecasts: Forecast trajectory
        ground_truth: Ground Truth Trajectory

    Returns: minFDE
    """
    fde = torch.sqrt((forecasts[..., -1, 0] - ground_truth[..., -1, 0]) ** 2 + (forecasts[..., -1, 1] - ground_truth[..., -1, 1]) ** 2)
    if len(fde.shape) == 1:
        # normal
        min_fde_index = torch.argmin(fde, dim=0)
        return torch.mean(fde[min_fde_index]), min_fde_index
    elif len(fde.shape) == 2:
        # batched
        min_fde_index = torch.argmin(fde, dim=1)
        return torch.mean(fde[:, min_fde_index]), min_fde_index
    else:
        # invalid
        raise ValueError(f'Invalid tensor shape: {fde.shape}')


def minADE(forecasts: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimum Average Displacement Error (minADE) is ADE of trajectory with min FDE

    Args:
        forecasts: Forecast trajectory
        ground_truth: Ground Truth Trajectory

    Returns: minADE
    """
    _, min_fde_index = minFDE(forecasts, ground_truth)
    ade = torch.mean(torch.sqrt((forecasts[..., 0] - ground_truth[..., 0]) ** 2 + (forecasts[..., 1] - ground_truth[..., 1]) ** 2), dim=-1)
    return torch.mean(ade[:, min_fde_index] if len(ade.shape) == 2 else ade[min_fde_index]), min_fde_index


def probaMinFDE(forecasts: torch.Tensor, probas: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Probability Minimum Final Displacement Error (minFDE) is minimum FDE over all forecasts
    with probability score for chosen forecast

    Args:
        forecasts: Forecast trajectory
        probas: Forecasts probabilities
        ground_truth: Ground Truth Trajectory

    Returns: minFDE
    """
    min_fde, min_fde_index = minFDE(forecasts, ground_truth)
    proba_score = -torch.log(torch.maximum(probas[min_fde_index], LOW_PROB_THRESHOLD_FOR_METRICS))
    return min_fde + proba_score


def probaMinADE(forecasts: torch.Tensor, probas: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Minimum Average Displacement Error (minADE) is ADE of trajectory with min FDE
    with probability score for chosen forecast

    Args:
        forecasts: Forecast trajectory
        probas: Forecast probabilities
        ground_truth: Ground Truth Trajectory

    Returns: minADE
    """
    min_ade, min_ade_index = minADE(forecasts, ground_truth)
    proba_score = -torch.log(torch.maximum(probas[min_ade_index], LOW_PROB_THRESHOLD_FOR_METRICS))
    return min_ade + proba_score
