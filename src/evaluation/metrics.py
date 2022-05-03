import torch


def ADE(forecast: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Calculates Average Displacement Error for given forecast and ground truth
    Average Displacement Error is average error of each step

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
    Calculates Final Displacement Error for given forecast and ground truth
    Final Displacement Error is error of last step

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

