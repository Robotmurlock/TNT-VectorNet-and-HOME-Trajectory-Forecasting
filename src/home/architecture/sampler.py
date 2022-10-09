"""
Algorithms for trajectory end point sampling from estimated heatmap
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class TorchModalitySampler(nn.Module):
    def __init__(self, n_targets: int, radius: float, upscale: int = 1, swap_rc: bool = True):
        """
        Greedy algorithm to sample N targets from heatmap with the highest area probability
        (Optimized version of ModalitySampler - up to 50 times faster)

        Args:
            n_targets: Number of targets to sample
            radius: Target radius to sum probality of heatmap
            swap_rc: Swap coordinates axis in output
        """
        super(TorchModalitySampler, self).__init__()
        self._n_targets = n_targets
        self._upscale = upscale
        self._radius = round(radius*upscale)
        self._reclen = 2*self._radius+1
        self._swap_rc = swap_rc
        self._square = radius*radius

        # components
        self._avgpool = nn.AvgPool2d(kernel_size=self._reclen, stride=1)

    @torch.no_grad()
    def forward(self, heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = heatmap.shape[0]
        hm = torch.clone(heatmap)
        hm = TF.resize(hm, size=[2*hm.shape[-1], 2*hm.shape[-1]])

        batch_end_points, batch_confidences = [], []
        for batch_index in range(batch_size):
            end_points, confidences = [], []
            for _ in range(self._n_targets):
                agg = self._avgpool(hm[batch_index])[0]
                agg_max = torch.max(agg)
                coords = (agg == agg_max).nonzero()[0]
                hm[batch_index, 0, coords[0]:coords[0]+self._reclen, coords[1]:coords[1]+self._reclen] = 0.0
                end_points.append((coords+self._radius) / self._upscale)
                confidences.append(agg_max.detach().item()*self._square)

            end_points = torch.stack(end_points)
            confidences = torch.tensor(confidences, dtype=torch.float32)
            batch_end_points.append(end_points)
            batch_confidences.append(confidences)

        final_end_points = torch.stack(batch_end_points)
        final_confidences = torch.stack(batch_confidences)

        if self._swap_rc:
            final_end_points = final_end_points[:, :, [1, 0]]
        return final_end_points, final_confidences


class KMeansProbSampler(nn.Module):
    def __init__(self, n_targets: int, n_iterations: int):
        """
        FIXME: deprecated

        Args:
            n_targets:
            n_iterations:
        """
        super(KMeansProbSampler, self).__init__()
        self._n_targets = n_targets
        self._n_iterations = n_iterations

    def forward(self, clusters: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        hm = heatmap.numpy().copy()
        if len(hm.shape) == 3:
            assert hm.shape[0] == 1, f'Invalid heamap shape: {hm.shape}'
            hm = hm.squeeze(0)

        cs = clusters.numpy()

        for _ in range(self._n_iterations):
            cs = self._update_clusters(cs, hm)

        return torch.tensor(cs, dtype=torch.float32)

    def _update_clusters(self, clusters: np.ndarray, hm: np.ndarray) -> np.ndarray:
        n_rows, n_cols = hm.shape

        new_clusters = np.zeros_like(clusters, dtype=np.float32)

        for row in range(n_rows):
            for col in range(n_cols):
                best_cluster, best_dist, best_coords = None, None, None
                for c in range(clusters.shape[0]):
                    dist = max(1, np.sqrt((clusters[c, 0] - row) ** 2 + (clusters[c, 1] - col) ** 2))
                    if best_dist is None or dist < best_dist:
                        best_cluster = c
                        best_dist = dist
                        best_coords = [row, col]

                new_clusters[best_cluster, :] += np.array(best_coords, dtype=np.float32) * hm[row, col] / best_dist

        return new_clusters