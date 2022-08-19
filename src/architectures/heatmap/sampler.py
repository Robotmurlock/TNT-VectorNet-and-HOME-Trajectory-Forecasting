import torch
import torch.nn as nn
import numpy as np
from typing import Union
from time import time


class ModalitySampler(nn.Module):
    def __init__(self, n_targets: int, radius: int, device: Union[str, torch.device], swap_rc: bool = True):
        super(ModalitySampler, self).__init__()
        self._n_targets = n_targets
        self._radius = radius
        self._reclen = 2*self._radius+1
        self._device = device
        self._swap_rc = swap_rc

    def _init_rec_sum(self, row: int, heatmap: torch.Tensor) -> float:
        rec_sum = 0.0
        for ri in range(self._reclen):
            for rj in range(self._reclen):
                rec_sum += heatmap[row+ri, rj]

        return rec_sum

    def _next_rec_sum(self, rec_sum: float, row: int, col: int, heatmap: np.ndarray) -> float:
        for ri in range(self._reclen):
            rec_sum += heatmap[row+ri, col+self._reclen] - heatmap[row+ri, col]
        return rec_sum

    def _clear_rec(self, row: int, col: int, heatmap: np.ndarray) -> np.ndarray:
        for ri in range(self._reclen):
            for rj in range(self._reclen):
                heatmap[row+ri, col+rj] = 0.0

        return heatmap

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        batch_size = heatmap.shape[0]
        hm_all = heatmap.detach().cpu().numpy().copy()
        result_all = []

        for index in range(batch_size):
            hm = hm_all[index].squeeze(0)

            n_rows, n_cols = hm.shape
            result = []

            for _ in range(self._n_targets):
                best_sum, best_row, best_col = None, None, None
                for row in range(n_rows-self._reclen):
                    curr_sum = self._init_rec_sum(row, hm)
                    for col in range(n_cols-self._reclen):
                        if best_sum is None or curr_sum > best_sum:
                            best_sum = curr_sum
                            best_row = row
                            best_col = col

                        curr_sum = self._next_rec_sum(curr_sum, row, col, hm)

                coords = [best_row+self._radius, best_col+self._radius]
                if self._swap_rc:
                    coords = [coords[1], coords[0]]  # swap row/col coords
                result.append(coords)
                hm = self._clear_rec(best_row, best_col, hm)

            result_all.append(torch.tensor(result, dtype=torch.long))

        return torch.stack(result_all).to(self._device)


class TorchModalitySampler(nn.Module):
    def __init__(self, n_targets: int, radius: int, swap_rc: bool = True):
        super(TorchModalitySampler, self).__init__()
        self._n_targets = n_targets
        self._radius = radius
        self._reclen = 2*self._radius+1
        self._swap_rc = swap_rc

        # components
        self._avgpool = nn.AvgPool2d(kernel_size=self._reclen, stride=1)

    @torch.no_grad()
    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        batch_size = heatmap.shape[0]
        hm = torch.clone(heatmap)

        batch_result = []
        for batch_index in range(batch_size):
            result = []
            for _ in range(self._n_targets):
                agg = self._avgpool(hm[batch_index])[0]
                coords = (agg == torch.max(agg)).nonzero()[0]
                hm[batch_index, 0, coords[0]:coords[0]+self._reclen, coords[1]:coords[1]+self._reclen] = 0.0
                result.append(coords+self._radius)

            result = torch.stack(result)
            batch_result.append(result)

        batch_result = torch.stack(batch_result)
        if self._swap_rc:
            batch_result = batch_result[:, :, [1, 0]]
        return batch_result




class KMeansProbSampler(nn.Module):
    def __init__(self, n_targets: int, n_iterations: int):
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


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=100, std=16):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d


def test():
    size = 100
    iterations = 1000

    kernel = gkern(kernlen=size)
    modality_sampler = ModalitySampler(n_targets=6, radius=2, device='cpu', swap_rc=True)
    torch_sampler = TorchModalitySampler(n_targets=6, radius=2, swap_rc=True)

    start_time = time()
    for _ in range(iterations):
        heatmap = torch.abs(torch.randn(4, 1, size, size))
        heatmap = heatmap * kernel
        _ = modality_sampler(heatmap)
    print(f'Average Time: {(time() - start_time) / iterations:.2f}s')

    start_time = time()
    for _ in range(iterations):
        heatmap = torch.abs(torch.randn(4, 1, size, size))
        heatmap = heatmap * kernel
        _ = torch_sampler(heatmap)
    print(f'Average Time: {(time() - start_time) / iterations:.2f}s')


if __name__ == '__main__':
    test()
