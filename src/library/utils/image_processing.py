"""
Image processing functions
"""
import numpy as np


def create_gauss_kernel(size=5, sigma=1.):
    """
    Creates gauss kernel of given size

    Args:
        size: Square matrix size
        sigma: Deviation parameter

    Returns: Gauss Kernel of size: [size]x[size]
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
