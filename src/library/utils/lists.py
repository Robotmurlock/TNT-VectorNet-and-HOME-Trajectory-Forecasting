"""
Lists collection functions
"""
from typing import List


def flatten(itemlist: List[list]) -> list:
    """
    Transforms list of lists to a single list
    Example: flatten([[1,2], [3,4], [5]]) = [1,2,3,4,5]

    Args:
        itemlist: List of lists

    Returns: list
    """
    return [item for sublist in itemlist for item in sublist]


def chunks(itemlist: list, n_chunks: int) -> List[list]:
    """
    Splits list into almost "equal" number of chunk - abs(len(chunk1) - len(chunk2)) <= 1 for any chunk1 and chunk2

    Args:
        itemlist: List
        n_chunks: Number of chunks to split list

    Returns: Chunks generator
    """
    if n_chunks < 1:
        raise ValueError(f'Minimum number of chunks is 1 but found {chunks}!')

    for i in range(n_chunks):
        yield itemlist[i::n_chunks]
