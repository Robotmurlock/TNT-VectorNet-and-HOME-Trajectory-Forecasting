from typing import List


def flatten(itemlist: List[list]) -> list:
    return [item for sublist in itemlist for item in sublist]


def chunks(itemlist: list, n_chunks: int) -> List[list]:
    if n_chunks < 1:
        raise ValueError(f'Minimum number of chunks is 1 but found {chunks}!')

    for i in range(n_chunks):
        yield itemlist[i::n_chunks]
