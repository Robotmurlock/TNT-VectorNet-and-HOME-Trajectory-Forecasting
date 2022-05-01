from typing import List, Any


def flatten(itemlist: List[Any]) -> list:
    return [item for sublist in itemlist for item in sublist]
