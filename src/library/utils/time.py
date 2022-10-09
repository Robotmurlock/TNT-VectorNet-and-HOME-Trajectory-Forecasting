"""
Script time estimation functions
"""
import time
from typing import Callable


def timeit(f: Callable) -> Callable:
    """
    Decorated functions prints its execution time.
    :param f: Function
    :return: Decorated function
    """
    def timed(*args, **kw):

        start_time = time.time()
        result = f(*args, **kw)
        print("--- %s seconds ---" % (time.time() - start_time))
        return result

    return timed