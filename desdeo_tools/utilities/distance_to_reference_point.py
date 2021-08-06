from typing import Tuple, Type
import numpy as np
from desdeo_tools.scalarization import SimpleASF


def distance_to_reference_point(obj: np.ndarray, reference_point: np.ndarray) -> Tuple:
    """ 
        Computes the closest solution to a reference point using achievement scalarizing function.
    Args:

        obj (np.ndarray): Array of the solutions. Should be 2d-array.
        reference_point (np.ndarray): The reference point array. Should be one dimensional array.

    Returns: 
        Tuple: Returns a tuple containing the closest solution to a reference point and the index of it in obj. 
    """
    asf = SimpleASF(obj)
    d = (np.Inf, 1)
    for i, k in enumerate(obj):
        i_d = asf(k, reference_point=reference_point)
        if i_d < d[0]:
            d = i_d, i

    return d
