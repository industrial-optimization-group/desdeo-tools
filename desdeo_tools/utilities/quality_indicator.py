from numba import njit
import numpy as np
import hvwfg as hv
from desdeo_tools.scalarization import SimpleASF


@njit()
def epsilon_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        front (np.ndarray): The front that is compared. Should be one-dimensional array with the same shape as
        reference_front.

    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives.
    """
    eps = 0.0
    for i in range(reference_front.size):
        value = front[i] - reference_front[i]
        if value > eps:
            eps = value
    return eps


@njit()
def epsilon_indicator_ndims(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.
    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be one-dimensional array.
    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives.
    """

    eps = 0.0
    ref_len = reference_front.shape[0]
    front_len = front.shape[0]
    value = 0

    for i in np.arange(ref_len):
        for j in np.arange(front_len):
            value = front[j] - reference_front[i][j]
            if value > eps:
                eps = value

    return eps


def preference_indicator(reference_front: np.ndarray, front: np.ndarray, ref_point: np.ndarray, delta: float) -> float:
    """ Computes the preference-based quality indicator.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        front (np.ndarray): The front that is compared. Should be one-dimensional array with the same shape as
        reference_front.
        ref_point (np.ndarray): The reference point should be same shape as front.
        delta (float): The spesifity delta allows to set the amplification of the indicator to be closer or farther 
        from the reference point. Smaller delta means that all solutions are in smaller range around the reference
        point.

    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives taking into account the reference point given and spesifity.
    """
    ref_front_asf = SimpleASF(reference_front)
    front_asf = SimpleASF(front)
    norm = front_asf(front, reference_point=ref_point) + delta - np.min(ref_front_asf(reference_front, reference_point=ref_point))
    return epsilon_indicator(reference_front, front)/norm


def hypervolume_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the hypervolume-indicator between reference front and current approximating point.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be 2D array.

    Returns:
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    return hv.wfg(reference_front, front.reshape(-1))

