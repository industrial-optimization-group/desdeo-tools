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


def epsilon_indicator_ndims(reference_front: np.ndarray, front: np.ndarray) -> list:
    """ Computes the additive epsilon-indicator between reference front and current one-dimensional vector of front.
    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be one-dimensional array.
    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives.
    """
    eps_list = np.array(np.zeros(reference_front.shape[0]))
    for i in np.arange(reference_front.shape[0]):
        eps_list[i] = np.max(front - reference_front[i])
    return eps_list


def preference_indicator(reference_front: np.ndarray, front: np.ndarray, min_asf_value: float, ref_point: np.ndarray, delta: float) -> float:
    """ Computes the preference-based quality indicator.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        front (np.ndarray): The front that is compared. Should be one-dimensional array with the same shape as
        reference_front.
        ref_point (np.ndarray): The reference point should be same shape as front.
        min_asf_value (float): Minimum value of achievement scalarization of the reference_front. Used in normalization.
        delta (float): The spesifity delta allows to set the amplification of the indicator to be closer or farther 
        from the reference point. Smaller delta means that all solutions are in smaller range around the reference
        point.

    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives taking into account the reference point given and spesifity.
    """
    front_asf = SimpleASF(np.ones_like(front))
    norm = front_asf(front, reference_point=ref_point) + delta - min_asf_value
    return epsilon_indicator(reference_front, front) / norm


def hypervolume_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the hypervolume-indicator between reference front and current approximating point.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be 1D array.

    Returns:
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    ref = np.asarray(reference_front, dtype='double') # hv.wfg needs datatype to be double
    fr = np.asarray(front, dtype='double')
    return hv.wfg(ref, fr)


if __name__=="__main__":

    po_front = np.asarray([[1.0,0],[0.5,0.5], [0,1.0], [2, -1], [0,0]])
    sol1 = [2,2] # cant be better than po front, min is zero
    sol = np.asarray(sol1)

    print("eps indi value")
    print(epsilon_indicator(po_front[0], sol))
    print(epsilon_indicator(po_front[1], sol))
    print(epsilon_indicator(po_front[2], sol))
    print(epsilon_indicator(po_front[3], sol))
    print(epsilon_indicator(po_front[4], sol))
    print("ndims")
    print(epsilon_indicator_ndims(po_front, sol))
    print(hypervolume_indicator(po_front, sol))

