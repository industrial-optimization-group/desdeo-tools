from numba import njit 
import numpy as np
import hvwfg as hv


@njit 
def epsilon_indicator(reference_front: np.array, front: np.array) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to. 
        Should be matrix of numerics.
        front (np.ndarray): The front that is compared. Should be vector, will be flattened to vector if matrix.

    Returns: 
        float: The factor by which the approximating front is worse than the reference front with respect to all 
        objectives.
    """
    max_value = 0.0
    front = front.reshape(-1)     
    ref_len = len(reference_front)
    front_len = len(front)

    for j in range(ref_len):
        for i in range(front_len):  
            value = front[i] - reference_front[j][i]
            if value > max_value:
                max_value = value

    return max_value


def hypervolume_indicator(reference_front: np.array, front: np.array) -> float:
    """ Computes the hypervolume-indicator between reference front and current approximating front.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to. 
        Should be matrix of numerics.
        front (np.ndarray): The front that is compared. Should be vector, will be flattened to vector if matrix.

    Returns: 
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    return hv.wfg(reference_front, front.reshape(-1))


if __name__=="__main__":

    x = np.array([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
    ref = np.array([2.0,2.0])
    print(epsilon_indicator(x, ref))

    x_simple = np.array([[0.0,0.0]])
    ref_simple = np.array([2.0,1.0])
    print(epsilon_indicator(x_simple, ref_simple))

    obj = np.array([[0.3, 0.6],
                [0.4, 0.4],
                [0.6, 0.2]])

    print(epsilon_indicator(obj, ref))
    ref_hv = np.array([1.1, 1.1])
    print(hypervolume_indicator(obj, ref_hv))
    ref_hv2 = np.array([2.0, 2.0])
    print(hypervolume_indicator(obj, ref_hv2))

    # non-dominated reference point for HV results in a error
    ref_hv_err = np.array([0.0,0.0])
    print("Should be an error", hypervolume_indicator(obj, ref_hv_err))

    print("\n========= PERFORMANCE TEST ===========")
    objvalues = 12029
    objdims = 33

    x_hard = np.array(np.random.rand(objvalues,objdims))
    ref_hard = np.array(np.random.rand(1, objdims))
    print(epsilon_indicator(x_hard, ref_hard))
    print(hypervolume_indicator(x_hard, ref_hard))
