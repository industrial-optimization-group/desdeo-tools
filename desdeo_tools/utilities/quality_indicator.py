from numba import njit 
import numpy as np
import hvwfg as hv


@njit 
def epsilon_indicator(reference_front: np.ndarray, front: np.ndarray) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to. 
        Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        front (np.ndarray): The front that is compared. Should be set of arrays.

    Returns: 
        float: The factor by which the approximating front is worse than the reference front with respect to all 
        objectives.
    """
    eps = 0.0
    ref_len = len(reference_front)
    front_len = len(front)
    # number of objectives
    num_obj = len(front[0])

    for i in range(ref_len):
        for j in range(front_len):
            for k in range(num_obj):
                value = front[j][k] - reference_front[i][k]
                if value > eps:
                    eps = value

    return eps


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


if __name__=="__main__":
    x = np.array([[1, 0], [0.5, 0.5], [0, 1], [1.5, 0.75]])
    ref = np.array([[2.0,2.0]])
    print(epsilon_indicator(x, ref))
    print(hypervolume_indicator(x, ref))

    x_simple = np.array([[0.0,0.0]])
    ref_simple = np.array([[2.0,1.0]])
    print(epsilon_indicator(x_simple, ref_simple))

    obj = np.array([[0.3, 0.6,1.0],
                [0.4, 0.4,1.2],
                [0.6, 0.2, 0.3]])

    print(epsilon_indicator(obj, ref))
    ref_hv = np.array([[1.1, 1.1,1.1]])
    print(hypervolume_indicator(obj, ref_hv))
    ref_hv2 = np.array([[2.0, 2.0,2.0]])
    print(hypervolume_indicator(obj, ref_hv2))

    print("\n========= PERFORMANCE TEST ===========")
    objvalues = 1000
    objdims = 77
    solpoints = np.random.randint(1, objdims)
    print(solpoints)    

    x_hard = np.array(np.random.rand(objvalues,objdims))
    #print(x_hard)
    ref_hard = np.array(np.random.rand(solpoints, objdims))
    #print(ref_hard)
    ref_hard_vector = np.array(np.random.rand(1, objdims))
    print(epsilon_indicator(x_hard, ref_hard))
    print(hypervolume_indicator(x_hard, ref_hard_vector))
