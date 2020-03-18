import pytest
import numpy as np

from desdeo_tools.scalarization.Scalarizer import Scalarizer


def simple_vector_valued_fun(xs: np.ndarray, extra: int = 0):
    """A simple vector valued function for testing.
    
    Args:
        xs (np.ndarray): A 2D numpy array with argument vectors as its rows.
        Each vector consists of four values.
    
    Returns:
        np.ndarray: A 2D array with function evaluation results for each of
        the argument vectors on its rows. Each row contains three values.
    """
    f1 = xs[:, 0] + xs[:, 1] + extra
    f2 = xs[:, 1] - xs[:, 2] + extra
    f3 = xs[:, 2] * xs[:, 3] + extra

    return np.vstack((f1, f2, f3)).T


def simple_scalarizer(ys: np.ndarray, extra: int = 0):
    res = np.sum(ys, axis=1)

    if extra > 0:
        return -res

    else:
        return res


def test_scalarizer_simple():
    scalarizer = Scalarizer(simple_vector_valued_fun, simple_scalarizer)
    xs = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1, 5, 7, 3]])

    res = scalarizer.evaluate(xs)

    assert np.array_equal(res, [14, 60, 25])


def test_scalarizer_simple_with_arg():
    scalarizer = Scalarizer(
        simple_vector_valued_fun,
        simple_scalarizer,
        evaluator_args={"extra": 1},
        scalarizer_args={"extra": 4},
    )
    xs = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [1, 5, 7, 3]])

    res = scalarizer.evaluate(xs)

    assert np.array_equal(res, [-17, -63, -28])
