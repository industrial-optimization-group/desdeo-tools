"""Implements methods for scalarizing vector valued function.

"""
from typing import Callable, Optional, Any

import numpy as np


class Scalarizer:
    """Implements a class for scalarizing vector valued functions with a
    given scalarization function.
    """

    # How to scalarize?
    # What can be expected from the function?

    def __init__(
        self,
        evaluator: Callable,
        scalarizer: Callable,
        evaluator_args=None,
        scalarizer_args=None,
    ):
        """
        Args:
            evaluator (Callable): A Callable object returning numpy array.
            scalarizer (Callable): A function which should accepts as its
            arguments the output of evaluator and return a single value.
            evaluator_args (Any, optional): Optional arguments to be passed to
            evaluator. Defaults to None.
            scalarizer_args (Any, optional): Optional arguments to be passed to
            scalarizer. Defaults to None.
        """
        self._evaluator = evaluator
        self._scalarizer = scalarizer
        self._evaluator_args = evaluator_args
        self._scalarizer_args = scalarizer_args

    def evaluate(self, xs: np.ndarray) -> np.ndarray:
        """Evaluates the scalarized function with the given arguments and
        returns a scalar value for each vector on variables given in a numpy
        array.
        
        Args:
            xs (np.ndarray): A 2D numpy array containing vectors of variables
            on each of its rows.
        
        Returns:
            np.ndarray: A 1D numpy array with the values returne by the
            scalarizer for each row in xs.
        """
        if self._evaluator_args is not None:
            res_eval = self._evaluator(xs, **self._evaluator_args)
        else:
            res_eval = self._evaluator(xs)

        if self._scalarizer_args is not None:
            res_scal = self._scalarizer(res_eval, **self._scalarizer_args)
        else:
            res_scal = self._scalarizer(res_eval)

        return res_scal

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        """Wrapper to the evaluate method.
        """
        return self.evaluate(xs)


if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder

    # create the problem
    def f_1(x):
        res = 4.07 + 2.27 * x[:, 0]
        return -res

    def f_2(x):
        res = (
            2.60
            + 0.03 * x[:, 0]
            + 0.02 * x[:, 1]
            + 0.01 / (1.39 - x[:, 0] ** 2)
            + 0.30 / (1.39 - x[:, 1] ** 2)
        )
        return -res

    def f_3(x):
        res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
        return -res

    def f_4(x):
        res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
        return -res

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)
    f3 = _ScalarObjective(name="f3", evaluator=f_3)
    f4 = _ScalarObjective(name="f4", evaluator=f_4)
    f5 = _ScalarObjective(name="f5", evaluator=f_5)

    varsl = variable_builder(
        ["x_1", "x_2"],
        initial_values=[0.5, 0.5],
        lower_bounds=[0.3, 0.3],
        upper_bounds=[1.0, 1.0],
    )

    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5])

    scalarizer = Scalarizer(
        lambda xs: problem.evaluate(xs).objectives,
        lambda ys: np.sum(ys, axis=1),
    )

    res = scalarizer.evaluate(np.array([[0.5, 0.5], [0.4, 0.4]]))
    print(res)

