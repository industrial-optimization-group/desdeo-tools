"""Implements methods for solving scalar valued functions.

"""
from typing import Callable, Optional, Any, Tuple

import numpy as np
from scipy.optimize import minimize

from desdeo_tools.scalarization.Scalarizer import Scalarizer


class ScalarMethod:
    """A class the define and implement methods for minimizing scalar valued functions.
    """

    def __init__(self, method: Callable, method_args=None):
        """
        Args:
            method (Callable): A callable minimizer function which expects a
            callable scalar valued function to be minimized. The function should
            accept as its first argument a two dimensional numpy array.
            method_args ([type], optional): Any other arguments to be
            supllied to the method. Defaults to None.
        """
        self._method = method
        self._method_args = method_args

    def __call__(
        self,
        obj_fun: Callable,
        x0: np.ndarray,
        bounds: np.ndarray,
        constraint_evaluator: Callable,
    ) -> Tuple[np.ndarray, np.float]:
        """Minimizes a scalar valued function.
        
        Args:
            obj_fun (Callable): A callable scalar valued function that
            accpepts a two dimensional numpy array as its first arguments.
        
        Returns:
            Tuple[np.ndarray, np.float]: The optimal variables and funtion
            value found.
        """
        if self._method_args is not None:
            res = self._method(
                obj_fun,
                x0,
                bounds=bounds,
                constraints=constraint_evaluator,
                **self._method_args
            )
        else:
            res = self._method(
                obj_fun, x0, bounds=bounds, constraints=constraint_evaluator
            )

        return res


class ScalarMinimizer:
    """Implements a class for minimizing scalar valued functions with bounds set for the
    variables, and constraints.
    """

    def __init__(
        self,
        scalarizer: Scalarizer,
        bounds: np.ndarray,
        constraint_evaluator: Callable = None,
        method: ScalarMethod = None,
    ):
        """ 
        Args:
            scalarizer (Scalarizer): A Scalarizer to be minimized.
            bounds (np.ndarray): The bounds of the independent variables the
            scalarizer is called with.
            constraint_evaluator (Callable, optional): A Callable which
            representing a vector valued constraint function. The array the constraint
            function returns should be two dimensional with each row corresponding to the
            constraint function values when evaluated. A value of less than zero is
            understood as a non valid constraint. Defaults to None.
            method ([type], optional): The optimization method the scalarizer
            should be minimized with. It should accepts as keyword arguments 'buounds' and 
            'constraints' which will be used to pass it the bounds and constraint_evaluator.
            Defaults to None.
        """
        self._scalarizer = scalarizer
        self._bounds = bounds
        self._constraint_evaluator = constraint_evaluator
        if method is None:
            self._method = ScalarMethod(minimize)
        else:
            self._method = method

    def minimize(self, x0: np.ndarray):
        pass
