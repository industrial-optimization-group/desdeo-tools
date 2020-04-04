"""Implements methods for solving scalar valued functions.

"""
from typing import Callable, Optional, Any, Tuple, Dict, Union

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution

from desdeo_tools.scalarization.Scalarizer import Scalarizer


class ScalarMethod:
    """A class the define and implement methods for minimizing scalar valued functions.
    """

    def __init__(self, method: Callable, method_args=None):
        """
        Args:
            method (Callable): A callable minimizer function which expects a
            callable scalar valued function to be minimized. The function should
            accept as its first argument a two dimensional numpy array and should
            return a dictionary with at least the keys: "x" the found optimal solution,
            "success" boolean indicating if the minimization was successfull,
            "message" a string of additional info.
            method_args (Dict, optional): Any other keyword arguments to be supplied
            to the method. Defaults to None.
        """
        self._method = method
        self._method_args = method_args

    def __call__(
        self,
        obj_fun: Callable,
        x0: np.ndarray,
        bounds: np.ndarray,
        constraint_evaluator: Callable,
    ) -> Dict:
        """Minimizes a scalar valued function.
        
        Args:
            obj_fun (Callable): A callable scalar valued function that
            accpepts a two dimensional numpy array as its first arguments.
            x0 (np.ndarray): An initial guess.
            bounds (np.ndarray): The upper and lower bounds for each variable
            accepted by obj_fun. Expects a 2D numpy array with each row
            representing the lower and upper bounds of a variable. The first column
            should contain the lower bounds and the last column the upper bounds.
            Use np.inf to indicate no bound.
            constraint_evaluator (Callable): Should accepts exactly the
            same arguments as obj_fun. Returns a scalar value for each constraint
            present. This scalar value should be positive if a constraint holds, and negative
            otherwise.
        
        Returns:
            Dict: A dictionary with at least the following entries: 'x' indicating the optimal
            variables found, 'fun' the optimal value of the optimized functoin, and 'success' a boolean
            indicating whether the optimizaton was conducted successfully.
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
        method: Optional[Union[ScalarMethod, str]] = None,
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
            method (Optional[Union[Callable, str]], optional): The optimization method the scalarizer
            should be minimized with. It should accepts as keyword the arguments 'buounds' and 
            'constraints' which will be used to pass it the bounds and constraint_evaluator.
            If none is supplied, uses the minimizer implemented in SciPy. Otherwise a str can be given
            to use one of the preset solvers available. Use the method 'get_presets' to get a list
            of available preset solvers.
            Defaults to None.
        """
        self.presets = ["scipy_minimize", "scipy_de"]

        self._scalarizer = scalarizer
        self._bounds = bounds
        self._constraint_evaluator = constraint_evaluator
        if (method is None) or (method == "scipy_minimize"):
            # scipy minimize
            self._use_scipy = True
            # Assuming the gradient reqruies evaluation of the
            # scalarized function with out of bounds variable values.
            self._bounds[:, 0] += 1e-6
            self._bounds[:, 1] -= 1e-6
            self._method = ScalarMethod(minimize)

        elif method == "scipy_de":
            # Scipy differential evolution
            self._use_scipy = True
            # Assuming the gradient reqruies evaluation of the
            # scalarized function with out of bounds variable values.
            # only relevant if the 'polish' option is set in scipy's DE
            self._bounds[:, 0] += 1e-6
            self._bounds[:, 1] -= 1e-6
            scipy_de_method = ScalarMethod(
                lambda x, _, **y: differential_evolution(x, **y),
                method_args={"polish": False},
            )
            self._method = scipy_de_method

        else:
            self._use_scipy = False
            self._method = method

    def get_presets(self):
        """Return the list of preset minimizers available.

        """
        return self.get_presets

    def minimize(self, x0: np.ndarray) -> Dict:
        """Minimizes the scalarizer given an initial guess x0.
        
        Args:
            x0 (np.ndarray): A numpy array containing an initial guess of variable values.

        Returns:
            Dict: A dictionary with at least the following entries: 'x' indicating the optimal
            variables found, 'fun' the optimal value of the optimized functoin, and 'success' a boolean
            indicating whether the optimizaton was conducted successfully.
        """
        if self._use_scipy:
            # create wrapper for the constraints to be used with scipy's minimize routine.
            # assuming that all constraints hold when they return a positive value.
            if self._constraint_evaluator is not None:
                scipy_cons = NonlinearConstraint(
                    self._constraint_evaluator, 0, np.inf
                )
            else:
                scipy_cons = ()

            res = self._method(
                self._scalarizer,
                x0,
                bounds=self._bounds,
                constraint_evaluator=scipy_cons,
            )

        else:
            res = self._method(
                self._scalarizer,
                x0,
                bounds=self._bounds,
                constraint_evaluator=self._constraint_evaluator,
            )

        return res


""" if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_tools.scalarization.Scalarizer import Scalarizer

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

    # res = scalarizer(np.array([[0.5, 0.5], [0.4, 0.4]]))
    # print(problem.get_variable_bounds())

    solver = ScalarMinimizer(
        scalarizer,
        problem.get_variable_bounds(),
        # lambda xs: problem.evaluate(xs).constraints,
        None,
        "scipy_de",
    )

    opt_res = solver.minimize(np.array([0.5, 0.5]))
    print(opt_res.x)
"""
