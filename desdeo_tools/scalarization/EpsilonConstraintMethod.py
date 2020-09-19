import numpy as np
from desdeo_tools.scalarization import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer
from typing import Optional, Callable


class EpsilonConstraintMethod:
    """A class to represent a class for scalarizing MOO problems using the epsilon
        constraint method.
    Attributes:
        objective_vector (np.ndarray): Values of objective functions.
        to_be_minimized (int): Integer representing which objective function
        should be minimized.
        epsilons (np.ndarray): Upper bounds chosen by the decison maker.
        constraints (Optional[Callable]): Other constraints, if existing.
    """

    def __init__(
            self, obj_fun: Callable, objective_vector: np.ndarray, to_be_minimized: int, epsilons: np.ndarray,
            constraints: Optional[Callable]
    ):
        """

        Args:
            constraints (object):
            to_be_minimized (int): integer representing which objective function
            should be minimized.
            epsilons (np.ndarray): Upper bounds chosen by the decison maker.
            constraints (Optional[Callable]): Other constraints, if existing.
        """
        self.obj_fun = obj_fun
        self.objective_vector = objective_vector
        self._to_be_minimized = to_be_minimized
        self.epsilons = epsilons
        self.constraints = constraints

    def get_constraints(self, xs) -> np.ndarray:
        """
        Returns values of constraints
        Args:
            xs (np.ndarray): Decision variables

        Returns: Values of constraint functions (both "original" constraints as well as epsilon constraints) in a vector.

        """
        xs = np.atleast_2d(xs)
        if self.constraints:
            c = self.constraints(xs)
        else:
            c = []
        # epsilon function values with current decision variables
        eps_functions = np.array(
            [self.obj_fun(xs)[0][i] for i in range(len(self.objective_vector[0])) if i != self._to_be_minimized])

        # epsilon constraint values
        e = np.array([-(f - v) for f, v in zip(eps_functions, self.epsilons)])
        return np.concatenate([c, e])

    def __call__(self, objective_vector: np.ndarray) -> float:
        """
        Returns the value of objective function to be minimized.
        Args:
            objective_vector (np.ndarray): Values of objective functions.

        Returns: Value of objective function to be minimized.

        """
        return objective_vector[0][self._to_be_minimized]


"""
TESTING THE METHOD
"""
if __name__ == "__main__":
    def volume(r, h):
        return np.pi * r ** 2 * h


    def area(r, h):
        return 2 * np.pi ** 2 + np.pi * r * h


    def objective(xs):
        # xs is a 2d array like, which has different values for r and h on its first and second columns respectively.
        xs = np.atleast_2d(xs)
        return np.stack((volume(xs[:, 0], xs[:, 1]), -area(xs[:, 0], xs[:, 1]))).T


    # bounds
    r_bounds = np.array([2.5, 15])
    h_bounds = np.array([10, 50])
    bounds = np.stack((r_bounds, h_bounds))


    # constraints

    def con_golden(xs):
        # constraints are defined in DESDEO in a way were a positive value indicates an agreement with a constraint, and
        # a negative one a disagreement.
        xs = np.atleast_2d(xs)
        return -(xs[:, 0] / xs[:, 1] - 1.618)


    obj_min = 0
    epsil = np.array([-2000])
    x0 = np.array([10, 11])
    o = objective(x0)
    xs = np.atleast_2d(x0)
    cons = -(xs[:, 0] / xs[:, 1] - 1.618)
    eps = EpsilonConstraintMethod(objective, o, obj_min, epsil, con_golden)
    c = eps.get_constraints(x0)
    print(c)

    scalarized_objective = Scalarizer(objective, eps)
    print(scalarized_objective)

    minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=eps.get_constraints, method=None)

    res = minimizer.minimize(x0)
    final_r, final_h = res["x"][0], res["x"][1]
    final_obj = objective(res["x"]).squeeze()
    final_V, final_A = final_obj[0], final_obj[1]

    print(f"Final cake specs: radius: {final_r}cm, height: {final_h}cm.")
    print(f"Final cake dimensions: volume: {final_V}, area: {-final_A}.")
    print(final_r / final_h)
    print(res)
