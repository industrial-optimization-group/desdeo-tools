import numpy as np
from desdeo_tools.scalarization import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer


class EpsilonConstraintMethod:
    """A class to represent a class for scalarizing MOO problems using the epsilon
        constraint method.
    Attributes:
        epsilons (np.ndarray): The epsilon values to set as the upper limit
        for each objective when treated as a constraint.
        to_be_minimized (int): Integer representing which objective function
        should be minimized.
    """

    def __init__(
            self, epsilons: np.ndarray, to_be_minimized: int
    ):
        """

        Args:
            epsilons (np.ndarray): The epsilon values to set as the upper limit
            for each objective when treated as a constraint.
            to_be_minimized (int): integer representing which objective function
            should be minimized.
        """

        self._epsilons = epsilons
        self._to_be_minimized = to_be_minimized

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        return xs[self._to_be_minimized]



""" 
FOR TESTING THE METHOD
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


    def simple_sum(xs):
        xs = np.atleast_2d(xs)
        return np.sum(xs, axis=1)

    eps = EpsilonConstraintMethod(epsilons=[],to_be_minimized=0)
    scalarized_objective = Scalarizer(objective, eps)
    print(scalarized_objective)

    minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=con_golden, method=None)
    x0 = np.array([2.6, 11])
    sum_res = minimizer.minimize(x0)
    print(sum_res)
