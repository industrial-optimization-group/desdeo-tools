from desdeo_tools.solver.ScalarSolver import ScalarMethod, ScalarMinimizer

import pytest
import numpy as np


def simple_problem(xs: np.ndarray):
    return xs[0] * 2 - xs[1] + xs[2]


def simple_constr(xs: np.ndarray):
    if xs[0] > 0.2:
        con_1 = 1
    else:
        con_1 = -1

    if xs[2] < 0.2:
        con_2 = 1
    else:
        con_2 = -1

    return np.array([con_1, con_2])


def dummy_minimizer(fun, x0, bounds, constraints=None):
    res_dict = {}

    if constraints is not None:
        con_vals = constraints(x0)
        if np.all(con_vals > 0):
            res_dict["success"] = True
        else:
            res_dict["success"] = False
    else:
        res_dict["success"] = True

    res_dict["x"] = x0

    res_dict[
        "message"
    ] = "I just retruned the initial guess as the optimal solution."

    return res_dict


def test_dummy_no_cons():
    method = ScalarMethod(dummy_minimizer)
    solver = ScalarMinimizer(
        simple_problem, np.array([[0, 0, 0], [1, 1, 1]]), None, method
    )

    x0 = np.array([0.5, 0.5, 0.5])
    res = solver.minimize(x0)

    assert np.array_equal(res["x"], x0)
    assert res["success"]
    assert (
        res["message"]
        == "I just retruned the initial guess as the optimal solution."
    )


def test_dummy_cons():
    method = ScalarMethod(dummy_minimizer)
    solver = ScalarMinimizer(
        simple_problem, np.array([[0, 0, 0], [1, 1, 1]]), simple_constr, method
    )

    res = solver.minimize(np.array([0.5, 0.5, 0.1]))

    assert res["success"]

    res = solver.minimize(np.array([0.5, 0.5, 0.5]))

    assert not res["success"]


if __name__ == "__main__":
    method = ScalarMethod(dummy_minimizer)
    solver = ScalarMinimizer(
        simple_problem, np.array([[0, 0, 0], [1, 1, 1]]), simple_constr, method
    )

    res = solver.minimize(np.array([0.5, 0.1, 0.1]))
    print(res)
    res = solver.minimize(np.array([0.5, 0.1, 0.5]))
    print(res)
