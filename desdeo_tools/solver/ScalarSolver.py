"""Implements methods for solving scalar valued functions.
"""
import numpy as np
import os
import rbfopt

from typing import Callable, Dict, Optional, Union
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer, Scalarizer
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from desdeo_tools.scalarization.ASF import PointMethodASF
#from desdeo_problem import variable_builder, ScalarObjective, MOProblem
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer


class ScalarSolverException(Exception):
    pass

class ScalarMethod:
    """A class the define and implement methods for minimizing scalar valued functions.
    """

    def __init__(self, method: Callable, method_args=None, use_scipy: Optional[bool] = False):
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
            use_scipy (Optional[bool]): Whether to use scipy's NonLinearConstraint to
                handle the constraints.
        """
        self._method = method
        self._method_args = method_args
        self._use_scipy = use_scipy

    def __call__(self, obj_fun: Callable, x0: np.ndarray, bounds: np.ndarray, constraint_evaluator: Callable) -> Dict:
        """Minimizes a scalar valued function.

        Args:
            obj_fun (Callable): A callable scalar valued function that
                accepts a two dimensional numpy array as its first arguments.
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
                variables found, 'fun' the optimal value of the optimized function, and 'success' a boolean
                indicating whether the optimization was conducted successfully.
        """
        if self._method_args is not None:
            res = self._method(obj_fun, x0, bounds=bounds, constraints=constraint_evaluator, **self._method_args)
        else:
            res = self._method(obj_fun, x0, bounds=bounds, constraints=constraint_evaluator)

        return res


class MixedIntegerMinimizer:
    def __init__(self, problem: MOProblem):
        """
        Args:
            problem (MOProblem): A scalarized MOProblem instance to be minimized.
        """
        self.problem = problem
        self.lower_bounds = [var.get_bounds()[0] for var in self.problem.variables]
        self.upper_bounds = [var.get_bounds()[1] for var in self.problem.variables]
        self.var_types = [var.type for var in self.problem.variables]

        # Print out the initialized values
        print(type(problem)) 
        print(f"Problem: {self.problem}")
        print(f"Lower bounds: {self.lower_bounds}")
        print(f"Upper bounds: {self.upper_bounds}")
        print(f"Var_types: {self.var_types}")
        
    def create_settings(self, max_evaluations=25, nlp_solver_path="ipopt", 
                        minlp_solver_path='/Users/seanjana/Desktop/Työt/project_codes/COIN_Bundle/coin.macos64.20211124/bonmin'):
        #TO DO: need to change the solver_path
        settings = rbfopt.RbfoptSettings(
            max_evaluations=max_evaluations,
            global_search_method="solver", 
            nlp_solver_path=nlp_solver_path, 
            minlp_solver_path=minlp_solver_path,
            print_solver_output=False
            
        )
        return settings
    
    def evaluate_objective(self, x):
        result = self.problem.objectives[0].evaluate(x).objectives[0]
        print(f"Evaluating at {x}, result: {result}")
        return result
    
    def minimize(self, x0, **kwargs):
        bb = rbfopt.RbfoptUserBlackBox(
            len(self.lower_bounds),
            self.lower_bounds,
            self.upper_bounds,
            self.var_types,
            #self.problem
            #lambda x: self.problem.objectives[0].evaluate(x).objectives[0]
            self.evaluate_objective
        )
        
        null_stream = open(os.devnull, 'w')
        alg = rbfopt.RbfoptAlgorithm(self.create_settings(), bb)
        alg.set_output_stream(null_stream)

        val, x, itercount, evalcount, fast_evalcount = alg.optimize()
        null_stream.close()
        
        return {'x': x, 'fun': val, 'success': itercount > 0, 'itercount': itercount, 'evalcount': evalcount, 'fast_evalcount': fast_evalcount}

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
                should be minimized with. It should accepts as keyword the arguments 'bounds' and
                'constraints' which will be used to pass it the bounds and constraint_evaluator.
                If none is supplied, uses the minimizer implemented in SciPy. Otherwise a str can be given
                to use one of the preset solvers available. Use the method 'get_presets' to get a list
                of available preset solvers.
                Defaults to None.
        """
        self.presets = ["scipy_minimize", "scipy_de", "MixedIntegerMinimizer"]
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
                lambda x, _, **y: differential_evolution(x, **y), method_args={"polish": True}
            )
            self._method = scipy_de_method
        
        #Add mixedIntegerSolver
        elif method == "MixedIntegerMinimizer":
            self._use_scipy = False
            self._mixed_integer_minimizer = MixedIntegerMinimizer(problem)
            self._method = ScalarMethod(lambda x, _, **y: self._mixed_integer_minimizer.minimize(x, **y))
        
        else:
            self._use_scipy = method._use_scipy
            self._method = method

            if self._use_scipy:
                # Assuming the gradient reqruies evaluation of the
                # scalarized function with out of bounds variable values.
                # only relevant if the 'polish' option is set in scipy's DE
                self._bounds[:, 0] += 1e-6
                self._bounds[:, 1] -= 1e-6

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
                variables found, 'fun' the optimal value of the optimized function, and 'success' a boolean
                indicating whether the optimizaton was conducted successfully.
        """
        if self._use_scipy:
            # create wrapper for the constraints to be used with scipy's minimize routine.
            # assuming that all constraints hold when they return a positive value.
            if self._constraint_evaluator is not None:
                scipy_cons = NonlinearConstraint(self._constraint_evaluator, 0, np.inf)
            else:
                scipy_cons = ()

            res = self._method(self._scalarizer, x0, bounds=self._bounds, constraint_evaluator=scipy_cons)

        else:
            res = self._method(
                self._scalarizer, x0, bounds=self._bounds, constraint_evaluator=self._constraint_evaluator
            )

        return res


class DiscreteMinimizer:
    """Implements a class for finding the minimum value of a discrete of scalarized vectors.
    """

    def __init__(
        self,
        discrete_scalarizer: DiscreteScalarizer,
        constraint_evaluator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            discrete_scalarizer (DiscreteScalarizer): A discrete scalarizer
                which takes as its arguments an array of vectors and returns a
                scalar value for each vector.
            constraint_evaluator (Optional[Callable[[np.ndarray],
            np.ndarray]], optional): An evaluator which returns True if a
                given vector(s) adheres to given constraints, and False
                otherwise. Defaults to None.
        """
        self._scalarizer = discrete_scalarizer
        self._constraint_evaluator = constraint_evaluator

    def minimize(self, vectors: np.ndarray) -> dict:
        """Find the index of the element in vectors which minimizes the
        scalar value returned by the scalarizer. If multiple minimum values
        are found, returns the index of the first occurrence.

        Args:
            vectors (np.ndarray): The vectors for which the minimum scalar
                value should be computed for.

        Raises:
            ScalarSolverException: None of the given vectors adhere to the
                given constraints.

        Returns:
            Dict: A dictionary with at least the following entries: 'x' indicating the optimal
                variables found, 'fun' the optimal value of the optimized function, and 'success' a boolean
                indicating whether the optimizaton was conducted successfully.
        """
        if self._constraint_evaluator is None:
            res = self._scalarizer(vectors)
            min_value = np.nanmin(res)
            min_index = np.nanargmin(res)
            return {"x": min_index, "fun": min_value, "success": True}
        else:
            bad_con_mask = ~self._constraint_evaluator(vectors)
            if np.all(bad_con_mask):
                raise ScalarSolverException("None of the supplied vectors adhere to the given " "constraint function.")
            tmp = np.copy(vectors)
            tmp[bad_con_mask] = np.nan
            res = self._scalarizer(tmp)
            min_value = np.nanmin(res)
            min_index = np.nanargmin(res)
            return {"x": min_index, "fun": min_value, "success": True}

if __name__ == "__main__":

    #DISCRETE PROBLEM

    from desdeo_tools.scalarization.ASF import PointMethodASF
    from desdeo_problem import variable_builder, ScalarObjective, MOProblem
    from desdeo_tools.scalarization.ASF import PointMethodASF
    from desdeo_tools.scalarization.Scalarizer import Scalarizer

    ideal = np.array([0, 0, 0, 0])
    nadir = np.array([1, 1, 1, 1])

    asf = PointMethodASF(nadir, ideal)
    dscalarizer = DiscreteScalarizer(asf, {"reference_point": None})
    dminimizer = DiscreteMinimizer(dscalarizer)

    non_dominated_points = np.array(
        [[0.2, 0.4, 0.6, 0.8], [0.4, 0.2, 0.6, 0.8], [0.6, 0.4, 0.2, 0.8], [0.4, 0.8, 0.6, 0.2]]
    )

    z = np.array([0.55, 0.4, 0.6, 0.8])

    dscalarizer._scalarizer_args = {"reference_point": z}

    print(asf(non_dominated_points, reference_point=z))

    res = dminimizer.minimize(non_dominated_points)
    print("res", res)
    
    
    #INTEGER PROBLEM
    
    def f1(x):
        x = np.atleast_2d(x)
        return -(x[:,0] + x[:,1])

    def f2(x):
        x = np.atleast_2d(x)
        return abs(x[:,0] - x[:,1])

    objective_1 = ScalarObjective(name="f1", evaluator=f1)
    objective_2 = ScalarObjective(name="f2", evaluator=f2)
    ideal = np.array([-1,5])
    nadir = np.array([-10,0])
    ref_point = np.array([-7,0])
    
    #Bounds
    l_bounds=[0, 1]
    u_bounds=[5, 5]
    bounds = np.stack((l_bounds, u_bounds))
    
    # Define the variables and their bounds
    varsl = variable_builder(["x1", "x2"],
                             initial_values=[4, 3],
                             lower_bounds=[0, 1],
                             upper_bounds=[5, 5],
                             types=["R", "I"])
    
    # Create the problem instance
    problem = MOProblem(variables=varsl, objectives=[objective_1, objective_2], ideal=ideal, nadir=nadir)
    
    #ASF
    
    #use problem.evaluate and use fitness values 
    #can use lambda function after that
    #add variable types as well as bounds 
    
    #change from R and I to Real and Integer and Binary and Other
    #If using bonmin change from Real to R and I
    
    asf = PointMethodASF(nadir, ideal)
    scalarized_objectives = Scalarizer(asf, lambda x: problem.evaluate(x).fitness, scalarizer_args={"reference_point": ref_point})
    
    # Given a point x
    x = np.array([1, 1])

    # The scalarized objective value at x can be computed as follows:
    #scalarized_objectives(x)

    #print("The scalarized objective value is: ", lambda x: problem.evaluate(x).fitness)
    
    initial_guess = np.array([1, 1])  # Or any other starting point you prefer
    minimizer = ScalarMinimizer(scalarized_objectives, bounds, method="MixedIntegerMinimizer")
    res = minimizer.minimize(initial_guess)
    print(res)

    
