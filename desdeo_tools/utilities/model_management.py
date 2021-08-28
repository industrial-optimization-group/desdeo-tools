from numpy.core.numeric import indices
from numpy.lib.arraysetops import unique
from desdeo_tools.scalarization.ASF import SimpleASF
from numba import njit
import numpy as np
from desdeo_problem.problem.Objective import ScalarDataObjective
import pandas as pd
import copy
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,\
    WhiteKernel, RBF, Matern, ConstantKernel



def remove_duplicate(X, archive_x):
    """identifies of the duplicate rows for decision variables"""
    indicies = None
    archive = archive_x.to_numpy()
    tmp = X
    for i in archive:
        for k in range(len(X)):
            for j in range(len(i)):
                tmp[k,j] = X[k,j] - i[j]
        tmp = np.round(tmp, 3)
        if indicies is None:
            indicies = np.where(~tmp.any(axis=1))[0]
        else:
            tmp = np.where(~tmp.any(axis=1))[0]
            if tmp.size > 0:
                indicies = np.hstack((indicies.squeeze(),tmp))
        if indicies.size == 0:
            indicies = None
    if indicies is None:
        return None
    else:

        return indicies


def ikrvea_mm(reference_point, evolver , problem, u: int) -> float:
    """ Computes the additive epsilon-indicator between reference front and current approximating front.

    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        front (np.ndarray): The front that is compared. Should be one-dimensional array with the same shape as
        reference_front.

    Returns:
        float: The factor by which the approximating front is worse than the reference front with respect to all
        objectives.
    """
    archive = problem.archive.to_numpy()
    surrogate_obj = copy.deepcopy( evolver.population.objectives)
    decision_variables = copy.deepcopy(evolver.population.individuals)
    unc = copy.deepcopy(evolver.population.uncertainity)
    #pd.concat([b,b], ignore_index= True)
    nd = remove_duplicate(decision_variables, problem.archive.drop(
            problem.objective_names, axis=1)) #removing duplicate solutions
    if nd is not None:
        non_duplicate_dv = evolver.population.individuals[nd]
        non_duplicate_obj = evolver.population.objectives[nd]
        non_duplicate_unc = evolver.population.uncertainity[nd]
    else:
        non_duplicate_dv = evolver.population.individuals
        non_duplicate_obj = evolver.population.objectives
        non_duplicate_unc = evolver.population.uncertainity


    asf_solutions = SimpleASF([1,1,1]).__call__(non_duplicate_obj, reference_point)
    idx = np.argpartition(asf_solutions, 2*u)
    asf_unc = np.max(non_duplicate_unc [idx[0:2*u]], axis= 1)
    lowest_unc_index = np.argpartition(asf_unc, u)[0:u]
    names = np.hstack((problem.variable_names,problem.objective_names))
    reevaluated_objs = problem.evaluate(non_duplicate_dv[lowest_unc_index], use_surrogate=False)[0]
    new_results = np.hstack((non_duplicate_dv[lowest_unc_index], reevaluated_objs))
    archive = np.vstack((archive, new_results))
    new_archive = pd.DataFrame(archive, columns=names)
    problem.archive = new_archive #updating the archive
    problem.train(models=GaussianProcessRegressor,\
         model_parameters={'kernel': Matern(nu=1.5)}) 

    return problem


