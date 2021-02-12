import abc
from abc import abstractmethod
from os import path
from typing import List, Union

import numpy as np


class MOEADSFError(Exception):
    """Raised when an error related to the MOEADSF classes is encountered.

    """


class MOEADSFBase(abc.ABC):
    """A base class for representing scalarizing functions for the MOEA/D algorithm.
    Instances of the implementations of this class should work as
    function.

    """

    @abstractmethod
    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, ideal_vector: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the SF.

        Args:
            objective_vectors (np.ndarray): The objective vectors to calculate
            the values.
            weight_vectors (np.ndarray): The weight vector to calculate the
            values.

        Returns:
            Union[float, np.ndarray]: Either a single SF value or a vector of
            values if objective is a 2D array.
        """
        pass


class Tchebycheff(MOEADSFBase):
    """Implements the Tchebycheff scalarizing function.
    """

    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, ideal_vector: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the Tchebycheff scalarizing function.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
            the solution space.
            weight_vector (np.ndarray): 
            ideal_vector (np.ndarray): 
        Raises:
            MOEADSFError: The dimensions of the objective vector and weight_vector don't match.

        Note:
            The shaped of objective_vector and weight_vector must match.

        """
        if not objective_vector.shape == weight_vector.shape:
            msg = ("The dimensions of the objective vector {} and " "weight_vector {} do not match.").format(
                objective_vector, weight_vector
            )
            raise MOEADSFError(msg)

        feval   = np.abs(objective_vector - ideal_vector) * weight_vector
        max_fun = np.max(feval)
        return  max_fun


class WeightedSum(MOEADSFBase):
    """Implements the Weighted sum scalarization function
    """
    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, ideal_vector = None) -> Union[float, np.ndarray]:
        feval   = np.sum(objective_vector * weight_vector)
        return feval


class PBI(MOEADSFBase):
    """Implements the PBI scalarization function

    Args:
        ideal (np.ndarray): The ideal point.
        theta(float): 

    Attributes:
        ideal (np.ndarray): The ideal point.
        theta (float): 
    """

    def __init__(
        self,
        theta: float = 5
    ):
        self.theta = theta

    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, ideal_vector: np.ndarray) -> Union[float, np.ndarray]:
        norm_weights    = np.linalg.norm(weight_vector)
        weights         = weight_vector/norm_weights
        fx_a            = objective_vector - ideal_vector
        d1              = np.inner(fx_a, weights)

        fx_b            = objective_vector - (ideal_vector + d1 * weights)
        d2              = np.linalg.norm(fx_b)
        
        fvalue          = d1 + self.theta * d2
        return fvalue





