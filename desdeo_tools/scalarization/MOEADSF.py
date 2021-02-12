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
    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the SF.

        Args:
            objective_vectors (np.ndarray): The objective vectors to calculate
            the values.
            weight_vectors (np.ndarray): The weight vector to calculate the
            values.
            reference_point (np.ndarray): The reference point (i.e nadir or ideal vector).
        Returns:
            Union[float, np.ndarray]: Either a single SF value or a vector of
            values if objective is a 2D array.
        """
        pass


class Tchebycheff(MOEADSFBase):
    """Implements the Tchebycheff scalarizing function.
    """

    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the Tchebycheff scalarizing function.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
            the objective space.
            weight_vector (np.ndarray): A weight vector representing the direction
            reference_point (np.ndarray): A reference point (i.e the ideal vector or the nadir vector)
        Raises:
            MOEADSFError: The dimensions of the objective vector and weight_vector don't match.

        Note:
            The shaped of objective_vector and weight_vector must match.

        """
        if not objective_vector.shape == weight_vector.shape:
            msg = ("The dimensions of the objective vector {} and " "weight_vector {} do not match.").format(
                objective_vector.shape, weight_vector.shape
            )
            raise MOEADSFError(msg)

        feval   = np.abs(objective_vector - reference_point) * weight_vector
        max_fun = np.max(feval)
        return  max_fun


class WeightedSum(MOEADSFBase):
    """Implements the Weighted sum scalarization function
    """
    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, reference_point = None) -> Union[float, np.ndarray]:
        """Evaluate the WeightedSum scalarizing function.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
            the objective space.
            weight_vector (np.ndarray): A weight vector representing the direction
        Raises:
            MOEADSFError: The dimensions of the objective vector and weight_vector don't match.

        Note:
            The shaped of objective_vector and weight_vector must match. The reference point is not needed.

        """
        if not objective_vector.shape == weight_vector.shape:
            msg = ("The dimensions of the objective vector {} and " "weight_vector {} do not match.").format(
                objective_vector.shape, weight_vector.shape
            )
            raise MOEADSFError(msg)
        feval   = np.sum(objective_vector * weight_vector)
        return feval


class PBI(MOEADSFBase):
    """Implements the PBI scalarization function
    Args:
        theta(float): A penalty parameter used by the function

    Attributes:
        theta (float): A penalty parameter used by the function

    .. Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," 
       in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
    """

    def __init__(
        self,
        theta: float = 5
    ):
        self.theta = theta

    def __call__(self, objective_vector: np.ndarray, weight_vector: np.ndarray, reference_point: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate the PBI scalarizing function.

        Args:
            objective_vector (np.ndarray): A vector representing a solution in
            the objective space.
            weight_vector (np.ndarray): A weight vector representing the direction
            reference_point (np.ndarray): A reference point (i.e the ideal vector or the nadir vector)
        Raises:
            MOEADSFError: The dimensions of the objective vector and weight_vector don't match.

        Note:
            The shaped of objective_vector and weight_vector must match. The reference point is not needed.

        """
        if not objective_vector.shape == weight_vector.shape:
            msg = ("The dimensions of the objective vector {} and " "weight_vector {} do not match.").format(
                objective_vector.shape, weight_vector.shape
            )
            raise MOEADSFError(msg)

        norm_weights    = np.linalg.norm(weight_vector)
        weights         = np.true_divide(weight_vector, norm_weights)
        fx_a            = objective_vector - reference_point
        d1              = np.fabs(np.inner(fx_a, weights))

        fx_b            = objective_vector - (reference_point + d1 * weights)
        d2              = np.linalg.norm(fx_b)
        
        
        fvalue          = d1 + self.theta * d2
        return fvalue



