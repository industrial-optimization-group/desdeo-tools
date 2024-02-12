from numba import njit
import numba
import numpy as np
import hvwfg as hv
from desdeo_tools.utilities.fast_non_dominated_sorting import dominates, fast_non_dominated_sort_indices
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator

from desdeo_tools.scalarization import SimpleASF


@njit()
def epsilon_indicator(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes the additive epsilon-indicator between two solutions.

    Args:
        s1 (np.ndarray): Solution 1. Should be an one-dimensional array.
        s2 (np.ndarray): Solution 2. Should be an one-dimensional array.

    Returns:
        float: The maximum distance between the values in s1 and s2.
    """
    eps = 0.0
    for i in range(s1.size):
        value = s2[i] - s1[i]
        if value > eps:
            eps = value
    return eps


@njit()
def epsilon_indicator_ndims(front: np.ndarray, reference_point: np.ndarray) -> list:
    """Computes the additive epsilon-indicator between reference point and current one-dimensional vector of front.

    Args:
        front (np.ndarray): The front that the current reference point is being compared to.
            Should be set of arrays, where the rows are the solutions and the columns are the objective dimensions.
        reference_point (np.ndarray): The reference point that is compared. Should be one-dimensional array.

    Returns:
        list: The list of indicator values.
    """
    min_eps = 0.0
    eps_list = np.zeros((front.shape[0]), dtype=numba.float64)
    for i in np.arange(front.shape[0]):
        value = np.max(reference_point - front[i])
        if value > min_eps:
            eps_list[i] = value
    return eps_list


def preference_indicator(
    s1: np.ndarray,
    s2: np.ndarray,
    min_asf_value: float,
    ref_point: np.ndarray,
    delta: float,
) -> float:
    """Computes the preference-based quality indicator.

    Args:
        s1 (np.ndarray): Solution 1. Should be an one-dimensional array.
        s2 (np.ndarray): Solution 2. Should be an one-dimensional array.
        ref_point (np.ndarray): The reference point should be same shape as front.
        min_asf_value (float): Minimum value of achievement scalarization of the reference_front. Used in normalization.
        delta (float): The spesifity delta allows to set the amplification of the indicator to be closer or farther
            from the reference point. Smaller delta means that all solutions are in smaller range around the reference
            point.

    Returns:
        float: The maximum distance between the values in s1 and s2 taking into account
            the reference point and spesifity.
    """
    s2_asf = SimpleASF(np.ones_like(s2))
    norm = s2_asf(s2, reference_point=ref_point) + delta - min_asf_value
    return epsilon_indicator(s1, s2) / norm


def hypervolume_indicator(front: np.ndarray, reference_point: np.ndarray) -> float:
    """Computes the hypervolume-indicator between reference front and current approximating point.

    Args:
        front (np.ndarray): The front that is compared. Should be set of arrays, where the rows are the solutions and
            the columns are the objective dimensions.
        reference_point (np.ndarray): The reference point that the current front is being compared to.
        Should be 1D array.

    Returns:
        float: Measures the volume of the objective space dominated by an approximation set.
    """
    ref = np.asarray(
        reference_point, dtype="double"
    )  # hv.wfg needs datatype to be double
    fr = np.asarray(front, dtype="double")
    return hv.wfg(fr, ref)

"""This code implements the PHI (Preference-based Hypervolume Indicator) and related decision assessment
 methods as introduced in the paper "A Performance Indicator for Interactive Evolutionary Multiobjective 
 Optimization Methods." It's designed for analyzing multiobjective optimization problems, taking into 
 account decision-maker preferences. The PHI indicator evaluates the performance of solutions relative
 to a reference point, focusing on the coverage of the desired solution region.

 To run the code to get the phi values you should run get_phi(),and for the decision phase you should run assess_decision_phase()

For inquiries or further details, contact pouya(dot)aghaeipour(at)gmail.com.
 When using this code or its methodology in academic or research work, 
 please cite the paper appropriately to acknowledge the original work and its contributors.
 P. Aghaei Pour, S. Bandaru, B. Afsar, M. Emmerich and K. Miettinen, "A Performance Indicator
  for Interactive Evolutionary Multiobjective Optimization Methods," in IEEE Transactions
on Evolutionary Computation, doi: 10.1109/TEVC.2023.3272953.
 """
class phi():
    def __init__(self, ideal):
        """Initialize with an ideal point for hypervolume calculations."""
        self.name = 'test'
        self.ideal = ideal

    def check_rp_dominated(self, set_of_s, RP):
        """Check if the reference point (RP) is dominated by any solution in set_of_s."""
        r = False
        doms = []
        for s in set_of_s:
            if dominates(s, RP):
                doms.append(True)
                r = True
            else:
                doms.append(False)
        return r, doms

    def RP_dom_cal(self, set_of_s, RP, doms, nadir):
        """Calculate various hypervolume metrics when RP is dominated."""
        ind = np.where(doms)[0]
        nondoms = np.vstack((set_of_s, RP))[fast_non_dominated_sort_indices(np.vstack((set_of_s, RP)))[0][0]]
        max_phv = hypervolume_indicator(np.asanyarray(self.ideal).reshape(1, -1), nadir)
        all_phv = hypervolume_indicator(nondoms, nadir)
        rp_phv = hypervolume_indicator(np.asanyarray(RP).reshape(1, -1), nadir)
        pos_phv = hypervolume_indicator(np.asanyarray(set_of_s[ind]), nadir) - rp_phv
        neg_phv = all_phv - pos_phv - rp_phv
        if all_phv == 0:
            return 0, 0, 0
        else:
            return 1 + (pos_phv / max_phv), (pos_phv + rp_phv) / max_phv, neg_phv / max_phv, rp_phv / max_phv

    def RP_nondom_cal(self, set_of_s, RP, nadir):
        """Calculate various hypervolume metrics when RP is not dominated."""
        nondoms = np.vstack((set_of_s, RP))[fast_non_dominated_sort_indices(np.vstack((set_of_s, RP)))[0][0]]
        all_phv = hypervolume_indicator(nondoms, nadir)
        rp_phv = hypervolume_indicator(np.asanyarray(RP).reshape(1, -1), nadir)
        s_phv = hypervolume_indicator(np.asanyarray(set_of_s), nadir)
        nondom_area = all_phv - s_phv
        pos_phv = rp_phv - nondom_area
        neg_phv = all_phv - rp_phv
        if all_phv == 0:
            return 0, 0, 0
        else:
            return pos_phv / rp_phv, pos_phv / all_phv, neg_phv / all_phv, rp_phv

    def get_phi(self, set_of_s, RP, nadir):
        is_rp_dominated, doms = self.check_rp_dominated(set_of_s, RP)
        if is_rp_dominated:
            combined_array = np.vstack((set_of_s, RP))
            sorted_indices = fast_non_dominated_sort_indices(combined_array)

            # Check if sorted_indices is empty or does not contain index 0
            if len(sorted_indices) == 0 or len(sorted_indices[0]) == 0:
                print("Warning: No non-dominated solutions found.")
                return None

            results = self.RP_dom_cal(set_of_s, RP, doms, nadir)
        else:
            results = self.RP_nondom_cal(set_of_s, RP, nadir)
        return results

class phi_decision():
    def __init__(self, n_interactions, indicator_values, nadir):
        """Initialize with the number of interactions, indicator values, and nadir for hypervolume calculations."""
        self.name = 'test'
        self.n_interactions = n_interactions
        self.indicator_values = indicator_values
        self.nadir = nadir

    def get_areas(self, rp1, rp2):
        """Calculate the shared hypervolume area between two reference points."""
        # Ensure rp1 and rp2 are 2D arrays
        if rp1.ndim == 1:
            rp1 = rp1.reshape(1, -1)
        if rp2.ndim == 1:
            rp2 = rp2.reshape(1, -1)

        dom21 = dominates(rp2.flatten(), rp1.flatten())
        dom12 = dominates(rp1.flatten(), rp2.flatten())
        hv_rp1 = hypervolume_indicator(rp1, self.nadir_1d)
        hv_rp2 = hypervolume_indicator(rp2, self.nadir_1d)
        hv_rp12 = hypervolume_indicator(np.vstack((rp1, rp2)), self.nadir_1d)
        self.hv_rp12 = hv_rp12
        if dom21:
            shared_area = hv_rp1
        elif dom12:
            shared_area = hv_rp2
        else:
            extra_area_in_rp1 = abs(hv_rp12 - hv_rp2)
            shared_area = hv_rp1 - extra_area_in_rp1
        return shared_area

    def interactions_areas(self, set_of_RPs, main_RP, n_interactions):
        """Calculate interaction areas for a set of reference points and a main reference point."""
        areas = []
        if n_interactions > 2:
            for s in set_of_RPs:
                areas.append(self.get_areas(s, main_RP))
        else:
            areas = self.get_areas(set_of_RPs, main_RP)
        return areas

    def get_weights(self, w, main_w):
        """Calculate the weights for the hypervolume shared areas."""
        return w / self.hv_rp12

    def assess(self, w, assessment_values):
        """Assess the decision phase using weighted mean of assessment values."""
        assessment = np.mean(w * assessment_values)
        return assessment

    def assess_decision_phase(self, set_of_RPs, main_RP):
        """Assess the decision phase for a set of reference points and a main reference point."""
        # Reshape main_RP to 2D array if it is 1D
        if main_RP.ndim == 1:
            main_RP = main_RP.reshape(1, -1)

        # Ensure self.nadir is a 1D array
        self.nadir_1d = self.nadir.flatten()

        main_area = hypervolume_indicator(main_RP, self.nadir_1d)
        shared_areas = self.interactions_areas(set_of_RPs, main_RP, self.n_interactions)
        weights = self.get_weights(np.asarray(shared_areas), main_area)
        results = self.assess(np.asarray(weights), np.asarray(self.indicator_values))
        return results, weights


if __name__ == "__main__":
    po_front = np.asarray([[1.0, 0], [0.5, 0.5], [0, 1.0], [2, -1], [0, 0]])
    sol1 = [4, 4]  # cant be better than po front, min is zero
    sol = np.asarray(sol1)
    ref = np.asarray([0.7, 0.3])

    print("eps indi value")
    print(epsilon_indicator(po_front[0], sol))
    print(epsilon_indicator(po_front[1], sol))
    print(epsilon_indicator(po_front[2], sol))
    print(epsilon_indicator(po_front[3], sol))
    print(epsilon_indicator(po_front[4], sol))
    print("ndims")
    print(epsilon_indicator_ndims(po_front, sol))
    print(hypervolume_indicator(po_front, sol))
    print("pref")
    print(preference_indicator(po_front[1], sol, 0.1, ref, 0.1))
