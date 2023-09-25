import numpy as np
from math import sqrt


def mapping_p(obtainpop: np.ndarray, pref_point: list):
    """"""
    size = len(obtainpop)
    mapping_point = np.array([None] * size, dtype=object)
    fsize = 0
    T = 0.0
    for i in range(0, size):
        t = 0.0
        p = 0.0
        fsize = len(obtainpop[i])
        for j in range(0, fsize):
            t += obtainpop[i][j] * pref_point[j]
            p += pref_point[j] * pref_point[j]
        T = 1 - (t / p)
        f = np.zeros(fsize)
        for j in range(0, fsize):
            f[j] = obtainpop[i][j] + pref_point[j] * T
        mapping_point[i] = f
    return mapping_point


def mapping_distance(mapping_point, pref_point):
    """Calculate D1 in the paper, which is distance between mapping points and reference point
    Args:
        mapping_point (np.ndarray): The points that we moved to the hyperplane.
        pref_point (np.ndarray): The reference point that the DM provides.



    Returns:
        float: D1
    """
    sum_ = 0
    distance = 0
    size = len(mapping_point)
    pp_size = len(pref_point)
    for j in range(0, size):
        for i in range(0, pp_size):
            distance += (mapping_point[j][i] - pref_point[i]) * (
                mapping_point[j][i] - pref_point[i]
            )
        sum_ += sqrt(distance)
    return sum_ / distance


def dist_vector(vec1, vec2):
    dim = len(vec1)
    sum_ = 0
    for i in range(0, dim):
        sum_ += (vec1[i] - vec2[i]) ** 2
    return sqrt(sum_)


def d2_spcing(df: np.ndarray):
    """Calculate D2 in the paper, which is the standard deviation of each mapping point
    to the nearest point
    Args:
       df(np.ndarray): The points that we moved to the hyperplane.


    Returns:
        float: D2
    """
    size = len(df)
    sum_ = 0
    dist = []
    temp = 0
    for i in range(0, size):
        distance = 1.0e30
        for j in range(0, size):
            if j != i:
                temp = dist_vector(df[i], df[j])
                if temp < distance:
                    distance = temp
        dist.append(distance)
    average = sum(dist) / len(dist)
    for i in range(0, size):
        temp = average - dist[i]
        sum_ += temp ** 2
    return sqrt(sum_ / (size - 1))


def distan_d3(
    ref_point: np.ndarray,
    population: np.ndarray,
    r: float,
    k: float,
    inside_ROI=[],
    outside_ROI=[],
):
    """Calculate D3 in the paper, which is the distance between preferred solution
    and origin.
    Args:
       ref_point(np.ndarray): The points that we moved to the hyperplane.
       population(np.ndarray): The points that we moved to the hyperplane.
       r(float) : the size of region of interest
       k(float): penalty coefficient
       inside_ROI= (list): solutions inside ROI
       outside_ROI (list): solutions outside ROI


    Returns:
        float: D3
    """
    dim = len(population)
    sum1 = 0
    sum2 = 0
    d = 0
    origion = [0] * len(population[0])
    map_point = mapping_p(population, ref_point)
    for i in range(0, dim):
        d = dist_vector(map_point[i], ref_point)
        sum1 = dist_vector(population[i], origion)
        if d <= r:
            sum2 += sqrt(sum1)
            inside_ROI.append(i)
        else:
            sum2 += k * sqrt(sum1)
            outside_ROI.append(i)
    return sum2 / dim


def get_pmod(ref_point: np.ndarray, population: np.ndarray, r: float, k: float):
    """computes the PMOD indicator based on the following paper:
    "Hou, Zhanglu, et al. "A performance indicator for reference-point-based multiobjective evolutionary
    optimization." 2018 IEEE Symposium Series on Computational Intelligence (SSCI). IEEE, 2018".
    """
    mapping_point = mapping_p(population, ref_point)
    d1 = mapping_distance(population, ref_point)
    d2 = d2_spcing(mapping_point)
    d3 = distan_d3(ref_point, population, r, k)
    pmod_value = d1 + d2 + d3
    return pmod_value
