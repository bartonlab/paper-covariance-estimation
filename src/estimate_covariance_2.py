import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
from scipy import stats
import math
import MPL                                  # MPL inference tools


def getRegularizedEstimate(traj, times, window, population=1000, calibration=True, nonlinear_reg=True, loss=3, gamma=1e-5):

    est_cov = estimateIntegratedCovarianceMatrix(traj, times, window, population=population, calibration=calibration)
    if nonlinear_reg:
        est_corr = computeCorrelation(est_cov)
        multiplier = computeCorrToCovMultiplier(est_cov)
        shrunk_corr = shrinkCorrelation(est_corr, loss, gamma)
        reg_cov = np.linalg.multi_dot([multiplier, shrunk_corr, multiplier])
        return reg_cov
    else:
        return est_cov


def estimateIntegratedCovarianceMatrix(traj, times, window, population=1000, calibration=True):
    """Estimates integrated covariance matrix from allele frequency trajectories."""

    T = len(traj) - 1
    L = len(traj[0])
    variances = computeVariances(traj)
    dx = computeDx(traj)
    dxdx = computeDxdx(dx)
    est_cov = np.zeros((T, L, L), dtype=float)
    for t in range(T):
        # C_ij(t) = N * <dx_i(t) * dx_j(t)>
        count_time_point = 1
        est_cov[t] = dxdx[t]
        for dt in range(1, window + 1):
            if (t + dt >= T) or (t - dt < 0):
                break
            est_cov[t] += dxdx[t + dt]
            est_cov[t] += dxdx[t - dt]
            count_time_point += 2
        if not calibration:
            for i in range(L):
                est_cov[t, i, i] *= population / count_time_point
                for j in range(i):
                    est_cov[t, i, j] *= population / count_time_point
                    est_cov[t, j, i] = est_cov[t, i, j]
        else:
            rescaling_multiplier = np.full((L, L), 1, dtype=float)
            for i in range(L):
                # Only when frequency of site i remains unchanged in the time window, could est_cov[t, i, i] be 0, in which case we preserve the original estimate of covariance.
                if est_cov[t, i, i] != 0:
                    for j in range(i):
                        if est_cov[t, j, j] != 0:
                            rescaling_multiplier[i, j] = np.sqrt(variances[t, i] * variances[t, j] / (est_cov[t, i, i] * est_cov[t, j, j]))
                            rescaling_multiplier[j, i] = rescaling_multiplier[i, j]
            for i in range(L):
                est_cov[t, i, i] = variances[t, i]
                for j in range(i):
                    est_cov[t, i, j] *= rescaling_multiplier[i, j]
                    est_cov[t, j, i] = est_cov[t, i, j]

    int_est_cov = np.zeros((L, L), dtype=float)
    for t in range(T):
        int_est_cov += (times[t + 1] - times[t]) * est_cov[t]
    return int_est_cov


def computeVariances(traj):
    """Computes variances."""

    T = len(traj) - 1
    L = len(traj[0])
    variances = np.zeros((T, L), dtype=float)
    for t in range(T):
        for i in range(L):
            variances[t, i] = traj[t, i] * (1 - traj[t, i])
            # Might be negative due to precision limit, in which case we reset it to be 0
            if variances[t, i] < 0:
                variances[t, i] = 0
    return variances


def computeDx(traj):
    """Computes change of frequency by next generation at each time point."""

    T = len(traj) - 1
    L = len(traj[0])
    dx = np.zeros((T, L), dtype=float)
    for t in range(T):
        dx[t] = traj[t + 1] - traj[t]
    return dx


def computeDxdx(dx):
    """Computes [dx_i * dx_j] matrix."""

    T, L = dx.shape
    dxdx = np.zeros((T, L, L), dtype=float)
    for t in range(T):
        dxdx[t] = np.outer(dx[t], dx[t])
    return dxdx


def computeCorrelation(cov_matrix):
    """Computes correlation matrix from covariance matrix."""

    L = len(cov_matrix)
    multiplier = np.zeros(L, dtype=float)
    corr = np.zeros((L, L), dtype=float)

    for i in range(L):
        if cov_matrix[i, i] > 1e-309:
            multiplier[i] = 1 / np.sqrt(cov_matrix[i, i])

    for i in range(L):
        for j in range(i, L):
            if cov_matrix[i, i] != 0 and cov_matrix[j, j] != 0:
                corr[i, j] = cov_matrix[i, j] * multiplier[i] * multiplier[j]
                corr[j, i] = corr[i, j]

    return corr


def computeCorrToCovMultiplier(cov):
    """Computes the multiplier by which, multiplier x correlation x multiplier = covariance."""

    multiplier = np.zeros_like(cov)
    for t in range(len(cov)):
        multiplier[t, t] = math.sqrt(cov[t, t])
    return multiplier


def shrinkCorrelation(matrix, loss, gamma, PRESERVE_SMALL_END=False):
    """Shrinks eigenvalues of a matrix."""

    eigenVal, eigenVec = computeSortedEigenmodes(matrix)

    # Following the Marchenko–Pastur law
    bulk_edge = (1 + math.sqrt(gamma)) ** 2
    if PRESERVE_SMALL_END:
        bulk_edge_small = (1 - math.sqrt(gamma)) ** 2
    else:
        bulk_edge_small = 0

    matrix_shrunk = np.zeros_like(matrix)
    for i in range(len(matrix_shrunk)):
        tmp = np.float64(shrinkEigenvalue(eigenVal[i], bulk_edge, loss, gamma, bulk_edge_small=bulk_edge_small))
        matrix_shrunk += tmp * np.outer(eigenVec[i], eigenVec[i])

    return matrix_shrunk


def computeSortedEigenmodes(matrix):
    """Computes eigenvalues and eigenvectors, sorted from smallest eigenvalue to largest."""

    eigenVal_tmp, eigenVec_tmp = np.linalg.eigh(matrix)
    indices_smallest = np.argsort(eigenVal_tmp)
    eigenVal = np.array([eigenVal_tmp[i] for i in indices_smallest])
    eigenVec = np.array([np.array(eigenVec_tmp[:, i]) for i in indices_smallest])

    return eigenVal, eigenVec


def shrinkEigenvalue(eigenvalue, bulk_edge, loss, gamma, bulk_edge_small=0):
    """Shrinks an eigenvalue following optimal shrinkers proposed in Donoho et al. (2018).

    Args:
        eigenvalue: The eigenvalue to be shrunk.
        bulk_edge: The larger bulk_edge, above which an eigenvalue is thought to be from signals, rather than noises.
        loss: An integer in range [-1, 9], indicating a loss function. See loss function list below:
            -1. No shrinkage in this case, as a blank compare.
            0. Frobenius norm of A - B
            1. Frobenius norm of A^(-1) - B^(-1) (also Stein's loss)
            2. Frobenius norm of A^(-1) B - I
            3. Frobenius norm of B^(-1) A - I
            4. Frobenius norm of A^(-1/2) B A^(-1/2) - I
            5. Nuclear norm of A - B
            6. Nuclear norm of A^(-1) - B^(-1)
            7. Nuclear norm of A^(-1) B - I
            8. Nuclear norm of B^(-1) A - I
            9. Nuclear norm of A^(-1/2) B A^(-1/2) - I
        gamma: A number in range (0, 1], the asymptotic ratio between number of variables and number of samples when they both tend to infinity.
        bulk_edge_small: Optional, a non-negative number; When it is positive, we preserve eigenvalues smaller than it.
    """

    if eigenvalue <= bulk_edge and eigenvalue >= bulk_edge_small:
        return 1
    # Preserve small eigenvalues if bulk_edge_small > 0; Impossible if bulk_edge_small = 0
    elif eigenvalue < bulk_edge_small:
        return eigenvalue
    # Eigenvalues larger than bulk_edge
    else:
        if loss == -1:
            return eigenvalue
        l = ((eigenvalue + 1 - gamma) + math.sqrt((eigenvalue + 1 - gamma) ** 2 - 4 * eigenvalue)) / 2
        # Safety handling to prevent sqrt(-1) later
        if l < 1 + math.sqrt(gamma):
            return 1
        c = math.sqrt((1 - gamma / (l - 1) ** 2) / (1 + gamma / (l - 1)))
        s = math.sqrt(1 - c ** 2)
        if loss == 0:
            return (l * c ** 2 + s ** 2)
        elif loss == 1:
            return (l / (c ** 2 + l * s ** 2))
        elif loss == 2:
            return ((l * c ** 2 + l ** 2 * s ** 2) / (c ** 2 + l ** 2 * s ** 2))
        elif loss == 3:
            return ((l ** 2 * c ** 2 + s ** 2) / (l * c ** 2 + s ** 2))
        elif loss == 4:
            return (1 + (l - 1) * c ** 2 / (c ** 2 + l * s ** 2) ** 2)
        elif loss == 5:
            return (max(1 + (l - 1) * (1 - 2 * s ** 2), 1))
        elif loss == 6:
            return (max(l / (c ** 2 + (2 * l - 1) * s ** 2), 1))
        elif loss == 7:
            return (max(l / (c ** 2 + l ** 2 * s ** 2), 1))
        elif loss == 8:
            return (max((l ** 2 * c ** 2 + s ** 2) / l, 1))
        elif loss == 9:
            return (max((l - (l - 1) ** 2 * c ** 2 * s ** 2) / (c ** 2 + l * s ** 2) ** 2, 1))
