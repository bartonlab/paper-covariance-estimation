#!/usr/bin/env python
# Estimate covariance information from time-series genetic data

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
from scipy import stats
import math
import MPL                                  # MPL inference tools

# EPSILON = np.finfo(np.float64).eps
EPSILON = 1e-10


def estimateAndInferencePipeline(traj, times, window, mu, population=1000, calibrate=True, interpolate=True, nonlinear_reg=True, loss=3, gamma=1e-5, linear_reg=1):

    D = MPL.computeD(traj, times, mu)
    est_cov = getRegularizedEstimates(traj, times, window, population=population, calibrate=calibrate, interpolate=interpolate, nonlinear_reg=nonlinear_reg, loss=loss, gamma=gamma)

    if nonlinear_reg:
        selection = MPL.inferSelection(est_cov, D, np.identity(len(est_cov)))
    else:
        selection = MPL.inferSelection(est_cov, D, linear_reg * np.identity(len(est_cov)))
    return est_cov, selection


def getRegularizedEstimates(traj, times, window, population=1000, calibrate=True, interpolate=False, nonlinear_reg=True, loss=3, gamma=1e-5):

    est_cov = estimateIntegratedCovarianceMatrix(traj, times, window, population=population, calibrate=calibrate, interpolate=interpolate)
    if nonlinear_reg:
        return nonlinearRegularization(est_cov, loss=loss, gamma=gamma)
    else:
        return est_cov


def nonlinearRegularization(est_cov, loss=3, gamma=1e-5):

    est_corr = computeCorrelation(est_cov)
    multiplier = computeCorrToCovMultiplier(est_cov)
    shrunk_corr = shrinkCorrelation(est_corr, loss, gamma)
    reg_cov = np.linalg.multi_dot([multiplier, shrunk_corr, multiplier])

    return reg_cov


def computeCorrelation(cov_matrix):
    """Computes correlation matrix from covariance matrix."""

    L = len(cov_matrix)
    multiplier = np.zeros(L, dtype=float)
    corr = np.zeros((L, L), dtype=float)

    for i in range(L):
        if cov_matrix[i, i] >= EPSILON:
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
    for l in range(len(cov)):
        multiplier[l, l] = math.sqrt(cov[l, l])
    return multiplier


def estimateIntegratedCovarianceMatrix(traj, times, window, population=1000, calibrate=True, return_est_at_each_time=False, interpolate=True):
    """Estimates integrated covariance matrix from allele frequency trajectories."""

    T = len(traj) - 1
    L = len(traj[0])

    est_cov = compute_est_cov(traj, window, population=population, calibrate=calibrate)

    int_est_cov = integrateCovarianceMatrix(traj, times, est_cov, interpolate=(interpolate and calibrate))  # Only interpolate when calibration (normalization) is enabled. This is because interaction between non-calibration and interpolation will lead to variances with smaller magnitude such that inferred selection gets too off.
    if calibrate:
        int_var = MPL.computeIntegratedVariance(traj, times)
        calibrate_est_cov(int_est_cov, [int_var[i, i] for i in range(L)])  # When enabling the calibration (normalization), the variance terms should be the same as integrated true variances. However, because estimation at a time point requires frequencies at next time point, the integrated estimated variances actually use one less time point (the last one). Therefore there is slight difference, which we calibrate again here to eliminate.
    if return_est_at_each_time:
        return est_cov, int_est_cov
    else:
        return int_est_cov


def compute_est_cov(traj, window, population=1000, calibrate=True):
    """Computes estimated covariance matrix from allele frequency trajectories."""

    T = len(traj) - 1
    L = len(traj[0])
    variances = computeVariances(traj)
    for t in range(T):
        for l in range(L):
            if variances[t, l] < 0:
                print(f't={t}, l={l}, var={variances[t, l]}')
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
        if not calibrate:
            for i in range(L):
                est_cov[t, i, i] *= population / count_time_point
                for j in range(i):
                    est_cov[t, i, j] *= population / count_time_point
                    est_cov[t, j, i] = est_cov[t, i, j]
        else:
            calibrate_est_cov(est_cov[t], variances[t])
    return est_cov


def calibrate_est_cov(est_cov, variances):
    L = len(est_cov)
    rescaling_multiplier = np.full((L, L), 1, dtype=float)
    for i in range(L):
        # Only when frequency of site i remains unchanged in the time window, could est_cov[i, i] be 0, in which case we preserve the original estimate of covariance.
        if abs(est_cov[i, i]) >= EPSILON:
            for j in range(i):
                if abs(est_cov[i, i] * est_cov[j, j]) >= EPSILON:
                    rescaling_multiplier[i, j] = np.sqrt(variances[i] * variances[j] / (est_cov[i, i] * est_cov[j, j]))
                    rescaling_multiplier[j, i] = rescaling_multiplier[i, j]
    for i in range(L):
        est_cov[i, i] = variances[i]
        for j in range(i):
            est_cov[i, j] *= rescaling_multiplier[i, j]
            est_cov[j, i] = est_cov[i, j]


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


def integrateCovarianceMatrix(traj, times, cov, interpolate=True):
    """Computes integrated covariance matrix."""

    T = len(traj) - 1
    L = len(traj[0])
    int_cov = np.zeros((L, L), dtype=float)
    if interpolate:
        x = computeCooccurenceTrajFromCovariance(traj, cov)
        for t in range(T - 1):
            dg = times[t + 1] - times[t]
            for i in range(L):
                int_cov[i, i] += MPL.integratedVariance(traj[t, i], traj[t + 1, i], dg)
                for j in range(i + 1, L):
                    int_cov[i, j] += MPL.integratedCovariance(traj[t, i], traj[t, j], x[t, i, j], traj[t+1, i], traj[t+1, j], x[t+1, i, j], dg)
        for i in range(L):
            for j in range(i + 1, L):
                int_cov[j, i] = int_cov[i, j]
    else:
        # for t in range(T):
        #     int_cov += (times[t + 1] - times[t]) * cov[t]
        for t in range(T):
            if t == 0:
                dg = times[1] - times[0]
            elif t == len(times) - 1:
                dg = times[t] - times[t - 1]
            else:
                dg = (times[t + 1] - times[t - 1]) / 2
            int_cov += dg * cov[t]
    return int_cov


def computeCooccurenceTrajFromCovariance(traj, cov):
    """Computes cooccurence frequency x_{ij}."""

    T = len(traj) - 1
    L = len(traj[0])
    xixj = np.zeros((T, L, L), dtype=float)
    xij = np.zeros((T, L, L), dtype=float)
    for t in range(T):
        xixj[t] = np.outer(traj[t], traj[t])
        xij[t] = cov[t] + xixj[t]

    return xij


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


def shrinkOffDiagonalTerms(matrix, factor):
    """
    Shrinks off-diagonal terms of a matrix by multiplying a factor.
    """
    copy = np.copy(matrix)
    L = len(copy)
    for i in range(L):
        for j in range(i):
            copy[i, j] *= factor
            copy[j, i] = copy[i, j]
    return copy

############################################
#
# Utility functions
#
############################################

def computeMetricsForCovarianceEstimation(est_cov_mat, true_cov_mat):
    """Computes Spearmanr and MAE error between estimated and true covariance matrices, evaluating
     1. all entries (both covariances and variances)
     2. off-diagonal entries (only covariances)
     3. diagonal entries (only variances)
    """
    L = len(est_cov_mat)
    est_all, true_all = [], []
    for i in range(L):
        est_all.append(est_cov_mat[i, i])
        true_all.append(true_cov_mat[i, i])
    for i in range(L):
        for j in range(i + 1, L):
            est_all.append(est_cov_mat[i, j])
            true_all.append(true_cov_mat[i, j])

    est_all, true_all = np.array(est_all), np.array(true_all)
    spearmanr_all, error_all = computePerformance(est_all, true_all)
    spearmanr_off_diagonal, error_off_diagonal = computePerformance(est_all[L:], true_all[L:])
    spearmanr_diagonal, error_diagonal = computePerformance(est_all[:L], true_all[:L])

    spearmanr_three = np.array([spearmanr_all, spearmanr_off_diagonal, spearmanr_diagonal])
    error_three = np.array([error_all, error_off_diagonal, error_diagonal])
    return spearmanr_three, error_three


def computePerformance(inferred_selection, true_selection):
    """Computes Spearmanr and MAE error between inferred and true selections."""

    spearmanr, _ = stats.spearmanr(inferred_selection, true_selection)
    error = np.mean(np.absolute(inferred_selection - true_selection))
    return spearmanr, error


def get_diagonal_terms(matrix):
    return [matrix[i, i] for i in range(len(matrix))]


def get_off_diagonal_terms(matrix):
    return [matrix[i, j] for i in range(len(matrix)) for j in range(i)]
