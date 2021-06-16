#!/usr/bin/env python
# Estimate covariance information from time-series genetic data subsampled to different sampling depths and time intervals.
# Infer selection coefficients with estimated covariance, as well as only variance, and true covariance, as two compares.
# Test two regularization methods: 1. linear shrinkage with various strengths; 2. regularzing correlation with various shrinkers & strengths.

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
from scipy import stats
import math
import MPL                                  # MPL inference tools


def main(verbose=False):
    """Estimates integrated covariance matrix and infer selection coefficients using various methods;
    Investigates their robustness against limited-sampling.
    """

    # Read in simulation parameters from command line

    parser = argparse.ArgumentParser(description='Estimate int-cov-mat and infer selection coefficients.')

    parser.add_argument('-N',               type=int,       default=1000,       help='population size')
    parser.add_argument('-L',               type=int,       default=50,         help='sequence length')
    parser.add_argument('-T',               type=int,       default=700,        help='# of generations in simulation')
    parser.add_argument('-truncate',        type=int,       default=700,        help='# of generations used for estimation and inference, counting from the first generation')
    parser.add_argument('-window',          type=int,       default=1,          help='window parameter in the estimation algorithm')
    parser.add_argument('-num_selections',  type=int,       default=10,         help='# of different selections')
    parser.add_argument('-num_trials',      type=int,       default=20,         help='# of trials for each selection')
    parser.add_argument('--sample',         type=int,       default=[1000, 500, 100, 50, 10], nargs='*',
                        help='list of sampling depths')
    parser.add_argument('--record',         type=int,       default=[1, 3, 5, 10], nargs='*',
                        help='list of sampling time intervals')
    parser.add_argument('--linear',         type=int,       default=[5, 10, 20, 50, 100, 300], nargs='*',
                        help='list of linear shrinkage strengths')
    parser.add_argument('--gamma',          type=int,       default=[5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 1], nargs='*',
                        help='list of non-linear shrinkage parameter choices')
    parser.add_argument('--loss',           type=str,       nargs='*',
                        default=['Fro 1', 'Fro 2', 'Fro 3', 'Fro 4', 'Fro 5', 'Nuc 1', 'Nuc 2', 'Nuc 3', 'Nuc 4', 'Nuc 5'],
                        help='list of loss functions for non-linear shrinkage')
    parser.add_argument('-s',               type=str,       default=None,
                        help="prefix of .npy file containning selection matrix (filename is arg_list.s + f'_*.npy')")
    parser.add_argument('-i',               type=str,       default=None,
                        help="prefix of .npz file containning input (filename is arg_list.i + f'_s=*_n=*_sample=*_record=*.npz')")
    parser.add_argument('-o',               type=str,       default=None,
                        help="prefix of output .npz file (filename is arg_list.o + f'_truncate=*_window=*.npz')")
    parser.add_argument('--compact_size',   action='store_true', default=False,
                        help='whether or not generate a compact-size output (only selections, spearmanr & error)')
    parser.add_argument('--medium_size',    action='store_true', default=False,
                        help='whether or not generate a medium-size output (selections, spearmanr & error, and covariance without subsampling, and with selected loss and gamma options for non-linear shrinkage)')
    parser.add_argument('-sample_selected', type=int,            default=0,
                        help='one of sampling depths whose result will be reported when we want medium-size output')
    parser.add_argument('-record_selected', type=int,            default=0,
                        help='one of sampling intervals whose result will be reported when we want medium-size output')
    parser.add_argument('-loss_selected',   type=int,            default=3,
                        help='one of loss functions whose result will be reported when we want medium-size output')
    parser.add_argument('-gamma_selected',  type=int,            default=0,
                        help='one of gamma values whose result will be reported when we want medium-size output')
    arg_list = parser.parse_args(sys.argv[1:])

    N               = arg_list.N
    L               = arg_list.L
    T               = arg_list.T
    truncate        = arg_list.truncate
    window          = arg_list.window
    num_selections  = arg_list.num_selections
    num_trials      = arg_list.num_trials
    sample          = arg_list.sample
    record          = arg_list.record
    linear          = arg_list.linear
    gamma           = arg_list.gamma
    loss            = arg_list.loss
    loss_selected   = arg_list.loss_selected
    gamma_selected  = arg_list.gamma_selected
    sample_selected = arg_list.sample_selected
    record_selected = arg_list.record_selected

    selections = np.zeros((num_selections, L), dtype=float)
    for s in range(num_selections):
        selections[s] = np.load(arg_list.s + f'_{s}.npy')

    IdentityMatrix = np.zeros((L, L), dtype=float)
    UnityMatrix = np.full((L, L), 1)
    for i in range(L):
        IdentityMatrix[i, i] = 1

    start  = timer()  # track running time

    print(f'sample = {sample}\nrecord = {record}\nlinear = {linear}\ngamma={gamma}\nloss={loss}\n')
    print(f'truncate = {truncate}\nwindow = {window}\n')

    common_shape = (num_selections, num_trials, len(sample), len(record))
    combine_shape = (num_selections, len(sample), len(record))
    cov_shape = common_shape + (L, L)
    dcorr_shape = common_shape + (len(loss), len(gamma), L, L)

    # integrated (regularized) covariance & correlation
    int_cov = np.zeros(cov_shape, dtype=float)
    int_cov_SL = np.zeros(cov_shape, dtype=float)
    int_corr = np.zeros(cov_shape, dtype=float)
    int_dcov = np.zeros(cov_shape, dtype=float)
    int_dcov_uncalibrated = np.zeros(cov_shape, dtype=float)
    int_dcorr = np.zeros(cov_shape, dtype=float)
    int_dcorr_reg = np.zeros(dcorr_shape, dtype=float)
    int_dcov_dcorr_reg = np.zeros(dcorr_shape, dtype=float)
    D = np.zeros(common_shape + (L, ), dtype=float)

    # 4 basic methods without further regularization: SL, MPL, est, est(uncalibrated)
    num_basic = 4
    selections_basic = np.zeros(common_shape + (num_basic, L), dtype=float)
    spearmanr_basic = np.zeros(common_shape + (num_basic, ), dtype=float)
    error_basic = np.zeros(common_shape + (num_basic, ), dtype=float)
    # combining multiple replicates
    selections_basic_combine = np.zeros(combine_shape + (num_basic, L), dtype=float)
    spearmanr_basic_combine = np.zeros(combine_shape + (num_basic, ), dtype=float)
    error_basic_combine = np.zeros(combine_shape + (num_basic, ), dtype=float)

    # linear shrinkage on covariance matrix
    selections_linear = np.zeros(common_shape + (len(linear), L), dtype=float)
    spearmanr_linear = np.zeros(common_shape + (len(linear), ), dtype=float)
    error_linear = np.zeros(common_shape + (len(linear), ), dtype=float)
    # combining multiple replicates
    selections_linear_combine = np.zeros(combine_shape + (len(linear), L), dtype=float)
    spearmanr_linear_combine = np.zeros(combine_shape + (len(linear), ), dtype=float)
    error_linear_combine = np.zeros(combine_shape + (len(linear), ), dtype=float)

    # non-linear shrinkage on correlation matrix
    selections_dcorr = np.zeros(common_shape + (len(loss), len(gamma), L), dtype=float)
    spearmanr_dcorr = np.zeros(common_shape + (len(loss), len(gamma)), dtype=float)
    error_dcorr = np.zeros(common_shape + (len(loss), len(gamma)), dtype=float)
    # combining multiple replicates
    selections_dcorr_combine = np.zeros(combine_shape + (len(loss), len(gamma), L), dtype=float)
    spearmanr_dcorr_combine = np.zeros(combine_shape + (len(loss), len(gamma)), dtype=float)
    error_dcorr_combine = np.zeros(combine_shape + (len(loss), len(gamma)), dtype=float)


    for s in range(num_selections):
        print(f'\nIn progress: s = {s}', end=':\t')
        for n in range(num_trials):
            print(f'{n}', end=';')
            for p in range(len(sample)):
                for q in range(len(record)):
                    N, L, T = arg_list.N, arg_list.L, arg_list.T
                    str_case_info = f'Case s={s}, n={n}, sample={sample[p]}, record={record[q]}. '
                    str_no_precomputed_cov = f'Pre-computed intCov not found in subsampled dic file. '

                    dic = np.load(arg_list.i + f'_s={s}_n={n}_sample={sample[p]}_record={record[q]}.npz')
                    mu = dic['mu']
                    truncated_length = np.sum([t % record[q] == 0 for t in range(truncate + 1)])
                    traj = dic['traj'][:truncated_length]
                    times = dic['times'][:truncated_length]
                    assert len(times) == len(traj), f'len(times) {len(times)} does not equal to leng(traj) {len(traj)} after truncation!'

                    # true integrated covariance
                    if 'intCovTimes' in dic.keys() and 'intCovAtTimes' in dic.keys() and truncate in dic['intCovTimes']:
                        index = list(dic['intCovTimes']).index(truncate)
                        int_cov[s, n, p, q] = dic['intCovAtTimes'][index]
                    elif 'nVec' in dic.keys() and 'sVec' in dic.keys():
                        print(str_case_info + str_no_precomputed_cov + f'Computing using nVec and sVec...')
                        q = 2
                        MPL.processStandard(dic['sVec'], [i / N for i in dic['nVec']], times, q, int_cov[s, n, p, q])
                    else:
                        print(str_case_info + str_no_precomputed_cov + f'sVec & nVec not found either; Cannot retrieve true integrated covariance!')
                        return

                    int_corr[s, n, p, q] = computeCorrelation(int_cov[s, n, p, q])
                    int_cov_combine = np.mean(int_cov[s, :, p, q], axis=0)
                    int_corr_combine = np.mean(int_corr[s, :, p, q], axis=0)

                    # SL matrix: ignoring all covariances; Keep only variances
                    for k in range(L):
                        int_cov_SL[s, n, p, q, k, k] = int_cov[s, n, p, q, k, k]
                    int_cov_SL_combine = np.mean(int_cov_SL[s, :, p, q], axis=0)

                    # estimated integrated covariance
                    int_dcov[s, n, p, q] = estimateIntegratedCovarianceMatrix(traj, window, population=N)
                    int_dcorr[s, n, p, q] = computeCorrelation(int_dcov[s, n, p, q])
                    int_dcov_combine = np.mean(int_dcov[s, :, p, q], axis=0)
                    int_dcorr_combine = computeCorrelation(int_dcov_combine)

                    # estimated integrated covariance (uncalibrated)
                    int_dcov_uncalibrated[s, n, p, q]= estimateIntegratedCovarianceMatrix(traj, window, population=N, calibration=False)
                    int_dcov_uncalibrated_combine = np.mean(int_dcov_uncalibrated[s, :, p, q], axis=0)

                    # Dx term in MPL inference
                    D[s, n, p, q] = MPL.computeD(traj, times, mu)
                    D_combine = np.mean(D[s, :, p, q], axis=0)

                    # basic methods: SL, MPL, est, est(uncalibrated)
                    cov_list = [int_cov_SL[s, n, p, q], int_cov[s, n, p, q], int_dcov[s, n, p, q], int_dcov_uncalibrated[s, n, p, q]]
                    cov_combine_list = [int_cov_SL_combine, int_cov_combine, int_dcov_combine, int_dcov_uncalibrated_combine]
                    for i in range(num_basic):
                        selections_basic[s, n, p, q, i] = MPL.inferSelection(cov_list[i], D[s, n, p, q], 1 * IdentityMatrix)
                        selections_basic_combine[s, p, q, i] = MPL.inferSelection(cov_combine_list[i], D_combine, 1 * IdentityMatrix)
                        (spearmanr_basic[s, n, p, q, i],
                         error_basic[s, n, p, q, i]) = computePerformance(selections_basic[s, n, p, q, i], selections[s])
                        (spearmanr_basic_combine[s, p, q, i],
                         error_basic_combine[s, p, q, i]) = computePerformance(selections_basic_combine[s, p, q, i], selections[s])

                    # Linear shrinkage on integrated covariance
                    for r in range(len(linear)):
                        selections_linear[s, n, p, q, r] = MPL.inferSelection(int_dcov[s, n, p, q], D[s, n, p, q], linear[r] * IdentityMatrix)
                        selections_linear_combine[s, p, q, r] = MPL.inferSelection(int_dcov_combine, D_combine, linear[r] * IdentityMatrix)
                        (spearmanr_linear[s, n, p, q, r],
                         error_linear[s, n, p, q, r]) = computePerformance(selections_linear[s, n, p, q, r], selections[s])
                        (spearmanr_linear_combine[s, p, q, r],
                         error_linear_combine[s, p, q, r]) = computePerformance(selections_linear_combine[s, p, q, r], selections[s])

                    # Non-linear shrinkage on integrated correlation matrix
                    multiplier_corr_to_cov = computeCorrToCovMultiplier(int_dcov[s, n, p, q])
                    multiplier_corr_to_cov_combine = computeCorrToCovMultiplier(int_dcov_combine)

                    for l in range(len(loss)):
                        for g in range(len(gamma)):
                            int_dcorr_reg[s, n, p, q, l, g] = shrinkCorrelation(int_dcorr[s, n, p, q], l, gamma[g])
                            int_dcov_dcorr_reg[s, n, p, q, l, g] = np.linalg.multi_dot([multiplier_corr_to_cov,
                                                                                        int_dcorr_reg[s, n, p, q, l, g], multiplier_corr_to_cov])
                            selections_dcorr[s, n, p, q, l, g] = MPL.inferSelection(int_dcov_dcorr_reg[s, n, p, q, l, g],
                                                                                    D[s, n, p, q], 1 * IdentityMatrix)
                            (spearmanr_dcorr[s, n, p, q, l, g],
                             error_dcorr[s, n, p, q, l, g]) = computePerformance(selections_dcorr[s, n, p, q, l, g], selections[s])

                            # There are two possible ways to combine replicates with non-linear shrinkage regularization.
                            # Below is the 1st way: regularize the mean of replicates.
                            int_dcorr_reg_combine = shrinkCorrelation(int_dcorr_combine, l, gamma[g])
                            int_dcov_dcorr_reg_combine = np.linalg.multi_dot([multiplier_corr_to_cov_combine,
                                                                              int_dcorr_reg_combine,
                                                                              multiplier_corr_to_cov_combine])
                            # Below is the 2nd way: regularize each replicate individually, then take their average.
                            # int_dcov_dcorr_reg_combine = np.mean(int_dcov_dcorr_reg[s, :, p, q, l, g], axis=0)
                            selections_dcorr_combine[s, p, q, l, g] = MPL.inferSelection(int_dcov_dcorr_reg_combine,
                                                                                         D_combine, 1 * IdentityMatrix)
                            (spearmanr_dcorr_combine[s, p, q, l, g],
                             error_dcorr_combine[s, p, q, l, g]) = computePerformance(selections_dcorr_combine[s, p, q, l, g], selections[s])

    selections_dic   = {'selections_basic': selections_basic, 'selections_basic_combine': selections_basic_combine,
                        'selections_linear': selections_linear, 'selections_linear_combine': selections_linear_combine,
                        'selections_dcorr': selections_dcorr, 'selections_dcorr_combine': selections_dcorr_combine}
    performances_dic = {'spearmanr_basic': spearmanr_basic, 'spearmanr_basic_combine': spearmanr_basic_combine,
                        'spearmanr_linear': spearmanr_linear, 'spearmanr_linear_combine': spearmanr_linear_combine,
                        'spearmanr_dcorr': spearmanr_dcorr, 'spearmanr_dcorr_combine': spearmanr_dcorr_combine,
                        'error_basic': error_basic, 'error_basic_combine': error_basic_combine,
                        'error_linear': error_linear, 'error_linear_combine': error_linear_combine,
                        'error_dcorr': error_dcorr, 'error_dcorr_combine': error_dcorr_combine}
    cov_complete_dic = {'int_cov': int_cov, 'int_dcov': int_dcov,
                        'int_dcov_uncalibrated': int_dcov_uncalibrated, 'int_dcov_dcorr_reg': int_dcov_dcorr_reg}
    cov_selected_dic = {'int_cov': int_cov[:, :, sample_selected, record_selected],
                        'int_dcov': int_dcov[:, :, sample_selected, record_selected],
                        'int_dcov_uncalibrated': int_dcov_uncalibrated[:, :, sample_selected, record_selected],
                        'int_dcov_dcorr_reg': int_dcov_dcorr_reg[:, :, sample_selected, record_selected, loss_selected, gamma_selected]}

    f = open(arg_list.o + f'_truncate={truncate}_window={window}.npz', 'wb')
    if arg_list.compact_size:
        np.savez_compressed(f, **selections_dic, **performances_dic, size="compact")
    elif arg_list.medium_size:
        np.savez_compressed(f, **selections_dic, **performances_dic, **cov_selected_dic, size="medium")
    else:
        np.savez_compressed(f, **selections_dic, **performances_dic, **cov_complete_dic, size="complete")
    f.close()

    end = timer()
    print('\nTotal time: %lfs' % (end - start))


def computeD(traj, times, mu):
    """Computes the 'Dx' term in MPL inference."""

    T = len(times)
    tStart, tEnd = 0, T - 1
    return traj[tEnd] - traj[tStart] - mu * np.sum([(times[t + 1] - times[t]) * (1 - 2 * traj[t]) for t in range(tStart, tEnd)], axis=0)


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


def computePerformance(inferred_selection, true_selection):
    """Computes Spearmanr and Frobenius error between inferred and true selections."""

    spearmanr, _ = stats.spearmanr(inferred_selection, true_selection)
    error = np.linalg.norm(inferred_selection - true_selection)
    return spearmanr, error


def computeCorrToCovMultiplier(cov):
    """Computes the multiplier by which, multiplier x correlation x multiplier = covariance."""

    multiplier = np.zeros_like(cov)
    for t in range(len(cov)):
        multiplier[t, t] = math.sqrt(cov[t, t])
    return multiplier


def estimateIntegratedCovarianceMatrix(traj, window, population=1000, calibration=True):
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

    int_est_cov = np.sum(est_cov, axis=0)
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


if __name__ == '__main__': main()