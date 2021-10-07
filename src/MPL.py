#!/usr/bin/env python
# Functions related to MPL inference

import numpy as np                          # numerical tools
from copy import deepcopy


def processStandard(sequences, frequencies, times, q, totalCov, covAtEachTime=False, intCovTimes=[]):
    """Integrates covariance from time-series sequence and count data.

    Args:
        sequences: A list of lists of sequences; A time-series of Sequences present in the population.
        frequencies: A list of lists of sequence frequencies corresponding to sequences.
        times: A list of times corresponding to sequences.
        q: Number of states at each locus.
        totalCov: The covariance integrate to be updated.
        covAtEachTime: Optional; Whether or not return covariance at each time point.
        intCovTimes: Optional; A SORTED list pf time points at which we store integrated covariance.

    Returns:
        cov: Optional; covariance at each time point.
        intCovAtTimes: Optional; A list of integrated covariance at time points specified in intCovTimes.
    """
    L = len(totalCov)
    p1 = np.zeros(L, dtype=float)
    p2 = np.zeros((L, L), dtype=float)
    lastp1 = np.zeros(L, dtype=float)
    lastp2 = np.zeros((L, L), dtype=float)
    # store the total_cov at each time point when dynamics is enabled.
    if covAtEachTime:
        cov = np.zeros((len(sequences), L, L), dtype=float)
    elif intCovTimes:
        intCovAtTimes = [np.zeros((L, L), dtype=float) for _ in intCovTimes]

    computeAllelFrequencies(sequences[0], frequencies[0], q, lastp1, lastp2)
    if covAtEachTime:
        cov[0] = computeCovariances(lastp1, lastp2)

    covTimeIndex = 0
    for k in range(1, len(sequences)):
        computeAllelFrequencies(sequences[k], frequencies[k], q, p1, p2)
        updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov)
        if covAtEachTime:
            cov[k] = computeCovariances(p1, p2)
        elif intCovTimes and times[k] <= intCovTimes[covTimeIndex]:
            if k == len(sequences) - 1:
                # There is no next time point, but the projected next time point will pass intCovTimes[covTimeIndex]
                while covTimeIndex < len(intCovTimes) and 2 * times[k] - times[k - 1] > intCovTimes[covTimeIndex]:
                    intCovAtTimes[covTimeIndex] = deepcopy(totalCov)
                    covTimeIndex += 1
            elif times[k + 1] > intCovTimes[covTimeIndex]:
                # Next time point will pass intCovTimes[covTimeIndex]
                while times[k + 1] > intCovTimes[covTimeIndex]:
                    intCovAtTimes[covTimeIndex] = deepcopy(totalCov)
                    covTimeIndex += 1
        if not k == len(sequences) - 1:
            lastp1 = p1
            lastp2 = p2

    if covAtEachTime:
        return cov
    elif intCovTimes:
        return intCovAtTimes


def computeAllelFrequencies(sequences, frequencies, q, p1, p2):
    """Computes allele frequencies and pair-allele frequencies, recording only (q - 1) states.

    Args:
        sequences: A list of sequences present at a time point.
        frequencies: A list of sequence frequencies corresponding to sequences.
        q: Number of states at each locus.
        p1: A Numpy array of allele frequencies to be computed.
        p2: A Numpy array of pair-allele frequencies to be computed.
    """
    p1.fill(0)
    p2.fill(0)

    for k in range(len(sequences)):
        for i in range(len(sequences[k])):
            if not sequences[k, i] == 0:
                a = i * (q - 1) + sequences[k, i] - 1
                p1[a] += frequencies[k]

                for j in range(i + 1, len(sequences[k])):
                    if not sequences[k, j] == 0:
                        b = j * (q - 1) + sequences[k, j] - 1
                        p2[a, b] += frequencies[k]
                        p2[b, a] += frequencies[k]


def computeCovariances(p1, p2):
    """Computes covariances from allele & pair-allele frequencies."""

    L = len(p1)
    cov_tmp = np.zeros((L, L), dtype=float)
    for i in range(L):
        cov_tmp[i, i] = lastp1[i]
        for j in range(i + 1, L):
            cov_tmp[i, j] = lastp2[i, j] - lastp1[i] * lastp1[j]
            cov_tmp[j, i] = cov_tmp[i, j]
    return cov_tmp


def updateCovarianceIntegrate(dg, p1_0, p2_0, p1_1, p2_1, totalCov):
    """Interpolates allele frequencies and adds integrated covariance in a time interval to the integrate.

    Args:
        dg: Width of the time interval.
        p1_0: A Numpy array of allele frequencies at the beginning of the time interval.
        p2_0: A Numpy array of pair-allele frequencies at the beginning of the time interval.
        p1_1: A Numpy array of allele frequencies at the end of the time interval.
        p2_1: A Numpy array of pair-allele frequencies at the end of the time interval.
        totalCov: The covariance integrate to be updated.
    """
    N = len(p1_0)

    for a in range(N):
        totalCov[a, a] += dg * ( ((3 - 2 * p1_1[a]) * (p1_0[a] + p1_1[a])) - 2 * (p1_0[a] * p1_0[a]) ) / 6

        for b in range(a + 1, N):
            dCov1 = -dg * ((2 * p1_0[a] * p1_0[b]) + (2 * p1_1[a] * p1_1[b]) + (p1_0[a] * p1_1[b]) + (p1_1[a] * p1_0[b])) / 6
            dCov2 = dg * 0.5 * (p2_0[a, b] + p2_1[a, b])
            totalCov[a, b] += dCov1 + dCov2
            totalCov[b, a] += dCov1 + dCov2


def computeD(traj, times, mu):
    """Computes the 'Dx' term in MPL inference."""

    T = len(times)
    tStart, tEnd = 0, T - 1
    D = traj[tEnd] - traj[tStart] - mu * np.sum([(times[t + 1] - times[t]) * (1 - 2 * traj[t]) for t in range(tStart, tEnd)], axis=0)
    return D


def inferSelection(int_cov, D, regularization_matrix):
    """Infers selection coefficients.

    Args:
        int_cov: Integrated covariance matrix.
        D: 'Dx' term in MPL inference.
        regularization_matrix: A matrix added to integrated covariance matrix for regularization.
    """
    return np.dot(np.linalg.inv(int_cov + regularization_matrix), D)
