#!/usr/bin/env python

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import MPL                                  # MPL inference tools
from collections import Counter
from itertools import accumulate
from bisect import bisect_right
import importlib
WF = importlib.import_module("Wright-Fisher")


def main(verbose=False):
    """Saves a subsampled version of the input simulated results."""

    parser = argparse.ArgumentParser(description='Wright-Fisher evolutionary simulation.')
    parser.add_argument('-o',         type=str,   default='data/subsample', help='output path and filename')
    parser.add_argument('-i',         type=str,   default=None,             help='input file (dic of W-F.py)')
    parser.add_argument('--sample',   type=int,   default=1000,             help='# of sampled sequences')
    parser.add_argument('--record',   type=int,   default=1,                help='sampling time interval')
    parser.add_argument('--compact',              action='store_true',      default=False,
                        help='whether or not generate a compact output (excluding sVec and nVec)')
    parser.add_argument('--replace',              action='store_true',      default=False,
                        help='whether or not sample with replacement')
    parser.add_argument('--covAtEachTime',        action='store_true',      default=False,
                        help='whether or not record covariance at each time point')
    parser.add_argument('--intCovAtTimes',        action='store_true',      default=False,
                        help='whether or not record integrated covariance at some time points')
    parser.add_argument('--intCovTimes', type=int, default=[200, 300, 400, 500, 600, 700], nargs='*',
                        help='time points at which we store integrated covariance')
    arg_list = parser.parse_args(sys.argv[1:])

    out_file = arg_list.o
    dic_file = arg_list.i
    sample = arg_list.sample
    record = arg_list.record
    intCovTimes = sorted(arg_list.intCovTimes)

    start = timer() # track running time
    print(f'out_file = {out_file}\ndic_file = {dic_file}\nsample = {sample}\nrecord = {record}')

    dic = np.load(dic_file)

    sVec, nVec, times = subsample(dic['sVec'], dic['nVec'], dic['times'], sample, record, replace=arg_list.replace)

    traj = computeTraj(sVec, nVec)

    # Compute the integrated covariance matrix
    q = 2
    L = len(sVec[0][0])
    totalCov = np.zeros((L, L), dtype=float)
    if arg_list.covAtEachTime:
        covAtEachTime = MPL.processStandard(sVec, [i / sample for i in nVec], times, q, totalCov, covAtEachTime=True)
    elif arg_list.intCovAtTimes:
        intCovAtTimes = MPL.processStandard(sVec, [i / sample for i in nVec], times, q, totalCov, intCovTimes=intCovTimes)
    else:
        MPL.processStandard(sVec, [i / sample for i in nVec], times, q, totalCov)

    compact_dic = {'traj': traj, 'cov': totalCov, 'mu': dic['mu'], 'times': times}
    genotype_dic = {'nVec': nVec, 'sVec': sVec}
    f = open(out_file + '.npz', 'wb')
    if arg_list.compact:
        if arg_list.covAtEachTime:
            np.savez_compressed(f, **compact_dic, covAtEachTime=covAtEachTime)
        elif arg_list.intCovAtTimes:
            np.savez_compressed(f, **compact_dic, intCovTimes=intCovTimes, intCovAtTimes=intCovAtTimes)
        else:
            np.savez_compressed(f, **compact_dic)
    else:
        if arg_list.covAtEachTime:
            np.savez_compressed(f, **compact_dic, **genotype_dic, covAtEachTime=covAtEachTime)
        elif arg_list.intCovAtTimes:
            np.savez_compressed(f, **compact_dic, **genotype_dic, intCovTimes=intCovTimes, intCovAtTimes=intCovAtTimes)
        else:
            np.savez_compressed(f, **compact_dic, **genotype_dic)
    f.close()

    end = timer()
    print('\nTotal time: %lfs\n' % (end - start))


def subsample(sVec, nVec, times, sample, record, replace=False):
    """Takes output of Wright-Fisher.py and subsamples the population.

    Args:
        sVec: Genotypes present in the population at each time point.
        nVec: Genotype counts corresponding to sVec.
        times: Time points in unit of generation.
        sample: Number of sequences drawn at each time point.
        record: Interval between each two time points to keep.
    """
    nVec_tmp = []
    sVec_tmp = []
    times_tmp = []
    for t in range(len(times)):
        if times[t] % record == 0:
            nVec_tmp.append(nVec[t])
            sVec_tmp.append(sVec[t])
            times_tmp.append(times[t])

    sVec_tmp, nVec_tmp = sampleSequences(sVec_tmp, nVec_tmp, sample, replace=replace)

    return sVec_tmp, nVec_tmp, times_tmp


def sampleSequences(sVec, nVec, sample, replace=False):
    """Samples a certain number of sequences from the whole population."""

    N = int(np.sum(nVec[0]))
    if sample == N and replace == False:
        return sVec, nVec
    sVec_sampled = []
    nVec_sampled = []
    for t in range(len(nVec)):
        nVec_tmp = []
        sVec_tmp = []
        numbers_sampled = np.random.choice(np.arange(N), size=sample, replace=replace)
        prefix_sum = list(accumulate(nVec[t]))
        indices_sampled = [bisect_right(prefix_sum, n) for n in numbers_sampled]
        index_to_count = Counter(indices_sampled)
        for index, count in index_to_count.items():
            nVec_tmp.append(count)
            sVec_tmp.append(sVec[t][index])
        nVec_sampled.append(np.array(nVec_tmp))
        sVec_sampled.append(np.array(sVec_tmp))
    return sVec_sampled, nVec_sampled


def computeTraj(sVec, nVec):
    """Computes trajectories of allele frequencies."""

    L = len(sVec[0][0])
    T = len(nVec)
    N = np.sum(nVec[0])
    traj = np.zeros((T, L), dtype=float)
    for t in range(T):
        for i in range(L):
            traj[t, i] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / N
    return traj


if __name__ == '__main__': main()
