#!/usr/bin/env python
# Parse time-series genetic data from multiple papers/sources

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
from scipy import stats
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import MPL                                  # MPL inference tools
import simulation_setup as SS
import estimate_covariance as EC
sys.path.append('../evoracle/')  # path to Evoracle
import evoracle

SIMULATION_OUTPUT_DIR = './data/simulation_output'
SUBSAMPLE_OUTPUT_DIR = './data/subsample_output'
ESTIMATION_OUTPUT_DIR = './data/estimation_output'

LOLIPOP_INPUT_DIR = './data/lolipop/input'
LOLIPOP_OUTPUT_DIR = './data/lolipop/output'
LOLIPOP_PARSED_OUTPUT_DIR = './data/lolipop/parsed_output'

EVORACLE_DIR = './data/evoracle'
EVORACLE_DIR_PALTE = './data/evoracle/PALTE'
EVORACLE_DIR_SIMULATION = './data/evoracle/simulation'

HAPLOSEP_INPUT_DIR = './data/haploSep/input'
HAPLOSEP_OUTPUT_DIR = './data/haploSep/output'

SHEN_DATA_DIR = './data/Shen2021-ew'
Cry1Ac_OUTPUT_DIR = './data/Shen2021-ew/cry1ac_output'

PALTE_DIR = './data/PALTEanalysis'
PALTE_TRAJ_FILE = f'{PALTE_DIR}/Table_S3.csv'
PALTE_FITNESS_FILE = f'{PALTE_DIR}/Table_S1.xlsx'
PALTE_SHEETNAMES = ['selection rates 0-24 hr', 'selection rates 24-48 hr',  'selection rates 0-48 hr']
PALTE_CONDITIONS = ['Plank', 'biof - bead', 'biof - plank portion']
PALTE_SELECTED_SHEET = 'selection rates 0-24 hr'
PALTE_SELECTED_CONDITIONS = ['biof - bead', 'biof - plank portion']

PALTE_POPULATIONS = ['B1', 'B2', 'B3', 'P1', 'P2', 'P3']
PALTE_MEASURED_POPULATIONS = ['WT'] + PALTE_POPULATIONS
METHODS = ['recovered', 'Lolipop', 'est_cov', 'SL']
PALTE_TIMES = np.array([0, 17, 25, 44, 66, 75, 90]) * 600 // 90
PALTE_MU = 1e-6
REG = 4

MCDONALD_DATA_DIR = './data/McDonald2016-gh'
MCDONALD_FILE = f'{MCDONALD_DATA_DIR}/41586_2016_BFnature17143_MOESM54_ESM.tsv'
MCDONALD_TIME_COLUMNS = ['Gen0', 'Gen90', 'Gen180', 'Gen270', 'Gen360', 'Gen450', 'Gen540', 'Gen630', 'Gen720', 'Gen810', 'Gen900', 'Gen990']
MCDONALD_TIMES = [int(_[3:]) for _ in MCDONALD_TIME_COLUMNS]

BORDERIA_DATA_DIR = './data/Borderia2015-eo'
BORDERIA_FILE = f'{BORDERIA_DATA_DIR}/ppat.1004838.s004.xlsx'
BORDERIA_SHEETNAMES = [f'replicate {i}' for i in range(1, 7)]
# BORDERIA_TIME_COLUMNS = [f'p{i}' for i in range(1, 41)]
# BORDERIA_TIME_COLUMNS_rpl3 = [f'p{i}' for i in range(1, 41)]
# BORDERIA_TIMES = np.arange(1, 41)


KESSINGER_DATA_DIR = './data/kessinger2013-ou'
KESSINGER_FILE = f'{KESSINGER_DATA_DIR}/CH40longitudinal.fasta'

SIMULATION_PARAMS = {
    'mutation_rate': 1e-3,
    'linear_reg': 1,
    'times': None,
}
Cry1Ac_PARAMS = {
    'times': [0, 11, 23, 34, 46, 58, 69, 81, 92, 104, 116, 127, 139, 150, 162, 174, 185, 197, 209, 220, 232, 255, 267, 290, 313, 336, 360, 383, 394, 418, 441, 464, 487, 511],
    'mean_read_depth': 500,
    'num_timepoints': 34,
    'generation_per_hour': 511 / 528,
    'mutation_rate': 1e-8,  # Taking 1e-10 as the normal mutation rate of E. Coli, the mutation rate in Cry1Ac data is increased by 100-fold by PACE.
    'window': 7,
    'nonlinear_reg': True,
    'linear_reg': 10,
}
TadA_PARAMS = {
    'mean_read_depth': 110461,
    'num_timepoints': 36,
    'mutation_rate': 1e-8,  # PANCE
    'window': 7,
    'nonlinear_reg': False,
    'linear_reg': 1,
}
PALTE_PARAMS = {
    'mutation_rate': 1e-6,
    'times': PALTE_TIMES,
    'methods': ['SL', 'Est', 'Lolipop', 'Evoracle', 'haploSep'],
    'window': 2,
    'nonlinear_reg': True,
    'linear_reg': 1,
}
MCDONALD_PARAMS = {
    'mutation_rate': 1e-10,
    'times': MCDONALD_TIMES,
    'methods': ['SL', 'Est', 'Lolipop', 'Evoracle', 'haploSep'],
    'window': 3,
    'nonlinear_reg': True,
    'linear_reg': 1,
}
BORDERIA_PARAMS = {
    'mutation_rate': 1e-5,  # Not sure, an estimate
    'methods': ['SL', 'Est', 'Lolipop', 'Evoracle', 'haploSep'],
    'window': 2,
    'nonlinear_reg': True,
    'linear_reg': 1,
}
KESSINGER_PARAMS = {
    'mutation_rate': 5e-5,  # HIV-1 has been found to mutate on the order of 10−4 to 10−5 mutations/base pair/cycle (m/bp/c)
    'methods': ['SL', 'Est', 'Lolipop', 'Evoracle', 'haploSep'],
    'window': 2,
    'nonlinear_reg': True,
    'linear_reg': 1,
}

N = 1000        # number of sequences in the population
L = 50          # sequence length
T = 700         # number of generations
MU = 1e-3       # mutation rate
NUM_SELECTIONS = 10  # number of sets of selection coefficients that we used
NUM_TRIALS = 20  # for each set of selection coefficients, we simulated for 20 times
SAMPLE = [1000, 500, 100, 50, 10]  # sampling depth options when subsampling
RECORD = [1, 3, 5, 10]  # sampling time interval options when subsampling
LINEAR = [1, 5, 10, 20, 50, 100, 300]  # list of linear shrinkage strengths/sampling time intervals, e.g., interval=5, linear=5 ==> strength=25'
GAMMA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # non-linear shrinkage parameter choices that we tested
TRUNCATE = [200, 300, 400, 500, 600, 700]  # truncate options to test performance using different lengths of data
WINDOW = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 160]  # window choices that we tested
# loss functions for non-linear shrinkage that we tested
LOSS = ['Fro | $\hat{C}-C$', 'Fro | $\hat{C}^{-1}-C^{-1}$', 'Fro | $C^{-1}\hat{C}-I$',
        'Fro | $\hat{C}^{-1}C-I$', 'Fro | $\hat{C}^{-1/2}C\hat{C}^{-1/2}-I$',
        'Nuc | $\hat{C}-C$', 'Nuc | $\hat{C}^{-1}-C^{-1}$', 'Nuc | $C^{-1}\hat{C}-I$',
        'Nuc | $\hat{C}^{-1}C-I$', 'Nuc | $\hat{C}^{-1/2}C\hat{C}^{-1/2}-I$']

SHRINKING_FACTORS = np.arange(0, 1, 0.01)
LINEAR_SELECTED = 2
LOSS_SELECTED = 3
GAMMA_SELECTED = 0
SHRINK_SELECTED = 50  # Shrinking factor = 0.5
DCORR_SHRINK_SELECTED = 80  # Shrinking factor = 0.8

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.5)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN   = cm2inch(17.4)

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4
SUBLABELS    = ['A', 'B', 'C', 'D', 'E', 'F']
FREQUENCY_YLIM = [-0.04, 1.04]
GENERATION_XLIM = [-40, 1010]

# paper style

FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 10
SIZELEGEND   = 6
SIZEANNOTATION = 6
SIZEANNOTATION_SINGLE = 4
SIZEANNOTATION_HEATMAP = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZEDOT      = 12
SIZELINE     = 0.6

COLORS = sns.husl_palette(20)
BKCOLOR  = '#252525'
GREY_COLOR_RGB = (0.5, 0.5, 0.5)
GREY_COLOR_HEX = '#808080'

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}
DEF_TICKPROPS = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : True,
    'left'      : True,
    'top'       : False,
    'right'     : False,
}


############################################
#
# haploSep
#
############################################

def save_traj_for_haplosep(traj, filename):
    np.save(f"{HAPLOSEP_INPUT_DIR}/{filename}", traj.T)


def parse_haplosep_output(filename):

    res = np.load(f"{HAPLOSEP_OUTPUT_DIR}/{filename}", allow_pickle=True)
    dic = res['arr_0'].item()
    genotypes = dic['haploStr'].T.astype(int)
    geno_traj = dic['haploFrq'].T
    return genotypes, geno_traj


def infer_with_haplosep_output(output, traj, times=None, reg=1, mu=1e-3, compute_population_fitness=False):

    T, L = traj.shape
    result = {}

    genotypes, geno_traj = parse_haplosep_output(output)
    if times is None:
        times = np.arange(0, len(geno_traj))

    int_cov = MPL.integrateCovarianceFromStableGenotypes(genotypes, geno_traj, times)
    traj_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, genotypes)
    D = MPL.computeD(traj, times, mu)
    D_from_geno_traj = MPL.computeD(traj_from_geno_traj, times, mu)
    selection = MPL.inferSelection(int_cov, D, reg * np.identity(L))
    selection_from_geno_traj = MPL.inferSelection(int_cov, D_from_geno_traj, reg * np.identity(L))

    result = {
        'genotypes': genotypes,
        'geno_traj': geno_traj,
        'traj': traj,
        'traj_from_geno_traj': traj_from_geno_traj,
        'int_cov': int_cov,
        'selection': selection,
        'selection_from_geno_traj': selection_from_geno_traj,
    }

    if compute_population_fitness:
        fitness = MPL.computePopulationFitnessFromSelection(traj, selection)
        fitness_from_geno_traj = MPL.computePopulationFitnessFromSelection(traj_from_geno_traj, selection_from_geno_traj)
        result['fitness'] = fitness
        result['fitness_from_geno_traj'] = fitness_from_geno_traj

    return result


############################################
#
# Evoracle
#
############################################

def save_traj_for_evoracle(traj, file, times=None):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    dic = {t: [] for t in times}
    dic['Symbols and linkage group index'] = []
    for j in range(len(traj[0])):
        dic['Symbols and linkage group index'].append(f'M {j}')  # Can use any character other than '.'
        dic['Symbols and linkage group index'].append(f'. {j}')
        for i, t in enumerate(times):
            key = t
            dic[key].append(traj[i, j])
            dic[key].append(1 - traj[i, j])

    df = pd.DataFrame(dic)
    df.to_csv(file, index=False)


def parse_obs_reads_df(file):
    df = pd.read_csv(f'{file}')
    return df


def parse_evoracle_results(obs_reads_file, output_directory, params=SIMULATION_PARAMS, save_geno_traj=True, compute_genotype_fitness=False):
    mu = params['mutation_rate']
    reg = params['linear_reg']
    times = params['times']

    df_traj = pd.read_csv(f'{output_directory}/{obs_reads_file}')
    traj = parse_evoracle_traj(df_traj)
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)

    df_geno = pd.read_csv(f'{output_directory}/_final_genotype_matrix.csv')
    binary_genotypes, geno_traj = parse_evoracle_geno_traj(df_geno)

    traj_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, binary_genotypes)

    int_cov = MPL.integrateCovarianceFromStableGenotypes(binary_genotypes, geno_traj, times)
    D = MPL.computeD(traj, times, mu)
    D_from_geno_traj = MPL.computeD(traj_from_geno_traj, times, mu)
    selection = MPL.inferSelection(int_cov, D, reg * np.identity(L))
    selection_from_geno_traj = MPL.inferSelection(int_cov, D_from_geno_traj, reg * np.identity(L))

    results = {
        'traj': traj,
        'genotypes': binary_genotypes,
        'traj_from_geno_traj': traj_from_geno_traj,
        'int_cov': int_cov,
        'selection': selection,
        'selection_from_geno_traj': selection_from_geno_traj,
    }
    if save_geno_traj:
        results['geno_traj'] = geno_traj
    if compute_genotype_fitness:
        fitness = MPL.computeGenotypeFitnessFromSelection(binary_genotypes, selection)
        fitness_from_geno_traj = MPL.computeGenotypeFitnessFromSelection(binary_genotypes, selection_from_geno_traj)
        results['fitness'] = fitness
        results['fitness_from_geno_traj'] = fitness_from_geno_traj
    return results


def save_evoracle_results(results, file, complete=False):

    if complete:
        np.savez_compressed(file, **results)
    else:
        results_compact = {key: results[key] for key in ['traj', 'int_cov', 'selection']}
        np.savez_compressed(file, **results_compact)


def parse_evoracle_traj(df_traj):
    return df_traj.to_numpy()[0::2, :-1].T.astype(float)


def parse_evoracle_geno_traj(df_geno):

    K = len(df_geno)
    time_columns = list(df_geno.columns)[1:]
    T = len(time_columns)
    genotypes = list(df_geno['Unnamed: 0'])
    binary_genotypes = [binarize_genotype(genotype) for genotype in genotypes]
    geno_traj = np.zeros((T, K))
    for k, row in df_geno.iterrows():
        freqs = [row[_] for _ in time_columns]
        geno_traj[:, k] = freqs
    return binary_genotypes, geno_traj


############################################
#
# Lolipop
#
############################################

def generate_lolipop_command_for_simulated_data(s, n):

    input = f'{SIMULATION_DIR}/simulation_output_s={s}_n={n}'

    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def generate_lolipop_command(input, output):
    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def load_parsed_output(file):
    npz_dic = np.load(file, allow_pickle=True)
    dic_keys = ['genotypeToMembers', 'memberToGenotype', 'genotypeToParent', 'parentToGenotype']
    all_keys = list(npz_dic.keys())
    data = {}
    for key in all_keys:
        if key in dic_keys:
            data[key] = npz_dic[key][()]
        else:
            data[key] = npz_dic[key]
    return data


def inference_pipeline(data, mu=1e-10, regularization=1):
    # times, traj, genotypeToMembers, memberToGenotype, genotypeToParent = data['times'], data['traj'], data['genotypeToMembers'][()], data['memberToGenotype'][()], data['genotypeToParent'][()]
    times, traj, genotypeToMembers, memberToGenotype, genotypeToParent = data['times'], data['traj'], data['genotypeToMembers'], data['memberToGenotype'], data['genotypeToParent']
    intCov = computeCovariance(times, traj, genotypeToMembers, memberToGenotype, genotypeToParent)
    selection = inferSelection(times, traj, intCov, mu=mu, regularization=regularization)
    fitness = computeFitness(traj, selection)
    return intCov, selection, fitness


def computeCovariance(times, traj, genotypeToMembers, memberToGenotype, genotypeToParent, tStart=None, tEnd=None, interpolate=True, timing=False):

    T, L = traj.shape
    intCov = np.zeros((L, L), dtype=float)
    if tStart is None:
        tStart = 0
    if tEnd is None:
        tEnd = T

    x = getCooccurenceTraj(traj, memberToGenotype, genotypeToParent)

    if timing:
        start = timer()
    if interpolate:
        for t in range(tStart, tEnd - 1):
            dg = times[t + 1] - times[t]
            for i in range(L):
                intCov[i, i] += MPL.integratedVariance(traj[t, i], traj[t + 1, i], dg)
                for j in range(i + 1, L):
                    intCov[i, j] += MPL.integratedCovariance(traj[t, i], traj[t, j], x[t, i, j], traj[t+1, i], traj[t+1, j], x[t+1, i, j], dg)
    else:
        for t in range(tStart, tEnd):
            if t == 0:
                dg = times[1] - times[0]
            elif t == len(times) - 1:
                dg = times[t] - times[t - 1]
            else:
                dg = (times[t + 1] - times[t - 1]) / 2
            for i in range(L):
                intCov[i, i] += dg * (traj[t, i] * (1 - traj[t, i]))
                for j in range(i + 1, L):
                    intCov[i, j] += dg * (x[t, i, j] - traj[t, i] * traj[t, j])

    for i in range(L):
        for j in range(i + 1, L):
            intCov[j, i] = intCov[i, j]

    if timing:
        print('Computing covariance... took %.3fs' % (timer() - start))

    return intCov


def getCooccurenceTraj(traj, memberToGenotype, genotypeToParent, timing=False):
    """
    Return cooccurence frequency trajectories of all pairs.
    """
    if timing:
        start = timer()
    T, L = traj.shape
    x = np.zeros((T, L, L), dtype=float)
    for i in range(L):
        g_i = memberToGenotype[i]
        for j in range(i + 1, L):
            g_j = memberToGenotype[j]
            if isCorrelated(g_i, g_j, genotypeToParent):
                # Correlated
                for t in range(T):
                    x[t, i, j] = min(traj[t, i], traj[t, j])
            else:
                # Anti-correlated
                for t in range(T):
                    x[t, i, j] = max(0, traj[t, i] + traj[t, j] - 1)
            for t in range(T):
                x[t, j, i] = x[t, i, j]
    if timing:
        print('Computing cooccurence freq... took %.3fs' % (timer() - start))
    return x


def isCorrelated(g_i, g_j, genotypeToParent, debug=False):
    """
    Return if genotype i and genotype j are correlated (linked), e.g., one arises from the other.
    """
    if g_i == g_j:
        return True
    p_i = g_i
    if debug:
        print('Tracing back from genotype ', p_i, end='->')
    while p_i in genotypeToParent:
        p_i = genotypeToParent[p_i]
        if debug:
            print(p_i, end='->')
        if p_i == g_j:
            return True

    p_j = g_j
    if debug:
        print('\nTracing back from genotype ', p_j, end='->')
    while p_j in genotypeToParent:
        p_j = genotypeToParent[p_j]
        if debug:
            print(p_j, end='->')
        if p_j == g_i:
            return True
    if debug:
        print()
    return False


def inferSelection(times, traj, intCov, mu=None, regularization=1):
    if mu is None:
        print('Mutation rate not provided. Cannot infer selection with MPL.')
        return
    T, L = traj.shape
    D = MPL.computeD(traj, times, mu)
    selection = MPL.inferSelection(intCov, D, regularization * np.identity(L))
    return selection


def computeFitness(traj, selection, initial_fitness=1, normalizeFitness=False):
    """
    Computes fitness of the population throughout the evolution.
    """
    T, L = traj.shape
    fitness = initial_fitness + np.sum(selection * traj, axis=1)
    if normalizeFitness:
        fitness /= np.max(fitness)
    return fitness


############################################
#
# Simulated data
#
############################################

# def load_performance(truncate=700, window=20):
#     return np.load(f'{ESTIMATION_OUTPUT_DIR}/estimation_output_truncate={truncate}_window={window}.npz')

def load_performance(window=20, varying_initial=False, recombination=False, r=1e-5):
    """Loads spearmanr & error of inferred selections with a certain window choice."""

    spearmanr_cov_normalized, spearmanr_cov_unnormalized, spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_shrink, spearmanr_dcorr_shrink, error_cov_normalized, error_cov_unnormalized, error_basic, error_linear, error_dcorr, error_shrink, error_dcorr_shrink = [], [], [], [], [], [], [], [], [], [], [], [], [], []

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = [], [], [], [], [], []

    for tr, truncate in enumerate(TRUNCATE):
        dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r)

        spearmanr_cov_normalized.append(dic['spearmanr_cov_calibrated'])
        error_cov_normalized.append(dic['error_cov_calibrated'])
        spearmanr_cov_unnormalized.append(dic['spearmanr_cov_uncalibrated'])
        error_cov_unnormalized.append(dic['error_cov_uncalibrated'])

        spearmanr_basic.append(dic['spearmanr_basic'])
        spearmanr_linear.append(dic['spearmanr_linear'])
        spearmanr_dcorr.append(dic['spearmanr_dcorr'])
        spearmanr_shrink.append(dic['spearmanr_shrink'])
        spearmanr_dcorr_shrink.append(dic['spearmanr_dcorr_shrink'])

        error_basic.append(dic['error_basic'])
        error_linear.append(dic['error_linear'])
        error_dcorr.append(dic['error_dcorr'])
        error_shrink.append(dic['error_shrink'])
        error_dcorr_shrink.append(dic['error_dcorr_shrink'])

        std_var.append(dic['std_var'])
        std_cov.append(dic['std_cov'])
        std_var_est.append(dic['std_var_est'])
        std_cov_est.append(dic['std_cov_est'])
        std_var_nonlinear.append(dic['std_var_nonlinear'])
        std_cov_nonlinear.append(dic['std_cov_nonlinear'])

    performances = {
        'spearmanr_cov_calibrated': np.array(spearmanr_cov_normalized),
        'spearmanr_cov_uncalibrated': np.array(spearmanr_cov_unnormalized),
        'spearmanr_basic': np.array(spearmanr_basic),
        'spearmanr_linear': np.array(spearmanr_linear),
        'spearmanr_dcorr': np.array(spearmanr_dcorr),
        'spearmanr_shrink': np.array(spearmanr_shrink),
        'spearmanr_dcorr_shrink': np.array(spearmanr_dcorr_shrink),
        'error_cov_calibrated': np.array(error_cov_normalized),
        'error_cov_uncalibrated': np.array(error_cov_unnormalized),
        'error_basic': np.array(error_basic),
        'error_linear': np.array(error_linear),
        'error_dcorr': np.array(error_dcorr),
        'error_shrink': np.array(error_shrink),
        'error_dcorr_shrink': np.array(error_dcorr_shrink),
        'std_var': np.array(std_var),
        'std_cov': np.array(std_cov),
        'std_var_est': np.array(std_var_est),
        'std_cov_est': np.array(std_cov_est),
        'std_var_nonlinear': np.array(std_var_nonlinear),
        'std_cov_nonlinear': np.array(std_cov_nonlinear),
    }

    return performances

    # return (spearmanr_cov_normalized, spearmanr_cov_unnormalized, spearmanr_basic, spearmanr_linear, spearmanr_dcorr, error_cov_normalized, error_cov_unnormalized, error_basic, error_linear, error_dcorr)


def load_selection(window=20, truncate=700, varying_initial=False, recombination=False, r=1e-5, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED):

    dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r, complete=True)

    selections = {
        'selections_basic': dic['selections_basic'],
        'selections_linear': dic['selections_linear'][:, :, :, :, linear_selected],
        'selections_shrink': dic['selections_shrink'][:, :, :, :, shrink_selected],
        'selections_dcorr': dic['selections_dcorr'][:, :, :, :, loss_selected, gamma_selected],
        'selections_dcorr_shrink': dic['selections_dcorr_shrink'][:, :, :, :, dcorr_shrink_selected]
    }

    return selections


def load_covariances(window=20, truncate=700, varying_initial=False, recombination=False, r=1e-5, p=0, q=0, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED):

    dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r, complete=True)

    covariances_SL = np.copy(dic['int_cov'][:, :, p, q])
    for s in range(len(covariances_SL)):
        for n in range(len(covariances_SL[0])):
            for i in range(len(covariances_SL[0][0])):
                for j in range(i):
                    covariances_SL[s, n, i, j] = 0
                    covariances_SL[s, n, j, i] = 0

    covariances = {
        'covariances_true': dic['int_cov'][:, :, p, q],
        'covariances_est': dic['int_dcov'][:, :, p, q],
        'covariances_est_unnormalized': dic['int_dcov_uncalibrated'][:, :, p, q],
        'covariances_SL': covariances_SL,
        'covariances_linear': dic['int_dcov'][:, :, p, q],  # Linear shrinkage modifies variances, so covariances are the same as estimated ones.
        'covariances_nonlinear': dic['int_dcov_dcorr_reg'][:, :, p, q],
        'covariances_shrink': dic['int_dcov'][:, :, p, q] * SHRINKING_FACTORS[shrink_selected],
        'covariances_nonlinear_shrink': dic['int_dcov_dcorr_reg'][:, :, p, q] * SHRINKING_FACTORS[dcorr_shrink_selected],
    }

    return covariances


def load_performance_combine(window=20, varying_initial=False, recombination=False, r=1e-5):
    """Loads spearmanr & error of inferred selections by combining multiple replicates with a certain window choice."""

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_shrink, spearmanr_dcorr_shrink, error_basic, error_linear, error_dcorr, error_shrink, error_dcorr_shrink = [], [], [], [], [], [], [], [], [], []

    for tr, truncate in enumerate(TRUNCATE):
        dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r)

        spearmanr_basic.append(dic['spearmanr_basic_combine'])
        spearmanr_linear.append(dic['spearmanr_linear_combine'])
        spearmanr_dcorr.append(dic['spearmanr_dcorr_combine'])
        spearmanr_shrink.append(dic['spearmanr_shrink_combine'])
        spearmanr_dcorr_shrink.append(dic['spearmanr_dcorr_shrink_combine'])

        error_basic.append(dic['error_basic_combine'])
        error_linear.append(dic['error_linear_combine'])
        error_dcorr.append(dic['error_dcorr_combine'])
        error_shrink.append(dic['error_shrink_combine'])
        error_dcorr_shrink.append(dic['error_dcorr_shrink_combine'])

    performances = {
        'spearmanr_basic_combine': np.array(spearmanr_basic),
        'spearmanr_linear_combine': np.array(spearmanr_linear),
        'spearmanr_dcorr_combine': np.array(spearmanr_dcorr),
        'spearmanr_shrink_combine': np.array(spearmanr_shrink),
        'spearmanr_dcorr_shrink_combine': np.array(spearmanr_dcorr_shrink),
        'error_basic_combine': np.array(error_basic),
        'error_linear_combine': np.array(error_linear),
        'error_dcorr_combine': np.array(error_dcorr),
        'error_shrink_combine': np.array(error_shrink),
        'error_dcorr_shrink_combine': np.array(error_dcorr_shrink),
    }

    return performances


def load_cov_one_for_each_selection(tr=5, n=0, window=20, sample_selected=0, record_selected=0):
    """Loads intgrated covarianc matrices with a certain data length and a certain window choice."""

    dic = np.load(ESTIMATION_OUTPUT_DIR + f'/estimation_output_truncate={TRUNCATE[tr]}_window={window}.npz')

    if dic['size'] == 'minimal':
        return dic['int_cov'], dic['int_dcov'], dic['int_dcov_uncalibrated']
    elif dic['size'] == 'medium':
        return dic['int_cov'][:, n], dic['int_dcov'][:, n], dic['int_dcov_uncalibrated'][:, n]
    else: # dic['size'] == 'complete'
        return dic['int_cov'][:, n, p, q], dic['int_dcov'][:, n, p, q], dic['int_dcov_uncalibrated'][:, n, p, q]


def load_cov_error_with_various_windows(tr=5, n=0, window_list=WINDOW, sample_selected=0, record_selected=0, metric_index=0):
    """Loads MAE error of estimated covarianc matrices with a certain data length and various windows.

    Args:
        metrics_index: 0, 1, or 2
            0 stores MAE of all entries in covariance matrices (both covariances and variances).
            1 stores MAE of off-diagonal entries in covariance matrices (only covariances).
            2 stores MAE of diagonal entries in covariance matrices (only variances).
    """

    dic_list = [np.load(ESTIMATION_OUTPUT_DIR +
                        f'/estimation_output_truncate={TRUNCATE[tr]}_window={window}.npz')
                for window in window_list]

    return (np.array([dic['error_cov_calibrated'][:, n, sample_selected, record_selected, metric_index] for dic in dic_list]),
            np.array([dic['error_cov_uncalibrated'][:, n, sample_selected, record_selected, metric_index] for dic in dic_list]))


def load_complete_cov_for_a_simulation(s=4, n=0, truncate=700, window=20, linear_selected=LINEAR_SELECTED, varying_initial=False, recombination=False, r=1e-5):
    """Loads different estimates of intgrated covariances.
    Note: require complete output.
    """
    dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r, complete=True)

    if dic['size'] != 'complete':
        print(f'Output does not contain complete covariance information.')
        return

    keys = ['int_cov', 'int_dcov', 'int_dcov_uncalibrated', 'int_dcov_dcorr_reg', 'int_dcov_uncalibrated_dcorr_reg']

    dic_cov = {key: dic[key][s, n] for key in keys}
    L = len(dic_cov['int_cov'][0, 0])
    linear_reg = LINEAR[linear_selected]
    dic_cov['int_dcov_linear'] = dic_cov['int_dcov'] + linear_reg * np.identity(L)
    dic_cov['int_dcov_uncalibrated_linear'] = dic_cov['int_dcov_uncalibrated'] + linear_reg * np.identity(L)
    return dic_cov


def load_performance_window(truncate=700, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, varying_initial=False, recombination=False, r=1e-5):
    """Loads Spearmanr & error of inferred selections with different time window choices."""

    spearmanr_basic_window, spearmanr_linear_window, spearmanr_dcorr_window, spearmanr_shrink_window, spearmanr_dcorr_shrink_window, error_basic_window, error_linear_window, error_dcorr_window, error_shrink_window, error_dcorr_shrink_window = [], [], [], [], [], [], [], [], [], []

    for window in WINDOW:
        dic = SS.load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r)

        spearmanr_basic_window.append(dic['spearmanr_basic'])
        spearmanr_linear_window.append(dic['spearmanr_linear'][:, :, :, :, linear_selected])
        spearmanr_dcorr_window.append(dic['spearmanr_dcorr'][:, :, :, :, loss_selected, gamma_selected])
        spearmanr_shrink_window.append(dic['spearmanr_shrink'][:, :, :, :, shrink_selected])
        spearmanr_dcorr_shrink_window.append(dic['spearmanr_dcorr_shrink'][:, :, :, :, dcorr_shrink_selected])

        error_basic_window.append(dic['error_basic'])
        error_linear_window.append(dic['error_linear'][:, :, :, :, linear_selected])
        error_dcorr_window.append(dic['error_dcorr'][:, :, :, :, loss_selected, gamma_selected])
        error_shrink_window.append(dic['error_shrink'][:, :, :, :, shrink_selected])
        error_dcorr_shrink_window.append(dic['error_dcorr_shrink'][:, :, :, :, dcorr_shrink_selected])

    performances_window = {
        'spearmanr_basic_window': spearmanr_basic_window,
        'spearmanr_linear_window': spearmanr_linear_window,
        'spearmanr_dcorr_window': spearmanr_dcorr_window,
        'spearmanr_shrink_window': spearmanr_shrink_window,
        'spearmanr_dcorr_shrink_window': spearmanr_dcorr_shrink_window,
        'error_basic_window': error_basic_window,
        'error_linear_window': error_linear_window,
        'error_dcorr_window': error_dcorr_window,
        'error_shrink_window': error_shrink_window,
        'error_dcorr_shrink_window': error_dcorr_shrink_window,

    }

    return performances_window


def save_simulation_traj_for_haplosep():
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            sim = SS.load_simulation(s, n)
            traj = sim['traj']
            filename = f'simulation_s={s}_n={n}.npy'
            save_traj_for_haplosep(traj, filename)


def parse_simulation_results_for_haplosep(save=True):
    results_haplosep = [[] for s in range(NUM_SELECTIONS)]
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            output = f'haplosep_output_s={s}_n={n}.npz'
            traj = SS.load_simulation(s, n)['traj']
            res = infer_with_haplosep_output(output, traj)
            results_haplosep[s].append(res)
    if save:
        np.savez_compressed(f"{HAPLOSEP_OUTPUT_DIR}/haplosep_results_for_simulation.npz", results=results_haplosep)
    return results_haplosep


def load_simulation_results_for_haplosep():

    return np.load(f"{HAPLOSEP_OUTPUT_DIR}/haplosep_results_for_simulation.npz", allow_pickle=True)['results']


def check_simulation_results_for_haplosep(results_haplosep=None):
    if results_haplosep is None:
        results_haplosep = load_simulation_results_for_haplosep()
    columns = ['s', 'n', 'MAE_traj', '# geno', 'Spearmanr_s', 'Spearmanr_s_by_geno_traj']
    rows = []
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            res = results_haplosep[s][n]
            selection = SS.load_selection(s)
            a, b = res['traj'], res['traj_from_geno_traj']
            num_genotypes = len(res['geno_traj'][0])
            spearmanr = stats.spearmanr(res['selection'], selection)[0]
            spearmanr_from_geno_traj = stats.spearmanr(res['selection_from_geno_traj'], selection)[0]
            row = [s, n, MAE(a, b), num_genotypes, spearmanr, spearmanr_from_geno_traj]
            rows.append(row)
    print(tabulate(rows, columns, tablefmt='plain'))


def load_simulation_results_for_evoracle():
    results_evoracle = [[] for s in range(NUM_SELECTIONS)]
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            res = SS.load_evoracle(s, n)
            results_evoracle[s].append(res)
    return results_evoracle


def parse_performances_on_covariances_selections_all_methods(selections, covariances, performances_evoracle, performances_haplosep, p=0, q=0, selection_from_geno_traj=False):

    covariances_true, covariances_SL, covariances_est, covariances_est_unnormalized, covariances_linear, covariances_nonlinear, covariances_shrink, covariances_nonlinear_shrink = covariances['covariances_true'], covariances['covariances_SL'], covariances['covariances_est'], covariances['covariances_est_unnormalized'], covariances['covariances_linear'], covariances['covariances_nonlinear'], covariances['covariances_shrink'], covariances['covariances_nonlinear_shrink']

    covariances_evoracle = [[performances_evoracle[s][n]['int_cov'] for n in range(NUM_TRIALS)]
                            for s in range(NUM_SELECTIONS)]
    covariances_haplosep = [[performances_haplosep[s][n]['int_cov'] for n in range(NUM_TRIALS)]
                            for s in range(NUM_SELECTIONS)]

    selections_true = SS.load_selections()
    selections_MPL, selections_SL, selections_est, selections_est_unnormalzied, selections_linear, selections_nonlinear, selections_shrink, selections_nonlinear_shrink = selections['selections_basic'][:, :, p, q, 1], selections['selections_basic'][:, :, p, q, 0], selections['selections_basic'][:, :, p, q, 2], selections['selections_basic'][:, :, p, q, 3], selections['selections_linear'][:, :, p, q], selections['selections_dcorr'][:, :, p, q], selections['selections_shrink'][:, :, p, q], selections['selections_dcorr_shrink'][:, :, p, q]

    if selection_from_geno_traj:
        selections_evoracle = [[performances_evoracle[s][n]['selection_from_geno_traj'] for n in range(NUM_TRIALS)] for s in range(NUM_SELECTIONS)]
        selections_haplosep = [[performances_haplosep[s][n]['selection_from_geno_traj'] for n in range(NUM_TRIALS)] for s in range(NUM_SELECTIONS)]
    else:
        selections_evoracle = [[performances_evoracle[s][n]['selection'] for n in range(NUM_TRIALS)]
                                for s in range(NUM_SELECTIONS)]
        selections_haplosep = [[performances_haplosep[s][n]['selection'] for n in range(NUM_TRIALS)]
                                for s in range(NUM_SELECTIONS)]

    covariances_list = [
        covariances_true, covariances_SL, covariances_est, covariances_est_unnormalized,
        covariances_linear, covariances_nonlinear,
        # covariances_shrink, covariances_nonlinear_shrink,
        np.array(covariances_evoracle), np.array(covariances_haplosep)
        ]

    selections_list = [
        selections_MPL, selections_SL, selections_est, selections_est_unnormalzied,
        selections_linear, selections_nonlinear,
        # selections_shrink, selections_nonlinear_shrink,
        np.array(selections_evoracle), np.array(selections_haplosep)
        ]

    MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection = [], [], [], []
    method_list = [
        'MPL', 'SL', 'Est', 'Est (Unnormalized)',
        'Linear', 'Nonlinear',
        # 'Est-shrink', 'Nonlinear-shrink',
        'Evoracle', 'haploSep']

    # print(covariances_true.shape, selections_true.shape)

    for i, (cov, sel) in enumerate(zip(covariances_list, selections_list)):
        # print(method_list[i], cov.shape, sel.shape)
        MAE_cov.append([])
        spearmanr_cov.append([])
        MAE_selection.append([])
        spearmanr_selection.append([])
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                spearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(EC.get_off_diagonal_terms(covariances_true[s, n]), EC.get_off_diagonal_terms(cov[s][n]))
                MAE_cov[i].append(MAE)
                spearmanr_cov[i].append(spearmanr)
                spearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(selections_true[s], sel[s][n])
                MAE_selection[i].append(MAE)
                spearmanr_selection[i].append(spearmanr)

    return MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection


############################################
#
# Two datasets in Shen2021: Cry1Ac, TadA
#
############################################

def parse_obs_reads_df_cry1ac(allele_traj_file='cry1ac_1nt_obsreads.csv'):
    df = pd.read_csv(f'{SHEN_DATA_DIR}/{allele_traj_file}')
    return df


def parse_data_cry1ac(genotype_traj_file='cry1ac_pacbio_pivot_filt1pct.csv', allele_traj_file='cry1ac_1nt_obsreads.csv'):
    geno_data = pd.read_csv(f'{SHEN_DATA_DIR}/{genotype_traj_file}')
    geno_traj = geno_data.to_numpy()[:, 1:].T

    genotypes = list(geno_data['Abbrev genotype'])
    binary_genotypes = [binarize_genotype(genotype) for genotype in genotypes]

    # Only for verification purpose
    # allele_traj_computed_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, binary_genotypes)

    hour_strings = list(geno_data.columns)[1:1+Cry1Ac_PARAMS['num_timepoints']]
    hours = np.array([_[:-3] for _ in hour_strings]).astype('int')
    times = [int(hour * Cry1Ac_PARAMS['generation_per_hour']) for hour in hours]

    traj_data = pd.read_csv(f'{SHEN_DATA_DIR}/{allele_traj_file}')
    traj = traj_data.to_numpy()[1::2, :-1].T.astype(float)
    T, L = traj.shape

    int_cov = MPL.integrateCovarianceFromStableGenotypes(binary_genotypes, geno_traj, times)

    # int_cov_test = np.zeros((L, L))
    # MPL.processStandard(np.array([binary_genotypes] * T), geno_traj, times, 2, int_cov_test)
    # print(np.allclose(int_cov, int_cov_test))

    data = {
        'geno_traj': geno_traj,
        'traj': traj,
        'times': times,
        # 'allele_traj_computed_from_geno_traj': allele_traj_computed_from_geno_traj,
        'genotypes': genotypes,
        'binary_genotypes': binary_genotypes,
        'int_cov': int_cov,
    }

    return data


def save_traj_for_haplosep_cry1ac():
    data = parse_data_cry1ac()
    filename = f'Cry1Ac'
    save_traj_for_haplosep(data['traj'], filename)


def run_evoracle_cry1ac(directory=f'{SHEN_DATA_DIR}/cry1ac_output', obsreads_filename='cry1ac_1nt_obsreads.csv'):
    obsreads_input = f'{SHEN_DATA_DIR}/{obsreads_filename}'

    # Run Evoracle
    proposed_genotypes_output = f'{directory}/proposed_genotypes.csv'
    obs_reads_df = parse_obs_reads_df(obsreads_input)
    evoracle.propose_gts_and_infer_fitness_and_frequencies(obs_reads_df, proposed_genotypes_output, directory)


def parse_evoracle_results_cry1ac(directory=f'{SHEN_DATA_DIR}/cry1ac_output', obsreads_filename='cry1ac_1nt_obsreads.csv'):

    data = parse_data_cry1ac()
    results = parse_evoracle_results(obsreads_filename, directory, params=Cry1Ac_PARAMS, save_geno_traj=True, compute_genotype_fitness=True)

    results['fitness'] = MPL.computeGenotypeFitnessFromSelection(data['binary_genotypes'], results['selection'])
    results['fitness_from_geno_traj'] = MPL.computeGenotypeFitnessFromSelection(data['binary_genotypes'], results['selection_from_geno_traj'])

    return results



def parse_fitness_inferred_by_evoracle_cry1ac():

    fitness = pd.read_csv(f'{Cry1Ac_OUTPUT_DIR}/_final_fitness.csv')
    genotype_to_fitness = {}
    for i, row in fitness.iterrows():
        genotype_to_fitness[row['Genotype']] = row['Inferred fitness']
    return genotype_to_fitness


def parse_data_tada(genotype_traj_file_prefix='tada_filt0.0pct_threshold5', pace=2):
    geno_data = pd.read_csv(f'{SHEN_DATA_DIR}/{genotype_traj_file_prefix}_pace{pace}.csv')
    geno_traj = geno_data.to_numpy()[:, 1:].T
    T = len(geno_traj)
    mut_to_index = {}
    index = 0
    for i, row in geno_data.iterrows():
        for mut in row[0].split(','):
            if mut[-1] != '.' and mut not in mut_to_index:
                mut_to_index[mut] = index
                index += 1
    # print(mut_to_index)

    traj_all = np.zeros((len(mut_to_index), T), dtype=float)
    for i, row in geno_data.iterrows():
        for mut in row[0].split(','):
            if mut in mut_to_index:
                index = mut_to_index[mut]
                for t in range(len(traj_all[index])):
                    traj_all[index, t] += geno_traj[t, i]
    traj_all = traj_all.T

    locus_to_muts = {mut.split(' ')[0]: [] for mut in mut_to_index.keys()}
    for mut in mut_to_index.keys():
        locus = mut.split(' ')[0]
        # allele = mut.split(' ')[1]
        if mut not in locus_to_muts[locus]:
            locus_to_muts[locus].append(mut)
    # print(locus_to_muts)

    # for locus, muts in locus_to_muts.items():
    #     if len(muts) > 1:
    #         print(locus, muts)
    #         indices = [mut_to_index[mut] for mut in muts]
    #         plotTraj(traj[:, indices])

    traj = np.zeros((len(locus_to_muts), T), dtype=float)
    loci = sorted(list(locus_to_muts.keys()))
    for i, locus in enumerate(loci):
        muts = locus_to_muts[locus]
        indices = [mut_to_index[mut] for mut in muts]
        traj[i, :] = np.sum(traj_all[:, indices], axis=1)
    traj = traj.T

    times = np.arange(0, T)

    genotype_strings = list(geno_data['Full genotype'])
    genotypes = []
    for genotype_string in genotype_strings:
        genotype = ''.join([_[-1] for _ in genotype_string.split(',')])
        genotypes.append(genotype)
    L = len(genotypes[0])

    binary_genotypes = [binarize_genotype(genotype) for genotype in genotypes]
    # print(len(genotypes[0]))
    int_cov = MPL.integrateCovarianceFromStableGenotypes(binary_genotypes, geno_traj, times)

    # int_cov_test = np.zeros((L, L))
    # MPL.processStandard(np.array([binary_genotypes] * T), geno_traj, times, 2, int_cov_test)
    # print(np.allclose(int_cov, int_cov_test))

    traj = compute_allele_traj_from_geno_traj(geno_traj, binary_genotypes)

    data = {
        'geno_traj': geno_traj,
        'traj': traj,
        'times': times,
        'genotypes': genotypes,
        'binary_genotypes': binary_genotypes,
        'int_cov': int_cov,
    }

    return data


def infer_genotype_fitness(data, params, calibrate_at_the_end=False, shrink_off_diagonal_terms=1, nonlinear_reg=None, correct_locus=[], verbose=True):

    window, mu, linear_reg = params['window'], params['mutation_rate'], params['linear_reg']
    if nonlinear_reg is None:
        nonlinear_reg = params['nonlinear_reg']
    traj, times, int_cov = data['traj'], data['times'], data['int_cov']
    genotypes = data['binary_genotypes']
    T, L = traj.shape
    D = MPL.computeD(traj, times, mu)

    variances = [int_cov[i, i] for i in range(L)]
    est_cov, selection_by_est_cov = EC.estimateAndInferencePipeline(traj, times, window, mu, nonlinear_reg=nonlinear_reg, linear_reg=linear_reg)
    if calibrate_at_the_end:
        EC.calibrate_est_cov(est_cov, variances)

    if shrink_off_diagonal_terms != 1:
        if verbose:
            print('Before shrinking, off-diagonal terms of est_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))
            print('off-diagonal terms of true_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(int_cov)), np.std(EC.get_diagonal_terms(int_cov))))

        est_cov = EC.shrinkOffDiagonalTerms(est_cov, shrink_off_diagonal_terms)

        if verbose:
            print('After shrinking, off-diagonal terms of est_cov  has std=%.3f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

    if correct_locus:
        for l in correct_locus:
            for i in range(L):
                est_cov[i, l] = int_cov[i, l]
                est_cov[l, i] = int_cov[l, i]

    selection_by_est_cov = MPL.inferSelection(est_cov, D, linear_reg * np.identity(L))
    fitness_by_est_cov = MPL.computeGenotypeFitnessFromSelection(genotypes, selection_by_est_cov)

    # print(L, int_cov.shape)
    selection_by_true_cov = MPL.inferSelection(int_cov, D, linear_reg * np.identity(L))
    fitness_by_true_cov = MPL.computeGenotypeFitnessFromSelection(genotypes, selection_by_true_cov)

    cov_SL_test = MPL.computeIntegratedVariance(traj, times)
    cov_SL = copy_diagonal_terms_from(int_cov)
    selection_by_SL = MPL.inferSelection(cov_SL, D, linear_reg * np.identity(L))
    fitness_by_SL = MPL.computeGenotypeFitnessFromSelection(genotypes, selection_by_SL)

    inference_results = {
        'cov_SL': cov_SL,
        'est_cov': est_cov,
        'selection_by_est_cov': selection_by_est_cov,
        'fitness_by_est_cov': fitness_by_est_cov,
        'selection_by_true_cov': selection_by_true_cov,
        'fitness_by_true_cov': fitness_by_true_cov,
        'selection_by_SL': selection_by_SL,
        'fitness_by_SL': fitness_by_SL,
    }

    return inference_results


def find_optimal_shrink_factor(data, params, nonlinear_reg=None):
    shrinking_factors = np.arange(0, 1, 0.01)
    spearmanr_selection, pearsonr_selection, MAE_selection = [], [], []
    spearmanr_fitness, pearsonr_fitness, MAE_fitness = [], [], []
    for factor in shrinking_factors:
        res = infer_genotype_fitness(data, params, shrink_off_diagonal_terms=factor, nonlinear_reg=nonlinear_reg, verbose=False)
        spearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(res['selection_by_est_cov'], res['selection_by_true_cov'])
        spearmanr_selection.append(spearmanr)
        pearsonr_selection.append(pearsonr)
        MAE_selection.append(MAE)

        pearmanr, pearsonr, MAE = spearmanr_pearsonr_MAE(res['fitness_by_est_cov'], res['fitness_by_true_cov'])
        spearmanr_fitness.append(spearmanr)
        pearsonr_fitness.append(pearsonr)
        MAE_fitness.append(MAE)

    results = {
        'shrinking_factors': shrinking_factors,
        'spearmanr_selection': spearmanr_selection,
        'pearsonr_selection': pearsonr_selection,
        'MAE_selection': MAE_selection,
        'spearmanr_fitness': spearmanr_fitness,
        'pearsonr_fitness': pearsonr_fitness,
        'MAE_fitness': MAE_fitness,
    }

    return results


def plot_performance_vs_shrinking_factor(results, suptitle=None):

    shrinking_factors, spearmanr_selection, pearsonr_selection, MAE_selection, spearmanr_fitness, pearsonr_fitness, MAE_fitness = results['shrinking_factors'], results['spearmanr_selection'], results['pearsonr_selection'], results['MAE_selection'], results['spearmanr_fitness'], results['pearsonr_fitness'], results['MAE_fitness']

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))
    plt.sca(axes[0, 0])
    plt.plot(shrinking_factors, spearmanr_selection, marker='.', label='spearmanr')
    plt.plot(shrinking_factors, pearsonr_selection, marker='.', label='pearsonr')
    plt.legend()
    plt.xticks(ticks=[], labels=[])
    plt.title('Perf on selection')

    plt.sca(axes[1, 0])
    plt.plot(shrinking_factors, MAE_selection, marker='.', label='MAE', color='green')
    plt.xlabel('Shrinking factor')
    plt.legend()

    plt.sca(axes[0, 1])
    plt.plot(shrinking_factors, spearmanr_fitness, marker='.', label='spearmanr')
    plt.plot(shrinking_factors, pearsonr_fitness, marker='.', label='pearsonr')
    plt.legend()
    plt.xticks(ticks=[], labels=[])
    plt.title('Perf on fitness')

    plt.sca(axes[1, 1])
    plt.plot(shrinking_factors, MAE_fitness, marker='.', label='MAE', color='green')
    plt.xlabel('Shrinking factor')
    plt.legend()
    # plt.title('Optimal factor = %.2f for selection' % shrinking_factors[np.argmax(spearmanr_selection)] +
    #           'Optimal factor = %.2f for fitness' % shrinking_factors[np.argmax(spearmanr_fitness)])

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.show()

############################################
#
# PALTE dataset
#
############################################

def init_dic_PALTE():
    return {pop: None for pop in PALTE_POPULATIONS}


def parse_traj_PALTE(df):
    times = np.array(sorted([int(col[1:]) for col in df.columns if col[0] == 'X' and col[1:].isdigit()]))
    T, L = len(times), len(df)
    traj = np.zeros((T, L), dtype=float)
    l = 0
    for i, row in df.iterrows():
        for t, time in enumerate(times):
            value = row['X' + str(time)]
            if np.isnan(value):
                traj[t, l] = -1
            else:
                traj[t, l] = float(value)
        l += 1
    return traj, times


def save_traj_for_haplosep_PALTE():
    data = parse_data_PALTE()
    for pop in PALTE_POPULATIONS:
        filename = f'PALTE_{pop}'
        save_traj_for_haplosep(data[pop]['traj'], filename)


def parse_interpolated_traj_PALTE(pop, df):
    traj, times = parse_traj_PALTE(df[df['Population'] == pop])
    return interpolate_traj(traj, times)


def parse_data_PALTE():
    df_traj = pd.read_csv(PALTE_TRAJ_FILE)
    df_fitness = pd.read_excel(PALTE_FITNESS_FILE, engine='openpyxl', sheet_name=PALTE_SHEETNAMES)
    data = {pop: {} for pop in PALTE_POPULATIONS}
    for pop in PALTE_POPULATIONS:
        traj = parse_interpolated_traj_PALTE(pop, df_traj)
        data[pop] = {
            'traj': traj,
            'times': PALTE_TIMES,
        }
    return data


def parse_inferred_fitness_list_PALTE(inference_results, results_evoracle, results_haplosep, methods=PALTE_PARAMS['methods'], use_fitness_from_geno_traj_for_evoracle=False, use_fitness_from_geno_traj_for_haplosep=False):

    inferred_fitness_list = []
    keys = ['fitness_by_est_cov', 'fitness_by_SL', 'fitness_by_lolipop']
    for key in keys:
        inferred_fitness_list.append([1] + [inference_results[key][pop] for pop in PALTE_POPULATIONS])

    key = 'fitness_from_geno_traj' if use_fitness_from_geno_traj_for_evoracle else 'fitness'
    inferred_fitness_list.append([1] + [results_evoracle[pop][key] for pop in PALTE_POPULATIONS])

    key = 'fitness_from_geno_traj' if use_fitness_from_geno_traj_for_haplosep else 'fitness'
    inferred_fitness_list.append([1] + [results_haplosep[pop][key] for pop in PALTE_POPULATIONS])

    return inferred_fitness_list


def infer_population_fitness_PALTE(data, params=PALTE_PARAMS):

    populations = PALTE_POPULATIONS
    window, mu, nonlinear_reg, linear_reg = params['window'], params['mutation_rate'], params['nonlinear_reg'], params['linear_reg']

    est_cov, selection_by_est_cov, fitness_by_est_cov, cov_SL, selection_by_SL, fitness_by_SL = init_dic_PALTE(), init_dic_PALTE(), init_dic_PALTE(), init_dic_PALTE(), init_dic_PALTE(), init_dic_PALTE()

    for pop in populations:
        traj, times = data[pop]['traj'], data[pop]['times']
        T, L = traj.shape
        D = MPL.computeD(traj, times, mu)
        est_cov[pop], _ = EC.estimateAndInferencePipeline(traj, times, window, mu, interpolate=False, nonlinear_reg=nonlinear_reg, linear_reg=linear_reg)
        selection_by_est_cov[pop] = MPL.inferSelection(est_cov[pop], D, np.identity(L))
        fitness_by_est_cov[pop] = MPL.computePopulationFitnessFromSelection(traj, selection_by_est_cov[pop])

        cov_SL[pop] = MPL.computeIntegratedVariance(traj, times)
        selection_by_SL[pop] = MPL.inferSelection(cov_SL[pop], D, linear_reg * np.identity(L))
        fitness_by_SL[pop] = MPL.computePopulationFitnessFromSelection(traj, selection_by_SL[pop])

    lolipop_cov, selection_by_lolipop, fitness_by_lolipop = parse_Lolipop_results_PALTE()

    inference_results = {
        'est_cov': est_cov,
        'selection_by_est_cov': selection_by_est_cov,
        'fitness_by_est_cov': fitness_by_est_cov,
        'selection_by_SL': selection_by_SL,
        'fitness_by_SL': fitness_by_SL,
        'lolipop_cov': lolipop_cov,
        'selection_by_lolipop': selection_by_lolipop,
        'fitness_by_lolipop': {pop: fitness_by_lolipop[pop][-1] for pop in populations},
    }

    return inference_results


def parse_Lolipop_results_PALTE():
    mu = PALTE_PARAMS['mutation_rate']
    regularization = PALTE_PARAMS['linear_reg']
    populations = PALTE_POPULATIONS

    intCov_lolipop, selection_lolipop, fitness_lolipop = {pop: None for pop in populations}, {pop: None for pop in populations}, {pop: None for pop in populations}
    for pop in populations:
        file = f'{LOLIPOP_PARSED_OUTPUT_DIR}/{pop}.npz'
        data = load_parsed_output(file)
        intCov, selection, fitness = inference_pipeline(data, mu=mu, regularization=regularization)

        intCov_lolipop[pop] = intCov
        selection_lolipop[pop] = selection
        fitness_lolipop[pop] = fitness

    return intCov_lolipop, selection_lolipop, fitness_lolipop


def save_traj_for_evoracle_PALTE(use_generation_as_time=False):
    data_PALTE = parse_data_PALTE()
    for pop in PALTE_POPULATIONS:
        traj, times = data_PALTE[pop]['traj'], data_PALTE[pop]['times']
        if not use_generation_as_time:
            times = np.arange(0, len(times))
        filename = f'{EVORACLE_DIR}/{pop}_obsreads.csv'
        save_traj_for_evoracle(traj, filename, times=times)


def parse_obs_reads_df_PALTE(pop):
    df = pd.read_csv(f'{EVORACLE_DIR_PALTE}/{pop}_obsreads.csv')
    return df


def parse_evoracle_results_PALTE():
    mu = PALTE_PARAMS['mutation_rate']
    reg = PALTE_PARAMS['linear_reg']
    populations = PALTE_POPULATIONS
    times = PALTE_TIMES
    results = {pop: {} for pop in populations}
    for pop in populations:
        df_traj = pd.read_csv(f'{EVORACLE_DIR_PALTE}/{pop}_obsreads.csv')
        traj = parse_evoracle_traj(df_traj)
        T, L = traj.shape

        df_geno = pd.read_csv(f'{EVORACLE_DIR_PALTE}/{pop}/_final_genotype_matrix.csv')
        binary_genotypes, geno_traj = parse_evoracle_geno_traj(df_geno)

        traj_from_geno_traj = compute_allele_traj_from_geno_traj(geno_traj, binary_genotypes)

        int_cov = MPL.integrateCovarianceFromStableGenotypes(binary_genotypes, geno_traj, times)
        D = MPL.computeD(traj, times, mu)
        D_from_geno_traj = MPL.computeD(traj_from_geno_traj, times, mu)
        selection = MPL.inferSelection(int_cov, D, reg * np.identity(L))
        selection_from_geno_traj = MPL.inferSelection(int_cov, D_from_geno_traj, reg * np.identity(L))
        fitness = MPL.computePopulationFitnessFromSelection(traj, selection)
        fitness_from_geno_traj = MPL.computePopulationFitnessFromSelection(traj_from_geno_traj, selection_from_geno_traj)

        results[pop] = {
            'traj': traj,
            'genotypes': binary_genotypes,
            'geno_traj': geno_traj,
            'traj_from_geno_traj': traj_from_geno_traj,
            'int_cov': int_cov,
            'selection': selection,
            'selection_from_geno_traj': selection_from_geno_traj,
            'fitness':fitness,
            'fitness_from_geno_traj': fitness_from_geno_traj,
        }
    return results


def parse_haplosep_results_PALTE():
    mu = PALTE_PARAMS['mutation_rate']
    reg = PALTE_PARAMS['linear_reg']
    populations = PALTE_POPULATIONS
    times = PALTE_TIMES
    data = parse_data_PALTE()
    results = {pop: {} for pop in populations}
    for pop in populations:
        traj, times = data[pop]['traj'], data[pop]['times']
        filename = f'haplosep_output_PALTE_{pop}'
        results[pop] = infer_with_haplosep_output(filename, traj, times=times, reg=reg, mu=mu, compute_population_fitness=True)
    return results


def parse_all_measured_fitness_PALTE(numRows=6, th_large_fitness=9):
    fitness_all = {
        sheet: {f'{pop} {cond}': [] for pop in PALTE_MEASURED_POPULATIONS for cond in PALTE_CONDITIONS}
            for sheet in PALTE_SHEETNAMES}
    df_fitness = pd.read_excel(PALTE_FITNESS_FILE, engine='openpyxl', sheet_name=PALTE_SHEETNAMES)
    for sheet in PALTE_SHEETNAMES:
        df = df_fitness[sheet]
        for pop in PALTE_MEASURED_POPULATIONS:
            for cond in PALTE_CONDITIONS:
                col = pop + ' ' + cond
                if col not in set(df.columns):
                    col = pop + ' ' + cond.lower()
                if col not in set(df.columns):
                    col += ' '
                fitness_list = list(df[col])
                for i in range(numRows):
                    if fitness_list[i] <= th_large_fitness:
                        fitness_all[sheet][f'{pop} {cond}'].append(fitness_list[i])
    return fitness_all


def parse_measured_fitness_PALTE():
    fitness_all = parse_all_measured_fitness_PALTE()
    pop_to_measured_fitness = {pop: None for pop in PALTE_MEASURED_POPULATIONS}
    for pop in PALTE_MEASURED_POPULATIONS:
        pop_to_measured_fitness[pop] = np.mean([np.median(fitness_all[PALTE_SELECTED_SHEET][f'{pop} {cond}']) for cond in PALTE_SELECTED_CONDITIONS])
    measured_fitness = [pop_to_measured_fitness[pop] for pop in PALTE_MEASURED_POPULATIONS]
    return measured_fitness


############################################
#
# Two datasets in McDonald2016
#
############################################


def fig_to_fig_column(fig):
    if fig == '3a':
        return '2b,3a'
    if fig == '3b':
        return '2f,3b'
    return fig


def parse_df_mcdonald():
    df = pd.read_csv(MCDONALD_FILE, sep='\t', header=1)
    return df


def parse_data_mcdonald(fig='3a'):
    df = parse_df_mcdonald()
    traj = parse_traj_mcdonald(fig=fig)
    selections = parse_selection_mcdonald(fig=fig)
    selected_indices = parse_plotted_selection_indices_mcdonald(fig=fig)
    data = {
        'traj': traj,
        'times': MCDONALD_TIMES,
        'selections': selections,
        'selected_indices': selected_indices,
    }
    return data


def parse_traj_mcdonald(fig='3a'):
    df = parse_df_mcdonald()
    df = df[df['Figures']==fig_to_fig_column(fig)][MCDONALD_TIME_COLUMNS]
    df.fillna(0, inplace=True)
    return df.to_numpy().astype(float).T


def parse_selection_mcdonald(fig='3a', sequencing_based_assay_column='BS_sel_coef', reconstruction_column='Rec_sel_coef', selection_measured_from_reconstruction=['grs2', 'met2', 'rgt1', 'hul5', 'toh1'], gene_column='Gene'):
    selection_measured_from_reconstruction = {
        '3a': ['grs2', 'met2', 'rgt1'],
        '3b': ['hul5', 'grs2', 'rgt1', 'toh1', 'met2'],
    }
    df = parse_df_mcdonald()
    columns = [gene_column, sequencing_based_assay_column, reconstruction_column]
    df = df[df['Figures']==fig_to_fig_column(fig)][columns]
    selections = []
    for i, row in df.iterrows():
        if str(row[gene_column]).lower() in selection_measured_from_reconstruction[fig]:
            selections.append(row[reconstruction_column])
        else:
            selections.append(row[sequencing_based_assay_column])
    return np.array(selections)


def parse_plotted_selection_indices_mcdonald(fig='3a'):
    plotted_selection_genes = {
        '3a': ['ade3', 'ahp1', 'grs2', 'acs1', 'sda1', 'yak1', 'yur1', 'zap1', 'sun4', 'toh1', 'aro2', 'met2', 'rgt1', 'cys4', 888443, 41715],
        # Pasted from Figure 3, McDonald, M. J., Rice, D. P. & Desai, M. M. Sex speeds adaptation by altering the dynamics of molecular evolution. Nature 531, 233–236 (2016)
        # 16. Intergenic (IX)15. Intergenic (VII)14. cys4 (VII)13. rgt1* (XI)12. met2* (XIV)11. aro2 (VII)10. toh1 (X)9. sun4 (XIV)8. zap1 (X)7. yur1 (X)6. yak1 (X)5. sda1 (VII)4. acs1 (I)3. grs2* (XVI)2. ahp1 (XII)1. ade3 (VII)
        '3b': ['hul5', 'grs2', 'kre5', 'rgt1', 'toh1', 87869, 'met2', 'snf2'],
        # Pasted from Figure 3, McDonald, M. J., Rice, D. P. & Desai, M. M. Sex speeds adaptation by altering the dynamics of molecular evolution. Nature 531, 233–236 (2016)
        # 24. snf2 (XV)23. met2* (XIV)22. Intergenic (IX)21. toh1* (X)20. rgt1* (XI)19. kre5 (XV)18. grs2* (XVI) 17. hul5* (VII)
    }
    df = parse_df_mcdonald()
    columns = ['Gene', 'Effect', 'Position']
    df = df[df['Figures']==fig_to_fig_column(fig)][columns]
    indices = []
    print('Selected_indices', len(df))
    initial_i = None
    for i, row in df.iterrows():
        if initial_i is None:
            initial_i = i
        gene = str(row['Gene']).lower()
        if gene in plotted_selection_genes[fig]:
            indices.append(i)
        elif row['Effect'] == 'intergenic' and row['Position'] in plotted_selection_genes[fig]:
            indices.append(i)

    return np.array(indices) - initial_i


def infer_selection_mcdonald(data, params=MCDONALD_PARAMS, calibrate_at_the_end=False, shrink_off_diagonal_terms=1, nonlinear_reg=None, verbose=True):

    window, mu, linear_reg = params['window'], params['mutation_rate'], params['linear_reg']
    traj, times, selections, selected_indices = data['traj'], data['times'], data['selections'], data['selected_indices']
    if nonlinear_reg is None:
        nonlinear_reg = params['nonlinear_reg']
    T, L = traj.shape
    D = MPL.computeD(traj, times, mu)

    est_cov, selection_by_est_cov = EC.estimateAndInferencePipeline(traj, times, window, mu, nonlinear_reg=nonlinear_reg, linear_reg=linear_reg)

    if shrink_off_diagonal_terms != 1:
        if verbose:
            print('Before shrinking, off-diagonal terms of est_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

        est_cov = EC.shrinkOffDiagonalTerms(est_cov, shrink_off_diagonal_terms)

        if verbose:
            print('After shrinking, off-diagonal terms of est_cov  has std=%.3f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

    selection_by_est_cov = MPL.inferSelection(est_cov, D, linear_reg * np.identity(L))

    cov_SL = MPL.computeIntegratedVariance(traj, times)
    selection_by_SL = MPL.inferSelection(cov_SL, D, linear_reg * np.identity(L))

    inference_results = {
        'cov_SL': cov_SL,
        'est_cov': est_cov,
        'selection_by_est_cov': selection_by_est_cov,
        'selection_by_SL': selection_by_SL,
        'selection': selections,
        'selected_indices': selected_indices,
    }

    return inference_results


def plot_selection_comparison_mcdonald(results, plot_title=True, ylim=(-0.04, 0.09)):

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    selection, selection_est, selection_SL, selected_indices = results['selection'], results['selection_by_est_cov'], results['selection_by_SL'], results['selected_indices']

    indices_nan = np.arange(0, len(selection))[np.isnan(selection)]
    indices = np.array([_ for _ in range(len(selection)) if _ not in indices_nan]).astype(int)

    plt.sca(axes[0, 0])
    plot_comparison(selection[indices], selection_SL[indices], plot_title=plot_title, ylim=ylim)
    plt.ylabel('SL')
    plt.sca(axes[0, 1])
    plot_comparison(selection[selected_indices], selection_SL[selected_indices], plot_title=plot_title, ylim=ylim)
    plt.sca(axes[1, 0])
    plot_comparison(selection[indices], selection_est[indices], plot_title=plot_title, ylim=ylim)
    plt.ylabel('Est')
    plt.xlabel('All selections')
    plt.sca(axes[1, 1])
    plot_comparison(selection[selected_indices], selection_est[selected_indices], plot_title=plot_title, ylim=ylim)
    plt.xlabel('Selected selections')

    plt.show()


############################################
#
# Six replicate datasets in Borderia2015
#
############################################

def parse_df_borderia():
    df = pd.read_excel(BORDERIA_FILE, engine='openpyxl', sheet_name=BORDERIA_SHEETNAMES)
    return df


def parse_data_borderia():
    data_all = {}
    df_all = parse_df_borderia()
    for key, df in df_all.items():
        data = parse_traj_from_df_borderia(df)
        data_all[key] = data
    return data_all


def parse_traj_from_df_borderia(df, filter=0.01):

    print(f'{len(df)} rows(mutations)')

    df.fillna(0, inplace=True)

    time_columns = list(df.columns)[6:]
    print(time_columns)
    times = [int(_[1:]) for _ in time_columns]
    T = len(times)
    L = len(df)

    # print(df[time_columns].dtypes)
    # df = df.astype({time_col: float for time_col in time_columns})
    # print(df[time_columns].dtypes)
    #
    # traj_T = df[time_columns].to_numpy().astype(float)
    # traj = traj_T.T
    # print(f'traj.shape={traj.shape}')
    # T, L = traj.shape

    traj = np.zeros((T, L), dtype=float)
    AA_positions = np.zeros(L, dtype=int)
    AA_muts = np.zeros(L, dtype=str)
    NT_muts = np.zeros(L, dtype=str)
    # AA_muts = []
    # NT_muts = []

    for i, row in df.iterrows():
        for t, time_col in enumerate(time_columns):
            if row[time_col] == '' or row[time_col] == ' ':
                traj[t, i] = 0
            else:
                try:
                    traj[t, i] = float(row[time_col])
                except:
                    # print(i, row)
                    print(i, t, type(row[time_col]), row[time_col], len(row[time_col]), row[time_col]=='')
                    return

        AA_position = row['AA_position']
        AA_change = row['AA_change']
        AA_before, AA_after = AA_change.split('>')[0], AA_change.split('>')[1]
        if AA_before == 'synonymous':
            AA_before = 's'
        AA_mut = f'{AA_before}{AA_position}{AA_after}'
        AA_positions[i] = int(AA_position)
        AA_muts[i] = AA_mut
        # AA_muts.append(AA_mut)

        NT_position = row['Position']
        NT_before, NT_after = row['Reference'], row['Aternate']
        NT_mut = f'{NT_before}{NT_position}{NT_after}'
        NT_muts[i] = NT_mut
        # NT_muts.append(NT_mut)

    indices_filtered = np.array([i for i in range(L) if np.max(traj[:, i]) >= filter])
    traj_filtered = traj[:, indices_filtered]
    print(f'traj_filtered.shape={traj_filtered.shape}')

    unique_positions, counts = np.unique(AA_positions[indices_filtered], return_counts=True)
    dup = unique_positions[counts > 1]
    print('Are filtered mutations all have different AA_position?', len(np.unique(AA_positions[indices_filtered])) == len(indices_filtered))
    if len(dup) > 0:
        print('Duplicated AA_positions: ', dup)

    data = {
        'traj': traj,
        'times': times,
        'AA_muts': AA_muts,
        'NT_muts': NT_muts,
        'indices_filtered': indices_filtered,
        'traj_filtered': traj_filtered,
    }

    return data


def infer_selection_borderia(data_all, params=BORDERIA_PARAMS, shrink_off_diagonal_terms=1, nonlinear_reg=None, verbose=True):

    results_all = {}
    for key, data in data_all.items():
        results = infer_selection_borderia_helper(data, params=params, shrink_off_diagonal_terms=shrink_off_diagonal_terms, nonlinear_reg=nonlinear_reg, verbose=verbose)
        results_all[key] = data

    return results_all


def infer_selection_borderia_helper(data, params=BORDERIA_PARAMS, shrink_off_diagonal_terms=1, nonlinear_reg=None, verbose=True):

    window, mu, linear_reg = params['window'], params['mutation_rate'], params['linear_reg']
    traj, times, selections, selected_indices = data['traj'], data['times'], data['selections'], data['selected_indices']
    if nonlinear_reg is None:
        nonlinear_reg = params['nonlinear_reg']
    T, L = traj.shape
    D = MPL.computeD(traj, times, mu)

    est_cov, selection_by_est_cov = EC.estimateAndInferencePipeline(traj, times, window, mu, nonlinear_reg=nonlinear_reg, linear_reg=linear_reg)

    if shrink_off_diagonal_terms != 1:
        if verbose:
            print('Before shrinking, off-diagonal terms of est_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

        est_cov = EC.shrinkOffDiagonalTerms(est_cov, shrink_off_diagonal_terms)

        if verbose:
            print('After shrinking, off-diagonal terms of est_cov  has std=%.3f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

    selection_by_est_cov = MPL.inferSelection(est_cov, D, linear_reg * np.identity(L))

    cov_SL = MPL.computeIntegratedVariance(traj, times)
    selection_by_SL = MPL.inferSelection(cov_SL, D, linear_reg * np.identity(L))

    inference_results = {
        'cov_SL': cov_SL,
        'est_cov': est_cov,
        'selection_by_est_cov': selection_by_est_cov,
        'selection_by_SL': selection_by_SL,
        'selection': selections,
        'selected_indices': selected_indices,
    }

    return inference_results


def parse_fitness_borderia():
    gene_to_fitness = {
        'E76G': 1,
        'D138G': -0.8,
        'Q234K': -1.9,
        'N63Y': -0.2,
        'N63H': -0.3,
    }

    return gene_to_fitness


############################################
#
# Three longitudinal patient data from Kessinger2013
#
############################################

def extract_time_kessinger2013(tag):

    if 'con' in tag or 'transmitted' in tag:
        return None
    if len(tag.split('_')) == 2:
        tag_ = tag.split('_')[0]
    else:
        tag_ = tag.split('_')[1]
    if 'd' not in tag_:
        return 0
    index = tag_.index('d')
    return int(tag_[index+1:])


def read_alignment_kessinger2013(ch=40):
    ''' Read in a multiple sequence alignment in fasta format, and return the
    tags and seqs.
    '''
    filename = f'{KESSINGER_DATA_DIR}/CH{ch}longitudinal.fasta'
    filelines = open(filename, 'r').readlines()
    tags = list()
    seqs = list()
    notfirst = 0
    i = 0

    for line in filelines:
        if line[0] == '>':
            if notfirst > 0:
                seqs.append(seq.replace('\n', '').upper())
            tags.append(line[1:].replace('\n', ''))
            seq = ''
            notfirst = 1
        elif line != '\n':
            seq += line
    seqs.append(seq.replace('\n', '').upper())
    return tags, seqs


def parse_time_series_kessinger2013(ch=40):

    tags, seqs = read_alignment_kessinger2013(ch=ch)
    consensus = None
    sequences = []
    times = []
    for tag, seq in zip(tags, seqs):
        time = extract_time_kessinger2013(tag)
        if time is None and consensus is None:
            consensus = seq
        if time is not None:
            if time not in times:
                times.append(time)
            index = times.index(time)
            while index >= len(sequences):
                sequences.append([])
            sequences[index].append(binarize_genotype(seq, ref_seq=consensus))

    return sequences, times


def parse_data_kessinger2013(ch=40):

    sequences, times = parse_time_series_kessinger2013(ch=ch)
    T, L = len(times), len(sequences[0][0])
    traj = np.zeros((T, L), dtype=float)
    nTot = np.zeros((T, L), dtype=float)
    for t in range(T):
        nSeqs = len(sequences[t])
        for seq in sequences[t]:
            for l, locus in enumerate(seq):
                traj[t, l] += locus / nSeqs  # locus = 1 or 0

    traj_unique_T, loci_unique = np.unique(traj.T, axis=0, return_index=True)
    indices_filtered = []
    for l in range(len(loci_unique)):
        if np.all(traj_unique_T[l] == 0) or np.all(traj_unique_T[l] == 1):
            print(f'Locus {loci_unique[l]} is all wild-type or all mutant. ')
        else:
            indices_filtered.append(l)
    traj_unique_T, loci_unique = traj_unique_T[indices_filtered], loci_unique[indices_filtered]
    traj_unique = traj_unique_T.T

    sequences_compact = [[] for t in range(T)]
    for t in range(T):
        for seq in sequences[t]:
            sequences_compact[t].append(seq[loci_unique])

    int_cov = MPL.integrateCovarianceFromSequences(sequences_compact, times)

    data = {
        'traj': traj,
        'traj_unique': traj_unique,
        'loci_unique': loci_unique,
        'sequences': sequences_compact,
        'times': times,
        'int_cov': int_cov,
    }
    return data


def infer_selection_kessinger2013(data, params=KESSINGER_PARAMS, shrink_off_diagonal_terms=1, nonlinear_reg=None, verbose=True):

    window, mu, linear_reg = params['window'], params['mutation_rate'], params['linear_reg']
    if nonlinear_reg is None:
        nonlinear_reg = params['nonlinear_reg']
    traj, times, int_cov = data['traj_unique'], data['times'], data['int_cov']
    T, L = traj.shape
    D = MPL.computeD(traj, times, mu)

    variances = [int_cov[i, i] for i in range(L)]
    est_cov, selection_by_est_cov = EC.estimateAndInferencePipeline(traj, times, window, mu, nonlinear_reg=nonlinear_reg, linear_reg=linear_reg)

    if shrink_off_diagonal_terms != 1:
        if verbose:
            print('Before shrinking, off-diagonal terms of est_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))
            print('off-diagonal terms of true_cov has std=%.1f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(int_cov)), np.std(EC.get_diagonal_terms(int_cov))))

        est_cov = EC.shrinkOffDiagonalTerms(est_cov, shrink_off_diagonal_terms)

        if verbose:
            print('After shrinking, off-diagonal terms of est_cov  has std=%.3f, diagonal has std=%.1f' % (np.std(EC.get_off_diagonal_terms(est_cov)), np.std(EC.get_diagonal_terms(est_cov))))

    selection_by_est_cov = MPL.inferSelection(est_cov, D, linear_reg * np.identity(L))

    # print(L, int_cov.shape)
    selection_by_true_cov = MPL.inferSelection(int_cov, D, linear_reg * np.identity(L))

    cov_SL_test = MPL.computeIntegratedVariance(traj, times)
    cov_SL = copy_diagonal_terms_from(int_cov)
    selection_by_SL = MPL.inferSelection(cov_SL, D, linear_reg * np.identity(L))

    inference_results = {
        'cov_SL': cov_SL,
        'est_cov': est_cov,
        'selection_by_est_cov': selection_by_est_cov,
        'selection_by_true_cov': selection_by_true_cov,
        'selection_by_SL': selection_by_SL,
    }

    return inference_results


############################################
#
# Utility functions
#
############################################

def MAE(a, b):
    return np.mean(np.absolute(np.array(a) - np.array(b)))


def spearmanr_pearsonr_MAE(a, b):
    a, b = np.array(a), np.array(b)
    if len(a.shape) > 1:
        a, b = np.ndarray.flatten(a), np.ndarray.flatten(b)
    return stats.spearmanr(a, b)[0], stats.pearsonr(a, b)[0], MAE(a, b)


def binarize_genotype(genotype, wild_type='.', ref_seq=None):
    seq = np.zeros(len(genotype), dtype=int)
    if ref_seq is None:
        for i, locus in enumerate(genotype):
            if locus != wild_type:
                seq[i] = 1
    else:
        for i, locus in enumerate(genotype):
            if locus != ref_seq[i]:
                seq[i] = 1
    return seq


def compute_allele_traj_from_geno_traj(geno_traj, genotypes):
    L = len(genotypes[0])
    T = len(geno_traj)
    allele_traj = np.zeros((T, L))
    for t in range(T):
        for k, genotype in enumerate(genotypes):
            for i, locus in enumerate(genotype):
                allele_traj[t, i] += geno_traj[t, k] * locus
    return allele_traj


def interpolate_traj(traj, times):
    T, L = traj.shape
    for l in range(L):
        for t in range(T):
            if traj[t, l] < 0:
                tl, tr = t, t
                while traj[tl, l] < 0 and tl > 0:
                    tl -= 1
                while traj[tr, l] < 0 and tr < T - 1:
                    tr += 1
                if t == 0 or t == T - 1 or traj[tl, l] < 0 or traj[tr, l] < 0:
                    print(f'Can not interpolate. Must expolate. t={t}, l={l}, tl={tl}, {traj[tl, l]}, tr={tr}, {traj[tr, l]}')
                    return
                traj[t, l] = traj[tl, l] + (traj[tr, l] - traj[tl, l]) / (times[tr] - times[tl]) * (times[t] - times[tl])
    return traj


def copy_diagonal_terms_from(matrix):

    ret = np.zeros_like(matrix)
    for l in range(len(matrix)):
        ret[l, l] = matrix[l, l]
    return ret


############################################
#
# Plotting functions
#
############################################

def plotTraj(traj, times=None, linewidth=None, alphas=None, alpha=1, linestyle='solid', colors=None, labels=None, ylim=(-0.03, 1.03), figsize=(10, 3.3), fontsize=15, title='allele frequency trajectories', annot=False, annotTexts=None, annotXs=None, annotYs=None, plotShow=True, plotFigure=True, plotLegend=False, scatter=False, scatterOnly=False, s=0.5, marker='.', markersize=7):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    if alphas is None:
        alphas = np.full((L), alpha)
    if colors is None:
        if L >= 20:
            colors = COLORS
        else:
            colors = sns.husl_palette(L)
    if plotFigure:
        plt.figure(figsize=figsize)
    if annot:
        # Avoid annotation overlapping
        bucketToSite = {}
        bucketWidth = (times[-1] - times[0]) / (4 * L)
        if annotTexts is None:
            annotTexts = np.arange(0, L)
    for l in range(L):
        if scatterOnly:
            plt.scatter(times, traj[:, l], color=colors[l%len(colors)], alpha=alphas[l], s=s)
        elif scatter:
            plt.plot(times, traj[:, l], color=colors[l%len(colors)], alpha=alphas[l], marker=marker, markersize=markersize, linestyle=linestyle)
        else:
            plt.plot(times, traj[:, l], color=colors[l%len(colors)], linewidth=linewidth, alpha=alphas[l], label=labels[l] if labels is not None else None, linestyle=linestyle)
        if annot:
            if annotXs is not None:
                x = annotXs[l]
                if annotYs is not None:
                    y = annotYs[l]
                else:
                    i = np.argmin(np.absolute(times - x))
                    y = traj[i, l] + 0.05
            else:
                variance = np.zeros(T, dtype=float)
                for t in range(1, T - 1):
                    variance[t] = np.std(traj[t-1:t+2, l])
                t = np.argmax(variance)
                bucketIndex = int(times[t] / bucketWidth)
                while bucketIndex in bucketToSite:
                    bucketIndex += 1
                time = bucketIndex * bucketWidth
                t = np.argmin(np.absolute(times - time))
                bucketToSite[bucketIndex] = l
                x, y = times[t], traj[t, l] + 0.05
            plt.text(x, y, f'{annotTexts[l]}', fontsize=fontsize, color=colors[l%len(colors)])

    plt.ylim(ylim)
    plt.title(title, fontsize=fontsize)
    if labels is not None and plotLegend:
        plt.legend(fontsize=fontsize)
    if plotShow:
        plt.show()


def plotInferenceResults(results, ref_index=0, labels=None, use_pearsonr=False, annot=False, plot_cov=False, ylabel=None, ylim=None, plotShow=True, plotFigure=False, plotSubplots=True, plotLegend=False, suptitle="", figsize=(8, 3), wspace=0.3, fontsize=15):
    if plotFigure:
        plt.figure(figsize=figsize)
    if plotSubplots:
        fig, axes = plt.subplots(1, len(results) - 1, figsize=figsize)

    ref_result = results[ref_index]
    ref_label = labels[ref_index]

    values = np.concatenate(results)

    if ylim is None:
        pad = 0.1 * (np.max(values) - np.min(values))
        ylim = [np.min(values) - pad, np.max(values) + pad]

    n = 0
    for i, result in enumerate(results):
        if i == ref_index:
            continue
        plt.sca(axes[n])
        # plt.scatter(ref_result, result, label=None if labels is None else labels[i])
        plot_comparison(ref_result, result, xlabel=ref_label, ylabel=labels[i], ylim=ylim, use_pearsonr=use_pearsonr, annot=annot, plot_cov=plot_cov)
        n += 1

    if labels is not None and plotLegend:
        plt.legend(fontsize=fontsize)

    if plotSubplots:
        plt.subplots_adjust(wspace=wspace)
        if suptitle:
            plt.suptitle(suptitle)
    if plotShow:
        plt.show()


def plot_comparison(xs, ys, xlabel=None, ylabel=None, use_pearsonr=False, annot=False, plot_cov=False, label=None, annot_texts=None, alpha=0.6, ylim=None, yticks=None, xticks=None, xticklabels=None, plot_title=True, title_pad=4):
    L = len(xs)
    if plot_cov:
        diagonal_xs, diagonal_ys = EC.get_diagonal_terms(xs), EC.get_diagonal_terms(ys)
        off_diagonal_xs, off_diagonal_ys = EC.get_off_diagonal_terms(xs), EC.get_off_diagonal_terms(ys)
        plt.scatter(diagonal_xs, diagonal_ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label='diagonal')
        plt.scatter(off_diagonal_xs, off_diagonal_ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label='off-diagonal')
        if annot:
            index_to_pair = {}
            index = 0
            for i in range(L):
                for j in range(i):
                    index_to_pair[index] = (i, j)
                    index += 1
            for i, (x, y) in enumerate(zip(off_diagonal_xs, off_diagonal_ys)):
                if abs(y - x) > 5 and y != 0:
                    plt.text(x, y, index_to_pair[i], alpha=alpha, fontsize=10)
    else:
        plt.scatter(xs, ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label=label)
        if annot:
            for i, (x, y) in enumerate(zip(xs, ys)):
                plt.text(x, y, i, alpha=alpha, fontsize=10)
    if annot_texts is not None:
        for x, y, text in zip(xs, ys, annot_texts):
            plt.annotate(x, y, text)
    set_ticks_labels_axes(xlim=ylim, ylim=ylim, xticks=xticks, yticks=yticks, xticklabels=xticklabels, yticklabels=yticks, xlabel=xlabel, ylabel=ylabel)
    if ylim is not None:
        plt.plot(ylim, ylim, color=GREY_COLOR_RGB, alpha=alpha, linestyle='dashed')
    plt.gca().set_aspect('equal', adjustable='box')
    if plot_cov:
        plt.legend()
        if plot_title:
            pearsonr_title = ("Diagonal Pearson's " + r'$r$' + ' = %.2f' + '\n' + "Off-diagonal Pearson's " + r'$r$' + ' = %.2f') % (stats.pearsonr(diagonal_xs, diagonal_ys)[0], stats.pearsonr(off_diagonal_xs, off_diagonal_ys)[0])
            spearmanr_title = ("Diagonal Spearman's " + r'$\rho$' + ' = %.2f' + '\n' + "Off-diagonal Spearman's " + r'$\rho$' + ' = %.2f') % (stats.spearmanr(diagonal_xs, diagonal_ys)[0], stats.spearmanr(off_diagonal_xs, off_diagonal_ys)[0])
            if use_pearsonr:
                plt.title(f'{spearmanr_title}\n{pearsonr_title}', fontsize=SIZELABEL, pad=title_pad)
            else:
                plt.title(spearmanr_title, fontsize=SIZELABEL, pad=title_pad)
    elif plot_title:
        spearmanr_title = "Spearman's " + r'$\rho$' + ' = %.2f' % stats.spearmanr(xs, ys)[0]
        pearsonr_title = "Pearson's " + r'$r$' + ' = %.2f' % stats.pearsonr(xs, ys)[0]
        if use_pearsonr:
            plt.title(f'{spearmanr_title}\n{pearsonr_title}', fontsize=SIZELABEL, pad=title_pad)
        else:
            plt.title(spearmanr_title, fontsize=SIZELABEL, pad=title_pad)


def set_ticks_labels_axes(xlim=None, ylim=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None, xlabel=None, ylabel=None):

    ax = plt.gca()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if xticks is not None and xticklabels is not None:
        plt.xticks(ticks=xticks, labels=xticklabels, fontsize=SIZELABEL, rotation=0)
    else:
        plt.xticks(fontsize=SIZELABEL)

    if yticks is not None and yticklabels is not None:
        plt.yticks(ticks=yticks, labels=yticklabels, fontsize=SIZELABEL)
    else:
        plt.yticks(fontsize=SIZELABEL)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=SIZELABEL)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=SIZELABEL)

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
