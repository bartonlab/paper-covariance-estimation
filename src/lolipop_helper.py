import sys

import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
from scipy import stats

import importlib

import MPL

DATA_DIR = './data'
SIMULATION_OUTPUT_DIR = './data/simulation_output'
LOLIPOP_JOBS_DIR = './data/lolipop/jobs'
LOLIPOP_INPUT_DIR = './data/lolipop/input'
LOLIPOP_INPUT_DIR_REL = '../data/lolipop/input'
LOLIPOP_OUTPUT_DIR = './data/lolipop/output'
LOLIPOP_OUTPUT_DIR_REL = '../data/lolipop/output'
LOLIPOP_PARSED_OUTPUT_DIR = './data/lolipop/parsed_output'

COLORS = sns.husl_palette(20)

def generate_lolipop_command_for_simulated_data(s, n):

    input = f'{SIMULATION_DIR}/simulation_output_s={s}_n={n}'

    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def generate_lolipop_command(input, output):
    command = f"lolipop lineage --input {LOLIPOP_INPUT_DIR_REL}/{filename} --output {output_directory}"
    return command


def loadSimulation(n, simulation_directory=SIMULATION_DIR, meanS=0.03, stdS=0.01, minS=0.01, maxS=0.04, uniform=False, mu=2e-4, threshold=0.05):
    if uniform:
        filename = f'{simulation_directory}/simulation_output_min={minS}_max={maxS}_mu={mu}_th={threshold}_n={n}.npz'
    else:
        filename = f'{simulation_directory}/simulation_output_mean={meanS}_std={stdS}_mu={mu}_th={threshold}_n={n}.npz'
    dic = np.load(filename, allow_pickle=True)
    return dic


def saveTrajectoriesToTables(traj, filename, times=None, sep='\t'):
    T, L = traj.shape
    if times is None:
        times = np.arange(0, T)
    dic = {
        'Trajectory': list(np.arange(0, L)),
    }
    for i, t in enumerate(times):
        dic[t] = list(traj[i])
    df = pd.DataFrame(dic)
    df.to_csv(filename, sep=sep)


def loadOutputGenoTraj(filename, sep='\t'):
    df = pd.read_csv(filename, sep=sep)
    return df


def loadParsedOutput(file):
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


def parseOutput(directory, filename, times=None, sep='\t', saveFile=None):
    """
    Parse all output to be used later.
    """
    genotypesTable = pd.read_csv(directory + f'/{filename}.genotypes.tsv', sep=sep)
    populationsTable = pd.read_csv(directory + '/tables' + f'/{filename}.populations.tsv', sep=sep)
    trajectoriesTable = pd.read_csv(directory + '/tables' + f'/{filename}.trajectories.tsv', sep=sep)
    edgesTable = pd.read_csv(directory + '/tables' + f'/{filename}.edges.tsv', sep=sep)
    if times is None:
        times = [int(t) for t in list(genotypesTable.columns)[1:-1]]
    # else:
    #     print(f'Using given times: ', times)
    #     print(f'Times in table: ', [int(t) for t in list(genotypesTable.columns)[1:-1]])

    genotypeToMembers, memberToGenotype = parseGenotypeMemberMapping(genotypesTable)
    parentToGenotype, genotypeToParent = parseEdges(edgesTable)
    genoTraj = parseGenotypeTrajectories(populationsTable)
    cladeTraj = computeCladeTrajectories(genoTraj, parentToGenotype)
    traj = parseAlleleTrajectories(trajectoriesTable)
    reconstructedTraj = computeReconstructedTraj(cladeTraj, genotypeToMembers, parentToGenotype)

    data = {'genotypeToMembers': genotypeToMembers, 'memberToGenotype': memberToGenotype, 'genoTraj': genoTraj, 'cladeTraj': cladeTraj, 'traj': traj, 'reconstructedTraj': reconstructedTraj, 'times': times, 'parentToGenotype': parentToGenotype, 'genotypeToParent': genotypeToParent, }

    if saveFile is not None:
        f = open(saveFile + '.npz', 'wb')
        np.savez_compressed(f, **data)
        f.close()

    return data


def parseGenotypeMemberMapping(df):
    G = len(df) + 1  # This table does not have a row for genotype-0, hence add one here.
    allMembers = set()
    for i, row in df.iterrows():
        members = parseMembersList(row['members'])
        allMembers.update(members)
    if 0 not in allMembers:
        shiftToZeroIndexed = True
    else:
        shiftToZeroIndexed = False
    genotypeToMembers = [[] for g in range(G)]
    for i, row in df.iterrows():
        g = parseGenotypeIndex(row['Genotype'])
        genotypeToMembers[g] = parseMembersList(row['members'], shiftToZeroIndexed=shiftToZeroIndexed)
    L = np.sum([len(members) for members in genotypeToMembers])
    memberToGenotype = [None for l in range(L)]
    for g, members in enumerate(genotypeToMembers):
        for member in members:
            memberToGenotype[member] = g
    return genotypeToMembers, memberToGenotype


def parseMembersList(members, shiftToZeroIndexed=False):
    if shiftToZeroIndexed:
        return [int(i) - 1 for i in str(members).strip().split('|')]
    else:
        return [int(i) for i in str(members).strip().split('|')]


def parseGenotypeTrajectories(df):
    genotypes = np.unique(df['Identity'])
    generations = sorted(np.unique(df['Generation']))
    generationToIndex = {}
    for i, generation in enumerate(generations):
        generationToIndex[generation] = i
    G = len(genotypes)
    T = len(generations)
    genoTraj = np.zeros((T, G), dtype=float)
    for i, row in df.iterrows():
        generation = int(row['Generation'])
        t = generationToIndex[generation]
        g = parseGenotypeIndex(row['Identity'])
        freq = row['Population'] / 100
        genoTraj[t, g] = freq
    return genoTraj


def parseGenotypeIndex(identity):
    return int(identity.split('-')[-1])


def parseAlleleTrajectories(df):
    generations = [_ for _ in list(df.columns) if _.isdigit()]
    if 0 not in set(df['Trajectory']):
        shiftToZeroIndexed = True
    else:
        shiftToZeroIndexed = False
    L = len(df)
    T = len(generations)
    traj = np.zeros((T, L), dtype=float)
    for i, row in df.iterrows():
        l = row['Trajectory']
        if shiftToZeroIndexed:
            l -= 1
        for t, generation in enumerate(generations):
            traj[t, l] = row[generation]
    return traj


def computeCladeTrajectories(genoTraj, parentToGenotype):

    T, G = genoTraj.shape
    cladeTraj = np.copy(genoTraj)
    visited = set()
    for g in range(G):
        computeCladeTrajectories_dfs(g, visited, cladeTraj, parentToGenotype)

    return cladeTraj


def computeCladeTrajectories_dfs(g, visited, cladeTraj, parentToGenotype):
    if g in visited:
        return
    T, G = cladeTraj.shape
    for d in parentToGenotype[g]:
        computeCladeTrajectories_dfs(d, visited, cladeTraj, parentToGenotype)
    if len(parentToGenotype[g]) == 1:
        d = parentToGenotype[g][0]
        for t in range(T):
            cladeTraj[t, g] += cladeTraj[t, d]
    visited.add(g)


def computeReconstructedTraj(cladeTraj, genotypeToMembers, parentToGenotype):

    L = np.sum(len(_) for _ in genotypeToMembers)
    T, G = cladeTraj.shape
    reconstructedTraj = np.zeros((T, L), dtype=float)

    for l in range(L):
        pass
    pass


def parseEdges(df):
    parentToGenotype = {}
    genotypeToParent = {}
    genotypes = set()
    for i, row in df.iterrows():
        genotype = parseGenotypeIndex(row['Identity'])
        if genotype not in genotypes:
            genotypes.add(genotype)
        parent = parseGenotypeIndex(row['Parent'])
        genotypeToParent[genotype] = parent
        if parent in parentToGenotype:
            parentToGenotype[parent].append(genotype)
        else:
            parentToGenotype[parent] = [genotype]
    for genotype in genotypes:
        if genotype not in parentToGenotype:
            parentToGenotype[genotype] = []
    return parentToGenotype, genotypeToParent


def getTrueGenoTrajFromSimulation(data):
    genotypeToIndex = {}
    genotypeToTrajectory = {}
    T, L = data['traj'].shape
    N = np.sum(data['nVec'][0])
    for t in range(T):
        nVec_t, sVec_t = data['nVec'][t], data['sVec'][t]
        for i, seq in enumerate(sVec_t):
            seq_tuple = tuple(seq)
            if seq_tuple not in genotypeToIndex:
                genotypeToIndex[seq_tuple] = len(genotypeToIndex)
            index = genotypeToIndex[seq_tuple]
            if seq_tuple not in genotypeToTrajectory:
                genotypeToTrajectory[seq_tuple] = {}
            genotypeToTrajectory[seq_tuple][t] = nVec_t[i] / N
    genoTraj = np.zeros((T, len(genotypeToIndex)), dtype=float)
    for genotype in genotypeToTrajectory:
        index = genotypeToIndex[genotype]
        for t, freq in genotypeToTrajectory[genotype].items():
            genoTraj[t, index] = freq

    return genoTraj


def assignColorsToGenotypes(graph, ancestor=0):
    colors, alphas = {}, {}
    colors[0], alphas[0] = 'black', 1.0
    for i, genotype in enumerate(graph[0]):
        colors[genotype] = COLORS[i%len(COLORS)]  #AP.hexToRGB(AP.COLORS[i%len(AP.COLORS)])
        alphas[genotype] = 1.0
    color_i = 3
    for genotype in graph[0]:
        assignColorsToGenotypes_dfs(genotype, graph, colors, alphas, color_i)
    return colors, alphas


def assignColorsToGenotypes_dfs(node, graph, colors, alphas, color_i, color_inherit=0.2, alpha_decay=0.8):
    for i, nei in enumerate(graph[node]):
        if len(graph[node]) == 1:
            colors[nei]
        # colors[nei] = tuple(np.array(colors[node]) * color_inherit + np.array(AP.hexToRGB(AP.COLORS[i%len(AP.COLORS)])) * (1 - color_inherit))
        colors[nei] = tuple(np.array(colors[node]) * color_inherit + np.array(COLORS[i%len(COLORS)]) * (1 - color_inherit))
        alphas[nei] = alpha_decay * alphas[node]
        assignColorsToGenotypes_dfs(nei, graph, colors, alphas, color_i)


############################################
#
# Inference
#
############################################


def inferencePipeline(data, mu=1e-10, regularization=1):
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


def inferForSimulations(num_trials=40, meanS=0.03, stdS=0.01, minS=0.01, maxS=0.04, uniform=False, mu=2e-4, threshold=0.05, muForInference=1e-6, regularization=1, lolipop_parsed_output_dir='./test_data/lolipop/parsed_output', simulation_output_dir='./data/simulation_output'):
    intCov_all, selection_all, fitness_all, intCovError_all = [], [], [], []
    for n in np.arange(num_trials):
        if uniform:
            filename = f'simulation_traj_min={minS}_max={maxS}_mu={mu}_th={threshold}_n={n}'
        else:
            filename = f'simulation_traj_mean={meanS}_std={stdS}_mu={mu}_th={threshold}_n={n}'
        file = f'{lolipop_parsed_output_dir}/{filename}.npz'
        data = loadParsedOutput(file)
        intCov, selection, fitness = inferencePipeline(data, mu=muForInference, regularization=regularization)
        dic = loadSimulation(n, meanS=meanS, stdS=stdS, minS=minS, maxS=maxS, uniform=uniform, mu=mu, simulation_directory=simulation_output_dir)
        intCovError = intCov - dic['cov']

        intCov_all.append(intCov)
        selection_all.append(selection)
        fitness_all.append(fitness)
        intCovError_all.append(intCovError)
    return intCov_all, selection_all, fitness_all, intCovError_all
