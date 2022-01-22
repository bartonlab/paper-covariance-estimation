#!/usr/bin/env python
# A simple Wright-Fisher simulation with an additive fitness model.

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import MPL                                  # MPL inference tools


def main(verbose=False):
    """ Simulate Wright-Fisher evolution of a population and save the results. """

    parser = argparse.ArgumentParser(description='Wright-Fisher evolutionary simulation.')
    parser.add_argument('-o',   type=str,   default='data/simoutput',         help='output path and filename')
    parser.add_argument('-N',   type=int,   default=1000,                     help='population size')
    parser.add_argument('-L',   type=int,   default=50,                       help='sequence length')
    parser.add_argument('-T',   type=int,   default=700,                     help='number of generations in simulation')
    parser.add_argument('-s',   type=str,   default=None,                     help='.npy file containning selection matrix')
    parser.add_argument('-i',   type=str,   default=None,                     help='.npz file containing initial distribution')
    parser.add_argument('--mu', type=float, default=1.0e-3,                   help='mutation rate')
    parser.add_argument('--covAtEachTime', action='store_true', default=False, help='whether or not record covariance at each time point')

    arg_list = parser.parse_args(sys.argv[1:])

    N = arg_list.N
    L = arg_list.L
    T = arg_list.T
    mu = arg_list.mu
    out_str = arg_list.o
    selection = np.load(arg_list.s)

    class Species:
        """A group of sequences with the same genotype.

        Attributes:
            n: Number of members.
            sequence: Genotype of the species.
            f: Fitness value of the species.
        """
        def __init__(self, n=1, f=1., **kwargs):
            """Initializes clone-specific variables."""
            self.n = n
            if 'sequence' in kwargs:
                self.sequence = np.array(kwargs['sequence'])
                self.f = fitness(selection, self.sequence)
            else:
                self.sequence = np.zeros(L)
                self.f = f

        @classmethod
        def clone(cls, s):
            """Returns a new copy of the input Species"""
            return cls(n=1, f=s.f, sequence=[k for k in s.sequence])

        def mutate(self):
            """ Mutate and return self + new species."""
            newSpecies = []
            if self.n > 0:
                nMut = np.random.binomial(self.n, mu * L)  # get number of individuals that mutate
                self.n -= nMut
                mutSites = np.random.randint(L, size=nMut)  # choose mutation sites at random
                for i in mutSites:
                    s = Species.clone(self)
                    s.sequence[i] = 1 - s.sequence[i]  # mutate the randomly-selected site
                    s.f = fitness(selection, s.sequence)
                    newSpecies.append(s)
            if self.n > 0:
                newSpecies.append(self)
            return newSpecies

    start = timer()  # track running time
    pop = []  # Population
    sVec = []  # Array of sequences at each time point
    nVec = []  # Array of sequence counts at each time point
    record = 1  # record data every (record) generations

    if arg_list.i:
        # Use the given intial distribution
        initial = np.load(arg_list.i)
        initial_sVec = []
        initial_nVec = []
        for i in range(len(initial['counts'])):
            seq = np.array([int(j) for j in initial['sequences'][i]])
            n_seq = initial['counts'][i]
            initial_sVec.append(seq)
            initial_nVec.append(n_seq)
            pop.append(Species(n=n_seq, sequence=seq))
        sVec.append(np.array(initial_sVec))
        nVec.append(np.array(initial_nVec))
    else:
        # Start with all sequences being wild-type
        pop = [Species(n=N)]
        sVec = [np.array([np.zeros(L)])]
        nVec = [np.array([N])]

    # Evolve the population
    for t in range(T):
        # printUpdate(t, tEnd)    # status check

        # Select species to replicate
        r = np.array([s.n * s.f for s in pop])
        p = r / np.sum(r) # probability of selection for each species (genotype)
        n = np.random.multinomial(N, pvals=p) # selected number of each species

        # Update population size and mutate
        newPop = []
        seq_to_index = {} # keep track of species in newPop to prevent recording of duplicate sequences
        for i in range(len(pop)):
            pop[i].n = n[i] # set new number of each species
            mutated_pop = pop[i].mutate()
            for j, species in enumerate(mutated_pop):
                key = species.sequence.tobytes()
                if key in seq_to_index:
                    index = seq_to_index[key]
                    newPop[index].n += species.n
                else:
                    seq_to_index[key] = len(newPop)
                    newPop.append(species)
        pop = newPop

        if t % record == 0:
            nVec.append(np.array([s.n        for s in pop]))
            sVec.append(np.array([s.sequence for s in pop]))

    end = timer()
    print('\nTotal simulation time: %lfs, average per generation %lfs' % ((end - start), (end - start) / float(T)))

    start  = timer()

    traj = computeTrajectories(sVec, nVec)  # compute trajectories of allele frequencies
    times = np.array(range(len(sVec))) * record

    # Compute the integrated covariance matrix
    q = 2
    totalCov = np.zeros((L, L), dtype=float)
    covAtEachTime = MPL.processStandard(sVec, [i / N for i in nVec], times, q, totalCov, covAtEachTime=arg_list.covAtEachTime)

    f = open(out_str + '.npz', 'wb')
    if arg_list.covAtEachTime:
        np.savez_compressed(f, nVec=nVec, sVec=sVec, traj=traj, cov=totalCov, mu=mu, times=times, covAtEachTime=covAtEachTime)
    else:
        np.savez_compressed(f, nVec=nVec, sVec=sVec, traj=traj, cov=totalCov, mu=mu, times=times)
    f.close()

    end = timer()
    print('\nCompute & output time: %lfs' % (end - start))


def fitness(selection, seq):
    """Computes fitness for a binarized sequence according to a given selection."""

    seq = [int(i) for i in seq]
    h = 1
    for i, allele in enumerate(seq):
        h += allele * selection[i]
    return h


def printUpdate(current, end, bar_length=20):
    """Prints an update of the simulation status. h/t Aravind Voggu on StackOverflow."""

    percent = float(current) / end
    dash = ''.join(['-' for k in range(int(round(percent * bar_length)-1))]) + '>'
    space = ''.join([' ' for k in range(bar_length - len(dash))])

    sys.stdout.write("\rSimulating: [{0}] {1}%".format(dash + space, int(round(percent * 100))))
    sys.stdout.write("\n")
    sys.stdout.flush()


def computeTrajectories(sVec, nVec):
    """Computes allele frequency trajectories."""

    N = np.sum(nVec[0])
    L = len(sVec[0][0])
    traj = np.zeros((len(nVec), L), dtype=float)
    for t in range(len(nVec)):
        for i in range(L):
            traj[t, i] = np.sum([sVec[t][k][i] * nVec[t][k] for k in range(len(sVec[t]))]) / N
    return traj


if __name__ == '__main__': main()
