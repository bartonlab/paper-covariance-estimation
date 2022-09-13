#!/usr/bin/env python
# Infer genotypes and genotype trajectories with Evoracle.

import sys
import argparse
import numpy as np                          # numerical tools
from timeit import default_timer as timer   # timer for performance
import math
import MPL                                  # MPL inference tools
import data_parser as DP
import simulation_setup as SS
sys.path.append('../../evoracle/')  # path to Evoracle
import evoracle

# EPSILON = np.finfo(np.float64).eps
EPSILON = 1e-10

SIMULATION_OUTPUT_DIR_REL = '../data/simulation_output'
EVORACLE_DIR_REL = '../data/evoracle'
EVORACLE_DIR_PALTE_REL = '../data/evoracle/PALTE'
EVORACLE_DIR_SIMULATION_REL = '../data/evoracle/simulation'
EVORACLE_DIR_SIMULATION_PARSED_OUTPUT_REL = '../data/evoracle/simulation_parsed_output'

def main(verbose=False):
    """Infer genotypes and genotype trajectories with Evoracle.
    """

    parser = argparse.ArgumentParser(description='Infer genotypes and genotype trajectories with Evoracle.')
    parser.add_argument('-o', type=str, default=None, help="prefix of output .npz file (filename is arg_list.o + f'_truncate=*_window=*.npz')")
    parser.add_argument('-s', type=int, default=0, help='index of selection')
    parser.add_argument('-n', type=int, default=0, help='index of trial/initial distribution')
    parser.add_argument('--varying_initial', action='store_true', default=False, help='whether or not run on simulations with varying initial.')
    parser.add_argument('--recombination', action='store_true', default=False, help='whether or not enable recombination')
    parser.add_argument('-r', type=float, default=0, help='recombination rate (probability for any two sequence to recombine at a generation)')
    parser.add_argument('--save_geno_traj', action='store_true', default=False, help='whether or not save inferred genotype trajectories.')

    arg_list = parser.parse_args(sys.argv[1:])

    s               = arg_list.s
    n               = arg_list.n
    r               = arg_list.r
    varying_initial = arg_list.varying_initial
    recombination   = arg_list.recombination

    # Load simulation
    sim = SS.load_simulation(s, n, directory=SIMULATION_OUTPUT_DIR_REL, varying_initial=varying_initial, recombination=recombination, r=r)
    traj = sim['traj']

    # Save traj as input for Evoracle
    if recombination:
        directory = f'{EVORACLE_DIR_SIMULATION_REL}/r={r}_s={s}_n={n}'
        obsreads_filename = f'simulation_r={r}_s={s}_n={n}_obsreads.csv'
    elif varying_initial:
        directory = f'{EVORACLE_DIR_SIMULATION_REL}/vid_s={s}_n={n}'
        obsreads_filename = f'simulation_vid_r={r}_s={s}_n={n}_obsreads.csv'
    else:
        directory = f'{EVORACLE_DIR_SIMULATION_REL}/s={s}_n={n}'
        obsreads_filename = f'simulation_s={s}_n={n}_obsreads.csv'
    obsreads_input = f'{directory}/{obsreads_filename}'
    DP.save_traj_for_evoracle(traj, obsreads_input)

    # Run Evoracle
    proposed_genotypes_output = f'{directory}/proposed_genotypes.csv'
    obs_reads_df = DP.parse_obs_reads_df(obsreads_input)
    evoracle.propose_gts_and_infer_fitness_and_frequencies(obs_reads_df, proposed_genotypes_output, directory)
    results = DP.parse_evoracle_results(obsreads_filename, directory, save_geno_traj=arg_list.save_geno_traj)

    if arg_list.o is not None:
        output = arg_list.o
    else:
        if recombination:
            output = f'evoracle_parsed_output_r={r}_s={s}_n={n}.npz'
        elif varying_initial:
            output = f'evoracle_parsed_output_vid_s={s}_n={n}.npz'
        else:
            output = f'evoracle_parsed_output_s={s}_n={n}.npz'

    np.savez_compressed(f'{EVORACLE_DIR_SIMULATION_PARSED_OUTPUT_REL}/{output}', **results)


if __name__ == '__main__': main()
