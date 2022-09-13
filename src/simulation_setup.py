#!/usr/bin/env python
# Generate initial genotype distribution, and selection coefficients for Wright-Fisher simulations.

import numpy as np

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

# GitHub directories
DATA_DIR = './data'
SELECTION_DIR = './src/selection'
INITIAL_DIR = './src/initial'
SIMULATION_OUTPUT_DIR = './data/simulation_output'
SUBSAMPLE_OUTPUT_DIR = './data/subsample_output'
ESTIMATION_OUTPUT_DIR = './data/estimation_output'
JOB_DIR = './jobs'
EVORACLE_SIMULATION_PARSED_OUTPUT_DIR = './data/evoracle/simulation_parsed_output'

# relative directories looking from shell scripts in JOB_DIR
SRC_DIR_REL = '../src'
DATA_DIR_REL = '../data'
SELECTION_DIR_REL = '../src/selection'
INITIAL_DIR_REL = '../src/initial'
SIMULATION_OUTPUT_DIR_REL = '../data/simulation_output'
SUBSAMPLE_OUTPUT_DIR_REL = '../data/subsample_output'
ESTIMATION_OUTPUT_DIR_REL = '../data/estimation_output'
EVORACLE_DIR_SIMULATION_REL = '../data/evoracle/simulation'

def generate_selection_coefficients(overwrite=False):
    """Generates 10 sets of selection coefficients.

    This was how the 10 sets of selection coefficients were generated.
    Note that since some selectons are randomly sampled, re-run this function will yield
    different selections.
    """
    selections = np.zeros((NUM_SELECTIONS, L), dtype=float)

    # 2 sets of selections are drawn from uniform distribution in range [-0.05, 0.05], with 0 and 10 neutral alleles, respectively.
    num_nonneutral_sites = [50, 40]
    min_s, max_s = -0.05, 0.05
    for s in range(2):
        cur_s = min_s
        nonneutral_sites = np.random.choice(np.arange(L), size=num_nonneutral_sites[s], replace=False)
        step = (max_s - min_s) / num_nonneutral_sites[s]
        for l in nonneutral_sites:
            selections[s, l] = cur_s
            cur_s += step

    # 4 sets of selections are drawn from a tri-modal distribution, a combination of three Gaussian distributions with standard deviation of 0.01, centered at -0.03, 0, 0.03 respectively.
    std_s = 0.01
    for s in range(2, 6):
        for l in range(L):
            which = np.random.randint(0, 3)
            if which == 0:
                selections[s, l] = np.random.normal(-0.03, std_s)
            elif which == 1:
                selections[s, l] = np.random.normal(0, std_s)
            else:
                selections[s, l] = np.random.normal(0.03, std_s)

    # 4 sets of selections are drawn from a Gaussian distribution with standard deviation of 0.02, centered at 0.
    for s in range(6, NUM_SELECTIONS):
        for l in range(L):
            selections[s, l] = np.random.normal(0, 0.02)

    if overwrite:
        for s in range(NUM_SELECTIONS):
            np.save(SELECTION_DIR + f'/selection_{s}.npy', selections[s])
    else:
        for s in range(NUM_SELECTIONS):
            try:
                load_selection(s)
            except:
                np.save(SELECTION_DIR + f'/selection_{s}.npy', selections[s])


def generate_default_initial_distribution(overwrite=False):

    sequences, counts = generate_random_genotype_distribution(4)
    if overwrite:
        np.savez_compressed(INITIAL_DIR + f'/initial', sequences=sequences, counts=counts)
    else:
        try:
            initial = np.load(INITIAL_DIR + f'/initial.npz')
        except:
            np.savez_compressed(INITIAL_DIR + f'/initial', sequences=sequences, counts=counts)


def generate_initial_distributions(overwrite=False):
    """ Generates {NUM_TRIALS} different initial distributions of genotypes used in our simulations. """
    # 20 distributions take random composition of initial population. Number of initial genotypes goes from 5~24. Their initial sizes are also random. (Each 10 individuals are randomly assigned to one of the founding genotypes.)
    num_genotypes_list = list(np.arange(5, 25))
    for i, num_genotypes in enumerate(num_genotypes_list):
        sequences, counts = generate_random_genotype_distribution(num_genotypes)
        if overwrite:
            np.savez_compressed(INITIAL_DIR + f'/initial_{i}', sequences=sequences, counts=counts)
        else:
            try:
                load_initial(i + 1)
            except:
                np.savez_compressed(INITIAL_DIR + f'/initial_{i}', sequences=sequences, counts=counts)


def generate_random_genotype_distribution(num_genotypes):
    sequences = [generate_a_random_genotype() for _ in range(num_genotypes)]
    counts = [0 for seq in sequences]
    for _ in range(N // 10):
        counts[np.random.randint(0, num_genotypes)] += 1
    for i in range(num_genotypes):
        if counts[i] == 0:
            counts[i] = 1
            counts[np.argmax(counts)] -= 1
    for i in range(num_genotypes):
        counts[i] *= 10

    return sequences, counts


def generate_three_genotype_distribution():
    """Generates the initial distribution of genotypes used in our simulations.

    Initially the population mainly consists of three genotypes, each starting with 333
    sequences, the other one sequence is wild-type.
    For every 7 alleles, 1 allele is mutant in all three genotypes. 3 allele are mutant
    in two genotypes; The other 3 are mutant in only one of three genotypes.
    """
    sequences = []
    counts = []
    seq = np.zeros((3, L), dtype = int)
    for n in range(7):
        seq[0, 7 * n] = seq[1, 7 * n] = seq[2, 7 * n] = 1
        seq[0, 7 * n + 1] = seq[1, 7 * n + 1] = 1
        seq[1, 7 * n + 2] = seq[2, 7 * n + 2] = 1
        seq[0, 7 * n + 3] = seq[2, 7 * n + 3] = 1
        seq[0, 7 * n + 4] = 1
        seq[1, 7 * n + 5] = 1
        seq[2, 7 * n + 6] = 1

    sequences.append(np.array([int(i) for i in seq[0]]))
    counts.append(N//3)
    sequences.append(np.array([int(i) for i in seq[1]]))
    counts.append(N//3)
    sequences.append(np.array([int(i) for i in seq[2]]))
    counts.append(N//3)
    sequences.append(np.array([int(i) for i in CM.bin(0, L)]))
    counts.append(N - int(N//3) * 3)

    np.savez_compressed(INITIAL_DIR + '/initial', sequences=sequences, counts=counts)


def generate_a_random_genotype():
    seq = np.zeros(L, dtype=int)
    for l in range(L):
        seq[l] = np.random.randint(0, 2)
    return seq


def print_initial_distribution(init_dist, max_line=10):
    """Prints initial genotypes and their counts."""

    for i, seq in enumerate(init_dist['sequences']):
        if i >= max_line:
            break
        print(f'Genotype {i+1}', ''.join([str(allele) for allele in seq]), f'count={init_dist["counts"][i]}')
    print()


def binary_seq(a, L):
    """ Returns binary seq of length L for the ath genotype. """

    return format(a, '0' + str(L) + 'b')


def generate_shell_script(path2job, jobname, command, hours=4, mem=16, ncpus=1, env='cov_est', local=False):
    """Generates a shell script to run on HPCC cluster, or locally."""

    if local:
        fp = open(path2job + '/' + jobname + '_local.sh', 'w')
        fp.write('#!/bin/sh\n')
        fp.write(command)
        fp.close()
    else:
        fp = open(path2job + '/' + jobname + '.sh', 'w')
        fp.write('#!/bin/bash -l\n')
        fp.write(f'#SBATCH --nodes=1\n#SBATCH --ntasks=1\n')
        fp.write(f'#SBATCH --cpus-per-task={ncpus}\n')
        fp.write(f'#SBATCH --mem={mem}G\n#SBATCH --time={hours}:00:00\n')
        fp.write(f'#SBATCH --output={jobname}.stdout\n')
        fp.write(f'#SBATCH --job-name="{jobname}"\n')
        fp.write(f'date\nsource activate {env}\n')
        fp.write(command)
        fp.write('\ndate')
        fp.close()


def generate_simulation_scripts(env='cov_est', varying_initial=False, local=False, overwrite=True, recombination=False, r=1e-5):
    if recombination:
        prefix = f'simulation_r={r}'
    elif varying_initial:
        prefix = f'simulation_vid'
    else:
        prefix = f'simulation'
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            jobname = f'{prefix}_s={s}_n={n}'
            initial_filename = f'initial_{n}.npz' if varying_initial else 'initial.npz'
            job_pars = {'-N': N,
                        '-L': L,
                        '-T': T,
                        '-i': f'{INITIAL_DIR_REL}/{initial_filename}',
                        '-s': f'{SELECTION_DIR_REL}/selection_{s}.npy',
                        '-o': f'{SIMULATION_OUTPUT_DIR_REL}/{prefix}_output_s={s}_n={n}',
                        '--mu': MU}
            if recombination:
                job_pars['--recombination'] = ''
                job_pars['-r'] = r
            command = f'python {SRC_DIR_REL}/Wright-Fisher.py '
            command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
            command += '\n'
            generate_shell_script(JOB_DIR, jobname, command, hours=1, env=env, local=local)

    jobname = f'{prefix}_submission'
    command = ''
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            if (not overwrite) and test_single_simulation(s, n, varying_initial=varying_initial, recombination=recombination, r=r):
                continue
            if local:
                command += f'sh {prefix}_s={s}_n={n}_local.sh\n'
            else:
                command += f'sbatch {prefix}_s={s}_n={n}.sh\n'
    generate_shell_script(JOB_DIR, jobname, command, env=env, local=local)


def generate_subsample_scripts(env='cov_est', varying_initial=False, overwrite=True, recombination=False, r=1e-5, partition='intel'):
    if recombination:
        prefix = f'subsample_r={r}'
        sim_prefix = f'simulation_r={r}'
    elif varying_initial:
        prefix = f'subsample_vid'
        sim_prefix = f'simulation_vid'
    else:
        prefix = f'subsample'
        sim_prefix = f'simulation'

    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            jobname = f'{prefix}_s={s}_n={n}'
            command = ''
            for sample in SAMPLE:
                for record in RECORD:
                    job_pars = {'-i': f'{SIMULATION_OUTPUT_DIR_REL}/{sim_prefix}_output_s={s}_n={n}.npz',
                                '-o': f'{SUBSAMPLE_OUTPUT_DIR_REL}/{prefix}_output_s={s}_n={n}_sample={sample}_record={record}',
                                '--sample': sample,
                                '--record': record,
                                '--compact': '',
                                '--intCovAtTimes': ''}
                    command += f'python {SRC_DIR_REL}/subsample.py '
                    command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
                    command += '\n'
            generate_shell_script(JOB_DIR, jobname, command, hours=1, env=env)

    jobname = f'{prefix}_submission'
    command = ''
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            if (not overwrite) and test_subsample_for_single_simulation(s, n, varying_initial=varying_initial, recombination=recombination, r=r):
                continue
            command += f'sbatch -p {partition} {prefix}_s={s}_n={n}.sh\n'
    generate_shell_script(JOB_DIR, jobname, command, env=env)


def generate_subsample_script_with_time_series_covariance(s=4, n=0, p=0, q=0, env='cov_est', varying_initial=False, overwrite=True, recombination=False, r=1e-5, partition='intel'):

    sample, record = SAMPLE[p], RECORD[q]

    if recombination:
        prefix = f'subsample_covAtEachTime_r={r}'
        sim_prefix = f'simulation_r={r}'
    elif varying_initial:
        prefix = f'subsample_covAtEachTime_vid'
        sim_prefix = f'simulation_vid'
    else:
        prefix = f'subsample_covAtEachTime'
        sim_prefix = f'simulation'

    jobname = f'{prefix}_s={s}_n={n}'
    command = ''
    job_pars = {'-i': f'{SIMULATION_OUTPUT_DIR_REL}/{sim_prefix}_output_s={s}_n={n}.npz',
                '-o': f'{SUBSAMPLE_OUTPUT_DIR_REL}/{prefix}_output_s={s}_n={n}_sample={sample}_record={record}',
                '--sample': sample,
                '--record': record,
                '--compact': '',
                '--covAtEachTime': ''}
    command += f'python {SRC_DIR_REL}/subsample.py '
    command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
    command += '\n'
    generate_shell_script(JOB_DIR, jobname, command, hours=1, env=env)


def generate_estimate_scripts(env='cov_est', truncate_list=TRUNCATE, window_list=WINDOW, complete_output=False, varying_initial=False, recombination=False, r=1e-5, partition='intel'):
    job_prefix = 'estimate'
    if complete_output:
        job_prefix += '_complete'
    if recombination:
        job_prefix += f'_r={r}'
        est_prefix = f'estimation_r={r}'
        sub_prefix = f'subsample_r={r}'
        sim_prefix = f'simulation_r={r}'
    elif varying_initial:
        job_prefix += f'_vid'
        est_prefix = f'estimation_vid'
        sub_prefix = f'subsample_vid'
        sim_prefix = f'simulation_vid'
    else:
        est_prefix = f'estimation'
        sub_prefix = f'subsample'
        sim_prefix = f'simulation'
    for tr in truncate_list:
        for win in window_list:
            jobname = f'{job_prefix}_truncate={tr}_window={win}'
            command = ''
            job_pars = {'-N': N,
                        '-L': L,
                        '-T': T,
                        '-truncate': tr,
                        '-window': win,
                        '-num_selections': NUM_SELECTIONS,
                        '-num_trials': NUM_TRIALS,
                        '-s': f'{SELECTION_DIR_REL}/selection',
                        '-i': f'{SUBSAMPLE_OUTPUT_DIR_REL}/{sub_prefix}_output',
                        '-o': f'{ESTIMATION_OUTPUT_DIR_REL}/{est_prefix}_output',
                        }
            if not complete_output:
                job_pars['--minimal_size'] = ''
            command += f'python {SRC_DIR_REL}/estimate_batch_job.py '
            command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
            command += '\n'
            generate_shell_script(JOB_DIR, jobname, command, hours=tr // 10, env=env)

    jobname = f'{job_prefix}_submission'
    command = ''
    for tr in truncate_list:
        for win in window_list:
            command += f'sbatch -p {partition} {job_prefix}_truncate={tr}_window={win}.sh\n'
    generate_shell_script(JOB_DIR, jobname, command, env=env)


def generate_evoracle_scripts(env='evoracle', varying_initial=False, recombination=False, r=0, save_geno_traj=False, partition='intel', overwrite=False):
        job_prefix = 'evoracle'
        if recombination:
            job_prefix += f'_r={r}'
        elif varying_initial:
            job_prefix += f'_vid'
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                jobname = f'{job_prefix}_s={s}_n={n}'
                command = ''
                job_pars = {'-s': s,
                            '-n': n,
                            }
                if recombination:
                    job_pars['--recombination'] = ''
                    job_pars['-r'] = r
                if varying_initial:
                    job_pars['--varying_initial'] = ''
                if save_geno_traj:
                    job_pars['--save_geno_traj'] = ''
                command += f'python {SRC_DIR_REL}/evoracle_batch_job.py '
                command += ' '.join([k + ' ' + str(v) for k, v in job_pars.items()])
                command += '\n'
                generate_shell_script(JOB_DIR, jobname, command, hours=12, env=env)

        jobname = f'{job_prefix}_mkdir'
        command = ''
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                if recombination:
                    command += f'mkdir {EVORACLE_DIR_SIMULATION_REL}/r={r}_s={s}_n={n}\n'
                elif varying_initial:
                    command += f'mkdir {EVORACLE_DIR_SIMULATION_REL}/vid_s={s}_n={n}\n'
                else:
                    command += f'mkdir {EVORACLE_DIR_SIMULATION_REL}/s={s}_n={n}\n'
        generate_shell_script(JOB_DIR, jobname, command, env=env)

        jobname = f'{job_prefix}_submission'
        command = ''
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                if overwrite or not test_single_evoracle(s, n, varying_initial=varying_initial, recombination=recombination, r=r):
                    command += f'sbatch -p {partition} {job_prefix}_s={s}_n={n}.sh\n'
        generate_shell_script(JOB_DIR, jobname, command, env=env)


def scp_to_cluster(local_file, cluster_dir, directory=False):
    if directory == True:
        print(f'scp -r {local_file} yli354@cluster.hpcc.ucr.edu:{cluster_dir}')
    else:
        print(f'scp {local_file} yli354@cluster.hpcc.ucr.edu:{cluster_dir}')


def scp_from_cluster(local_dir, cluster_file, directory=False):
    if directory == True:
        print(f'scp -r yli354@cluster.hpcc.ucr.edu:{cluster_file} {local_dir}')
    else:
        print(f'scp yli354@cluster.hpcc.ucr.edu:{cluster_file} {local_dir}')


def load_selection(s):
    """ Loads one of {NUM_SELECTIONS} sets of selection coefficients. """

    return np.load(SELECTION_DIR + f'/selection_{s}.npy')


def load_selections():

    return np.array([load_selection(s) for s in range(NUM_SELECTIONS)])


def load_initial(i):
    """ Loads one of {NUM_TRIALS} initial distributions of genotypes. """

    return np.load(INITIAL_DIR + f'/initial_{i}.npz')


def load_default_initial():
    """ Loads the default initial distributions of four genotypes. """

    return np.load(INITIAL_DIR + f'/initial.npz')


def load_simulation(s, n, directory=SIMULATION_OUTPUT_DIR, varying_initial=False, recombination=False, r=1e-5):
    if recombination:
        prefix = f'simulation_r={r}'
    elif varying_initial:
        prefix = f'simulation_vid'
    else:
        prefix = f'simulation'
    file = f'{directory}/{prefix}_output_s={s}_n={n}.npz'
    return np.load(file, allow_pickle=True)


def load_subsample(s, n, sample, record, varying_initial=False, recombination=False, r=1e-5, covAtEachTime=False):
    if recombination:
        prefix = f'subsample_r={r}'
    elif varying_initial:
        prefix = f'subsample_vid'
    elif covAtEachTime:
        prefix = f'subsample_covAtEachTime'
    else:
        prefix = f'subsample'
    file = f'{SUBSAMPLE_OUTPUT_DIR}/{prefix}_output_s={s}_n={n}_sample={sample}_record={record}.npz'
    return np.load(file, allow_pickle=True)


def load_estimation(truncate, window, varying_initial=False, recombination=False, r=1e-5, complete=False):
    if recombination:
        prefix = f'estimation_r={r}'
    elif varying_initial:
        prefix = f'estimation_vid'
    else:
        prefix = f'estimation'

    postfix = '_complete' if complete else ''

    return np.load(ESTIMATION_OUTPUT_DIR + f'/{prefix}_output_truncate={truncate}_window={window}{postfix}.npz')


def load_evoracle(s, n, directory=EVORACLE_SIMULATION_PARSED_OUTPUT_DIR, varying_initial=False, recombination=False, r=1e-5):
    if recombination:
        prefix = f'evoracle_r={r}'
    elif varying_initial:
        prefix = f'evoracle_vid'
    else:
        prefix = f'evoracle'
    file = f'{directory}/{prefix}_parsed_output_s={s}_n={n}.npz'
    return np.load(file, allow_pickle=True)


def load_traj_example(s, n, p=0, q=0, tr=5, varying_initial=False, recombination=False, r=1e-5):
    """Loads allele frequency trajectories of one particular simulation.

    Args:
        s: 0 ~ 9, indicating one of 10 sets of selection coefficients.
        n: 0 ~ 19, indicating one of 20 trials.
        p: 0 ~ 4, indicating one of 5 sampling depths.
        q: 0 ~ 3, indicating one of 3 sampling time intervals.
        tr: 0 ~ 5, indicating one of 6 truncating lengths.
    """

    if p == 0 and q == 0:
        dic = load_simulation(s, n, varying_initial=varying_initial, recombination=recombination, r=r)
    else:
        dic = load_subsample(s, n, SAMPLE[p], RECORD[q], varying_initial=varying_initial, recombination=recombination, r=r)
    traj = dic['traj'][:TRUNCATE[tr] + 1]
    return traj


def load_cov_example(s=4, n=0, truncate=700, window=20, sample_selected=0, record_selected=0, recombination=False, r=1e-5):
    """Loads intgrated covariances with a certain data length and a certain window choice."""

    dic = load_estimation(truncate, window, recombination=recombination, r=r)

    if dic['size'] == 'minimal':
        return dic['int_cov'][s], dic['int_dcov'][s], dic['int_dcov_uncalibrated'][s]
    elif dic['size'] == 'medium':
        return dic['int_cov'][s, n], dic['int_dcov'][s, n], dic['int_dcov_uncalibrated'][s, n]
    else: # dic['size'] == 'complete'
        return dic['int_cov'][s, n, p, q], dic['int_dcov'][s, n, p, q], dic['int_dcov_uncalibrated'][s, n, p, q]


def test_load_simulation(varying_initial=False, recombination=False, r=1e-5):
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            try:
                sim = load_simulation(s, n, varying_initial=varying_initial, recombination=recombination, r=r)
                traj, nVec, sVec, times, cov, mu = sim['traj'], sim['nVec'], sim['sVec'], sim['times'], sim['cov'], sim['mu']
            except:
                print(s, n)


def test_load_subsample(varying_initial=False, recombination=False, r=1e-5):
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            try:
                for sample in SAMPLE:
                    for record in RECORD:
                        sub = load_subsample(s, n, sample, record, varying_initial=varying_initial, recombination=recombination, r=r)
                        traj, times, cov, mu, intCovTimes, intCovAtTimes = sub['traj'], sub['times'], sub['cov'], sub['mu'], sub['intCovTimes'], sub['intCovAtTimes']
            except:
                print(s, n)


def test_load_estimation(varying_initial=False, recombination=False, r=1e-5):
    for truncate in TRUNCATE:
        for window in WINDOW:
            try:
                est = load_estimation(truncate, window, varying_initial=varying_initial, recombination=recombination, r=r)
                for key in list(est.keys()):
                    est[key]
            except:
                print(truncate, window)


def test_load_evoracle(varying_initial=False, recombination=False, r=1e-5):
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            if not test_single_evoracle(s, n, varying_initial=varying_initial, recombination=recombination, r=r):
                print(s, n)


def test_single_evoracle(s, n, varying_initial=False, recombination=False, r=1e-5):
    try:
        res = load_evoracle(s, n, varying_initial=varying_initial, recombination=recombination, r=r)
        for key in list(res.keys()):
            res[key]
        return True
    except:
        return False


def test_single_simulation(s, n, varying_initial=False, recombination=False, r=1e-5):
    try:
        sim = load_simulation(s, n, varying_initial=varying_initial, recombination=recombination, r=r)
        traj, nVec, sVec, times, cov, mu = sim['traj'], sim['nVec'], sim['sVec'], sim['times'], sim['cov'], sim['mu']
        return True
    except:
        return False


def test_subsample_for_single_simulation(s, n, varying_initial=False, recombination=False, r=1e-5):
    try:
        for sample in SAMPLE:
            for record in RECORD:
                sub = load_subsample(s, n, sample, record, varying_initial=varying_initial, recombination=recombination, r=r)
                traj, times, cov, mu, intCovTimes, intCovAtTimes = sub['traj'], sub['times'], sub['cov'], sub['mu'], sub['intCovTimes'], sub['intCovAtTimes']
        return True
    except:
        return False


def test_load_initial():
    for n in range(NUM_TRIALS):
        try:
            initial = load_initial(n)
            assert np.sum(initial['counts']) == N
        except:
            print(n)
