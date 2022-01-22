#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=estimate_complete_truncate=700_window=20.stdout
#SBATCH --job-name="estimate_complete_truncate=700_window=20"
date
source activate env_pySCA
python ../estimate.py -N 1000 -L 50 -T 700 -truncate 700 -window 20 -num_selections 10 -num_trials 20 -s ../../src/selection/selection -i ../../data/subsample_output/subsample_output -o ../../data/estimation_output/estimation_output

date