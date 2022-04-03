#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=estimate_complete_submission.stdout
#SBATCH --job-name="estimate_complete_submission"
date
source activate env_pySCA
sbatch -p batch estimate_complete_truncate=700_window=20.sh

date