#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=estimate_submission.stdout
#SBATCH --job-name="estimate_submission"
date
source activate env_pySCA
sbatch -p batch estimate_truncate=200_window=0.sh
sbatch -p batch estimate_truncate=200_window=1.sh
sbatch -p batch estimate_truncate=200_window=2.sh
sbatch -p batch estimate_truncate=200_window=3.sh
sbatch -p batch estimate_truncate=200_window=4.sh
sbatch -p batch estimate_truncate=200_window=5.sh
sbatch -p batch estimate_truncate=200_window=10.sh
sbatch -p batch estimate_truncate=200_window=20.sh
sbatch -p batch estimate_truncate=200_window=40.sh
sbatch -p batch estimate_truncate=200_window=80.sh
sbatch -p batch estimate_truncate=200_window=160.sh
sbatch -p batch estimate_truncate=300_window=0.sh
sbatch -p batch estimate_truncate=300_window=1.sh
sbatch -p batch estimate_truncate=300_window=2.sh
sbatch -p batch estimate_truncate=300_window=3.sh
sbatch -p batch estimate_truncate=300_window=4.sh
sbatch -p batch estimate_truncate=300_window=5.sh
sbatch -p batch estimate_truncate=300_window=10.sh
sbatch -p batch estimate_truncate=300_window=20.sh
sbatch -p batch estimate_truncate=300_window=40.sh
sbatch -p batch estimate_truncate=300_window=80.sh
sbatch -p batch estimate_truncate=300_window=160.sh
sbatch -p batch estimate_truncate=400_window=0.sh
sbatch -p batch estimate_truncate=400_window=1.sh
sbatch -p batch estimate_truncate=400_window=2.sh
sbatch -p batch estimate_truncate=400_window=3.sh
sbatch -p batch estimate_truncate=400_window=4.sh
sbatch -p batch estimate_truncate=400_window=5.sh
sbatch -p batch estimate_truncate=400_window=10.sh
sbatch -p batch estimate_truncate=400_window=20.sh
sbatch -p batch estimate_truncate=400_window=40.sh
sbatch -p batch estimate_truncate=400_window=80.sh
sbatch -p batch estimate_truncate=400_window=160.sh
sbatch -p batch estimate_truncate=500_window=0.sh
sbatch -p batch estimate_truncate=500_window=1.sh
sbatch -p batch estimate_truncate=500_window=2.sh
sbatch -p batch estimate_truncate=500_window=3.sh
sbatch -p batch estimate_truncate=500_window=4.sh
sbatch -p batch estimate_truncate=500_window=5.sh
sbatch -p batch estimate_truncate=500_window=10.sh
sbatch -p batch estimate_truncate=500_window=20.sh
sbatch -p batch estimate_truncate=500_window=40.sh
sbatch -p batch estimate_truncate=500_window=80.sh
sbatch -p batch estimate_truncate=500_window=160.sh
sbatch -p batch estimate_truncate=600_window=0.sh
sbatch -p batch estimate_truncate=600_window=1.sh
sbatch -p batch estimate_truncate=600_window=2.sh
sbatch -p batch estimate_truncate=600_window=3.sh
sbatch -p batch estimate_truncate=600_window=4.sh
sbatch -p batch estimate_truncate=600_window=5.sh
sbatch -p batch estimate_truncate=600_window=10.sh
sbatch -p batch estimate_truncate=600_window=20.sh
sbatch -p batch estimate_truncate=600_window=40.sh
sbatch -p batch estimate_truncate=600_window=80.sh
sbatch -p batch estimate_truncate=600_window=160.sh
sbatch -p batch estimate_truncate=700_window=0.sh
sbatch -p batch estimate_truncate=700_window=1.sh
sbatch -p batch estimate_truncate=700_window=2.sh
sbatch -p batch estimate_truncate=700_window=3.sh
sbatch -p batch estimate_truncate=700_window=4.sh
sbatch -p batch estimate_truncate=700_window=5.sh
sbatch -p batch estimate_truncate=700_window=10.sh
sbatch -p batch estimate_truncate=700_window=20.sh
sbatch -p batch estimate_truncate=700_window=40.sh
sbatch -p batch estimate_truncate=700_window=80.sh
sbatch -p batch estimate_truncate=700_window=160.sh

date