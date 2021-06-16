#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=simulation_s=6_n=17.stdout
#SBATCH --job-name="simulation_s=6_n=17"
date
source activate env_pySCA
python ../Wright-Fisher.py -N 1000 -L 50 -T 700 -i ../../data/initial/initial.npz -s ../../data/selection/selection_6.npy -o ../../data/simulation_output/simulation_output_s=6_n=17 --mu 0.001

date