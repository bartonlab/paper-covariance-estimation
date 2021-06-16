#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=subsample_s=2_n=6.stdout
#SBATCH --job-name="subsample_s=2_n=6"
date
source activate env_pySCA
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=1000_record=1 --sample 1000 --record 1 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=1000_record=3 --sample 1000 --record 3 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=1000_record=5 --sample 1000 --record 5 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=1000_record=10 --sample 1000 --record 10 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=500_record=1 --sample 500 --record 1 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=500_record=3 --sample 500 --record 3 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=500_record=5 --sample 500 --record 5 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=500_record=10 --sample 500 --record 10 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=100_record=1 --sample 100 --record 1 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=100_record=3 --sample 100 --record 3 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=100_record=5 --sample 100 --record 5 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=100_record=10 --sample 100 --record 10 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=50_record=1 --sample 50 --record 1 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=50_record=3 --sample 50 --record 3 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=50_record=5 --sample 50 --record 5 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=50_record=10 --sample 50 --record 10 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=10_record=1 --sample 10 --record 1 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=10_record=3 --sample 10 --record 3 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=10_record=5 --sample 10 --record 5 --compact  --intCovAtTimes 
python ../subsample.py -i ../../data/simulation_output/simulation_output_s=2_n=6.npz -o ../../data/subsample_output/subsample_output_s=2_n=6_sample=10_record=10 --sample 10 --record 10 --compact  --intCovAtTimes 

date