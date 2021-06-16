#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=subsample_submission.stdout
#SBATCH --job-name="subsample_submission"
date
source activate env_pySCA
sbatch subsample_s=0_n=0.sh
sbatch subsample_s=0_n=1.sh
sbatch subsample_s=0_n=2.sh
sbatch subsample_s=0_n=3.sh
sbatch subsample_s=0_n=4.sh
sbatch subsample_s=0_n=5.sh
sbatch subsample_s=0_n=6.sh
sbatch subsample_s=0_n=7.sh
sbatch subsample_s=0_n=8.sh
sbatch subsample_s=0_n=9.sh
sbatch subsample_s=0_n=10.sh
sbatch subsample_s=0_n=11.sh
sbatch subsample_s=0_n=12.sh
sbatch subsample_s=0_n=13.sh
sbatch subsample_s=0_n=14.sh
sbatch subsample_s=0_n=15.sh
sbatch subsample_s=0_n=16.sh
sbatch subsample_s=0_n=17.sh
sbatch subsample_s=0_n=18.sh
sbatch subsample_s=0_n=19.sh
sbatch subsample_s=1_n=0.sh
sbatch subsample_s=1_n=1.sh
sbatch subsample_s=1_n=2.sh
sbatch subsample_s=1_n=3.sh
sbatch subsample_s=1_n=4.sh
sbatch subsample_s=1_n=5.sh
sbatch subsample_s=1_n=6.sh
sbatch subsample_s=1_n=7.sh
sbatch subsample_s=1_n=8.sh
sbatch subsample_s=1_n=9.sh
sbatch subsample_s=1_n=10.sh
sbatch subsample_s=1_n=11.sh
sbatch subsample_s=1_n=12.sh
sbatch subsample_s=1_n=13.sh
sbatch subsample_s=1_n=14.sh
sbatch subsample_s=1_n=15.sh
sbatch subsample_s=1_n=16.sh
sbatch subsample_s=1_n=17.sh
sbatch subsample_s=1_n=18.sh
sbatch subsample_s=1_n=19.sh
sbatch subsample_s=2_n=0.sh
sbatch subsample_s=2_n=1.sh
sbatch subsample_s=2_n=2.sh
sbatch subsample_s=2_n=3.sh
sbatch subsample_s=2_n=4.sh
sbatch subsample_s=2_n=5.sh
sbatch subsample_s=2_n=6.sh
sbatch subsample_s=2_n=7.sh
sbatch subsample_s=2_n=8.sh
sbatch subsample_s=2_n=9.sh
sbatch subsample_s=2_n=10.sh
sbatch subsample_s=2_n=11.sh
sbatch subsample_s=2_n=12.sh
sbatch subsample_s=2_n=13.sh
sbatch subsample_s=2_n=14.sh
sbatch subsample_s=2_n=15.sh
sbatch subsample_s=2_n=16.sh
sbatch subsample_s=2_n=17.sh
sbatch subsample_s=2_n=18.sh
sbatch subsample_s=2_n=19.sh
sbatch subsample_s=3_n=0.sh
sbatch subsample_s=3_n=1.sh
sbatch subsample_s=3_n=2.sh
sbatch subsample_s=3_n=3.sh
sbatch subsample_s=3_n=4.sh
sbatch subsample_s=3_n=5.sh
sbatch subsample_s=3_n=6.sh
sbatch subsample_s=3_n=7.sh
sbatch subsample_s=3_n=8.sh
sbatch subsample_s=3_n=9.sh
sbatch subsample_s=3_n=10.sh
sbatch subsample_s=3_n=11.sh
sbatch subsample_s=3_n=12.sh
sbatch subsample_s=3_n=13.sh
sbatch subsample_s=3_n=14.sh
sbatch subsample_s=3_n=15.sh
sbatch subsample_s=3_n=16.sh
sbatch subsample_s=3_n=17.sh
sbatch subsample_s=3_n=18.sh
sbatch subsample_s=3_n=19.sh
sbatch subsample_s=4_n=0.sh
sbatch subsample_s=4_n=1.sh
sbatch subsample_s=4_n=2.sh
sbatch subsample_s=4_n=3.sh
sbatch subsample_s=4_n=4.sh
sbatch subsample_s=4_n=5.sh
sbatch subsample_s=4_n=6.sh
sbatch subsample_s=4_n=7.sh
sbatch subsample_s=4_n=8.sh
sbatch subsample_s=4_n=9.sh
sbatch subsample_s=4_n=10.sh
sbatch subsample_s=4_n=11.sh
sbatch subsample_s=4_n=12.sh
sbatch subsample_s=4_n=13.sh
sbatch subsample_s=4_n=14.sh
sbatch subsample_s=4_n=15.sh
sbatch subsample_s=4_n=16.sh
sbatch subsample_s=4_n=17.sh
sbatch subsample_s=4_n=18.sh
sbatch subsample_s=4_n=19.sh
sbatch subsample_s=5_n=0.sh
sbatch subsample_s=5_n=1.sh
sbatch subsample_s=5_n=2.sh
sbatch subsample_s=5_n=3.sh
sbatch subsample_s=5_n=4.sh
sbatch subsample_s=5_n=5.sh
sbatch subsample_s=5_n=6.sh
sbatch subsample_s=5_n=7.sh
sbatch subsample_s=5_n=8.sh
sbatch subsample_s=5_n=9.sh
sbatch subsample_s=5_n=10.sh
sbatch subsample_s=5_n=11.sh
sbatch subsample_s=5_n=12.sh
sbatch subsample_s=5_n=13.sh
sbatch subsample_s=5_n=14.sh
sbatch subsample_s=5_n=15.sh
sbatch subsample_s=5_n=16.sh
sbatch subsample_s=5_n=17.sh
sbatch subsample_s=5_n=18.sh
sbatch subsample_s=5_n=19.sh
sbatch subsample_s=6_n=0.sh
sbatch subsample_s=6_n=1.sh
sbatch subsample_s=6_n=2.sh
sbatch subsample_s=6_n=3.sh
sbatch subsample_s=6_n=4.sh
sbatch subsample_s=6_n=5.sh
sbatch subsample_s=6_n=6.sh
sbatch subsample_s=6_n=7.sh
sbatch subsample_s=6_n=8.sh
sbatch subsample_s=6_n=9.sh
sbatch subsample_s=6_n=10.sh
sbatch subsample_s=6_n=11.sh
sbatch subsample_s=6_n=12.sh
sbatch subsample_s=6_n=13.sh
sbatch subsample_s=6_n=14.sh
sbatch subsample_s=6_n=15.sh
sbatch subsample_s=6_n=16.sh
sbatch subsample_s=6_n=17.sh
sbatch subsample_s=6_n=18.sh
sbatch subsample_s=6_n=19.sh
sbatch subsample_s=7_n=0.sh
sbatch subsample_s=7_n=1.sh
sbatch subsample_s=7_n=2.sh
sbatch subsample_s=7_n=3.sh
sbatch subsample_s=7_n=4.sh
sbatch subsample_s=7_n=5.sh
sbatch subsample_s=7_n=6.sh
sbatch subsample_s=7_n=7.sh
sbatch subsample_s=7_n=8.sh
sbatch subsample_s=7_n=9.sh
sbatch subsample_s=7_n=10.sh
sbatch subsample_s=7_n=11.sh
sbatch subsample_s=7_n=12.sh
sbatch subsample_s=7_n=13.sh
sbatch subsample_s=7_n=14.sh
sbatch subsample_s=7_n=15.sh
sbatch subsample_s=7_n=16.sh
sbatch subsample_s=7_n=17.sh
sbatch subsample_s=7_n=18.sh
sbatch subsample_s=7_n=19.sh
sbatch subsample_s=8_n=0.sh
sbatch subsample_s=8_n=1.sh
sbatch subsample_s=8_n=2.sh
sbatch subsample_s=8_n=3.sh
sbatch subsample_s=8_n=4.sh
sbatch subsample_s=8_n=5.sh
sbatch subsample_s=8_n=6.sh
sbatch subsample_s=8_n=7.sh
sbatch subsample_s=8_n=8.sh
sbatch subsample_s=8_n=9.sh
sbatch subsample_s=8_n=10.sh
sbatch subsample_s=8_n=11.sh
sbatch subsample_s=8_n=12.sh
sbatch subsample_s=8_n=13.sh
sbatch subsample_s=8_n=14.sh
sbatch subsample_s=8_n=15.sh
sbatch subsample_s=8_n=16.sh
sbatch subsample_s=8_n=17.sh
sbatch subsample_s=8_n=18.sh
sbatch subsample_s=8_n=19.sh
sbatch subsample_s=9_n=0.sh
sbatch subsample_s=9_n=1.sh
sbatch subsample_s=9_n=2.sh
sbatch subsample_s=9_n=3.sh
sbatch subsample_s=9_n=4.sh
sbatch subsample_s=9_n=5.sh
sbatch subsample_s=9_n=6.sh
sbatch subsample_s=9_n=7.sh
sbatch subsample_s=9_n=8.sh
sbatch subsample_s=9_n=9.sh
sbatch subsample_s=9_n=10.sh
sbatch subsample_s=9_n=11.sh
sbatch subsample_s=9_n=12.sh
sbatch subsample_s=9_n=13.sh
sbatch subsample_s=9_n=14.sh
sbatch subsample_s=9_n=15.sh
sbatch subsample_s=9_n=16.sh
sbatch subsample_s=9_n=17.sh
sbatch subsample_s=9_n=18.sh
sbatch subsample_s=9_n=19.sh

date