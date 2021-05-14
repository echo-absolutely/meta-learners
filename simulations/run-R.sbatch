#!/bin/bash

#SBATCH --job-name=RTest
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=10GB 
#SBATCH --time=01:00:00 

module purge 
module load r/gcc/4.0.4 

R --no-save -q -f sim1_rf.R
