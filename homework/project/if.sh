#!/bin/bash

#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:45:00
#SBATCH --mem=4GB
#SBATCH --job-name=ASA_project_eafk
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=/results/slurm_%j.txt
#SBATCH --array=0-99
#SBATCH --chdir=/scratch/jc12343/AppliedStochasticAnalysis/homework/project

module purge
module load python/intel/3.8.6

## Execute the desired python file
python3 pf.py --i $SLURM_ARRAY_TASK_ID %>% test.txt
