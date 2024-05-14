#!/bin/bash

#SBATCH --job-name=First_Search_MIF_run_1
#SBATCH --output=out/if-pf%A_%a.out
#SBATCH --error=error/if-pf%A_%a.err
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=3000
#SBATCH --cpus-per-task=1
#SBATCH --time=0:45:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=/results/slurm_%j.txt
#SBATCH --array=0-99
#SBATCH --chdir=/scratch/jc12343/AppliedStochasticAnalysis/homework/project

module purge
module load python/intel/3.8.6

## Execute the desired python file
python3 pf.py --i $SLURM_ARRAY_TASK_ID
