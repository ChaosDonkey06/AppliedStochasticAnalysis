#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=ASA_project_eafk
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --output=/results/slurm_%j.text
#SBATCH --array=0-99

module purge
module load python/intel/3.8.6

RUNDIR=/scratch/jc12343/AppliedStochasticAnalysis/homework/project
cd $RUNDIR

## Execute the desired python file
python3 eakf.py --i $SLURM_ARRAY_TASK_ID
