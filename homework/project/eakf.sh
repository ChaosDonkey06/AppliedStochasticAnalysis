#!/bin/bash

#SBATCH --job-name=ASA_project_eakf
#SBATCH --output=out/if-eakf%A_%a.out
#SBATCH --error=error/if-eakf%A_%a.err
#SBATCH --time=0:45:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jc12343@nyu.edu
#SBATCH --partition=cs
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=10GB
#SBATCH --cpus-per-task=1

#SBATCH --chdir=/scratch/jc12343/AppliedStochasticAnalysis/homework/project


module purge
module load python/intel/3.8.6

RUNDIR=/scratch/jc12343/AppliedStochasticAnalysis/homework/project
cd $RUNDIR

## Execute the desired python file
python3 eakf.py --i $SLURM_ARRAY_TASK_ID
