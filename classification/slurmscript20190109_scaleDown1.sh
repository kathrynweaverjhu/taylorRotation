#!/bin/bash

#SBATCH --job-name=Classification_scaleDown1
#SBATCH --time=72:0:0
#SBATCH --partition=lrgmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=1000000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kweave23@jhu.edu
#SBATCH --output=outfile20190109_scaleDown1.out

source activate /home-3/kweave23@jhu.edu/work/users/kweave23/tensorflow

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/ClassificationModel_scaleDown.py > printStatements_scaleDown1.out

echo "Finished with job $SLURM_JOBID"

source deactivate
