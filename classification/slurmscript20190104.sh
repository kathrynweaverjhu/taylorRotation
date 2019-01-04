#!/bin/bash

#SBATCH --job-name=ClassificationRound1
#SBATCH --time=72:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kweave23@jhu.edu
#SBATCH --output=outfile%a%x.out

source activate /home-3/kweave23@jhu.edu/work/users/kweave23/tensorflow

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/ClassificationModel.py

echo "Finished with job $SLURM_JOBID"

source deactivate