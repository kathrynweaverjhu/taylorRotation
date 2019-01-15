#!/bin/bash

#SBATCH --job-name=Classification_simplified_noGrid_scaleDown1
#SBATCH --time=72:0:0
#SBATCH --ntasks-per-node=48
#SBATCH --mem=1000000M
#SBATCH --partition=lrgmem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kweave23@jhu.edu
#SBATCH --output=outfile_simplifiedClasses_scaleDown1_simplifiedClasses_noGrid.out


source activate /home-3/kweave23@jhu.edu/work/users/kweave23/tensorflow

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/ClassificationModel_simplifiedClasses_noGrid.py > printStatements_simplify_noGrid.out

#echo "Finished with job $SLURM_JOBID"

source deactivate
