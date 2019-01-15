#!/bin/bash

#SBATCH --job-name=Classification_simplified_scaleDown1_part2
#SBATCH --time=72:0:0
#SBATCH --ntasks-per-node=48
#SBATCH --mem=1000000M
#SBATCH --partition=lrgmem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kweave23@jhu.edu
#SBATCH --output=outfile_simplifiedClasses_scaleDown1_part2.out


source activate /home-3/kweave23@jhu.edu/work/users/kweave23/tensorflow

#python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/tensorMatrix_simplifyClasses.py --outfile "savedMatrices_simplifiedClasses.npz" \
#--IDEAScalls "/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCfuegetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCfumgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCmpgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegEr4getfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegEryadgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegG1egetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegGmpgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegImkgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegLskgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegMepgetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegMongetfa.bed" \
#"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegNeugetfa.bed" \
#--RNAseq "/home-3/kweave23@jhu.edu/work/users/kweave23/data/scriptseq3.v3.filter4ChrLocAvgkw2.bed" \
#--ATACseq "/home-3/kweave23@jhu.edu/work/users/kweave23/data/VISIONmusHem_ccREs_filter2kw.txt" > printStatements_simplify.out

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/ClassificationModel_simplifiedClasses.py > printStatements2_simplify.out

#echo "Finished with job $SLURM_JOBID"

source deactivate
