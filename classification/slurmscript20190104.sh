#!/bin/bash

#SBATCH --job-name=ClassificationRound1
#SBATCH --time=72:0:0
#SBATCH --partition=lrgmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=1000000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kweave23@jhu.edu
#SBATCH --output=outfile%a%x.out

source activate /home-3/kweave23@jhu.edu/work/users/kweave23/tensorflow

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/tensorMatrixPD.py --outfile "savedMatrices.npz" \
--IDEAScalls "/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCfuegetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCfumgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegCmpgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegEr4getfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegEryadgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegG1egetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegGmpgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegImkgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegLskgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegMepgetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegMongetfa.bed" \
"/home-3/kweave23@jhu.edu/work/users/kweave23/data/ideasVisionV20p8SegNeugetfa.bed" \
--RNAseq "/home-3/kweave23@jhu.edu/work/users/kweave23/data/scriptseq3.v3.filter4ChrLocAvgkw2.bed" \
--ATACseq "/home-3/kweave23@jhu.edu/work/users/kweave23/data/VISIONmusHem_ccREs_filter2kw.txt" > printStatements.out

python /home-3/kweave23@jhu.edu/work/users/kweave23/classification_scripts/ClassificationModel.py

echo "Finished with job $SLURM_JOBID"

source deactivate
