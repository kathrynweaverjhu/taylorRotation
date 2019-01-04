#!/bin/bash

#B Cd4 Cd8 Cfue Cfum Clp Cmp Er4 Eryad Eryfl G1e Gmp Hpc7 Imk Lsk Mep Mk
for CELLTYPE in Mon Neu Nk
do
    bedtools getfasta -fi ~/taylorRotation/mm10Genome/allChromff.fa -bed ~/taylorRotation/IDEAS_calls/Archive/ideasVisionV20p8Seg$CELLTYPE.bed -bedOut > ideasVisionV20p8Seg${CELLTYPE}getfa.bed
    echo "done"
    echo $CELLTYPE
done
