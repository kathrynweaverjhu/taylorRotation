#!/usr/bin/env python3

import sys
import numpy as np
import argparse as ap
import subprocess #use check_output() since bedtools sends to stdout


"""
Usage: ~/taylorRotation/preprocessing/tensorMatrixPD.py --outfile filename.npz --IDEAScalls ~/taylorRotation/preprocessing/data/files --RNAseq ~/taylorRotation/preprocessing/data/files (scriptseq3.v3.filter4ChrLocAvgkw2.bed) --ATACseq ~/taylorRotation/preprocessing/data/files (VISIONmusHem_ccREs_filter2kw.txt)
"""
"""allows for -h for this file and for specific files to be given for annotation type"""
parser = ap.ArgumentParser(description='Preprocessing of data for machine learnings')
parser.add_argument('--IDEAScalls', action='store', nargs='+', type=str, required = True, help='List of files with IDEAS calls')
parser.add_argument('--RNAseq', action='store', nargs='+', type=str, required = True, help="List of files with RNAseq data")
parser.add_argument('--ATACseq', action='store', nargs='+', type=str, required = True, help="List of files with ATACseq data")
parser.add_argument('--outfile', action='store', nargs=1, type=str, required = True, help="Name of file to save Annotated/matched matrices")
args=parser.parse_args()

fileList1 = args.IDEAScalls #IDEAScalls files
file2 = args.RNAseq[0] #RNAseq files
file3 = args.ATACseq[0] #ATACseq files
outfile = args.outfile[0]

"""annotating data"""
numQuiescent=0
numTotal = 0
cell_types =[]
loc={} #dictionary key: cellType, chrom, start, end; value: [ideasLabel, sequence]
for file in fileList1: #IDEAScalls ##filter out if label == 0
    lhs,rhs=file.split("Seg") #file names are in the format ideasVisionV20p8Seg$CELLTYPEgetfa.bed -> ideasVisionV20p8 $CELLTYPEgetfa.bed
    cellTypeI, rhs2 = rhs.split("getfa") # -> $CELLTYPE .bed
    cell_types.append(cellTypeI)
    for line in open(file):
        numTotal += 1
        fields=line.strip("\r\n").split("\t")
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        if int(fields[3]) == 0: #quiescent
            numQuiescent += 1
            pass
        else:
            loc[cellTypeI,chrom,start,end] = [int(fields[3]), fields[9]] #ideaslabel=fields[3], sequence=fields[9]
        

"""annotating with RNAseq - file ex: chr    geneStart    geneEnd    Lsk=0.0;Cmp=72.0;Mep=0.0;G1e=180.0;Er4=0.0;Cfue=0.0;Eryad=0.0;Cfum=0.0;Imk=0.0;Gmp=0.0;Mon=0.0;Neu=0.0;"""
for cell_type in cell_types:
    bedtools_closest_out = subprocess.check_output("bedtools closest -k 3 -a ~/taylorRotation/preprocessing/data/ideasVisionV20p8Seg{}getfa.bed -b {} -t last".format(cell_type, file2), shell=True).decode("utf-8").splitlines()
    for line in bedtools_closest_out: #3 lines correspond to the same location in the IDEAS file
        fields=line.strip("\r\n;").split("\t")
        chrom, startL, endL, startf, endf = fields[0], int(fields[1]), int(fields[2]), int(fields[11]), int(fields[12])
        if int(fields[3]) == 0: #quiescent
            pass
        else:
            cellTypeIndex = fields[13].split(";")
            tpm = dict([x.split('=') for x in cellTypeIndex])[cell_type]
            loc[cell_type,chrom,startL,endL].append(float(tpm))


"""annotating with ATACseq - file ex: chr    geneStart    geneEnd    Lsk=0;Cmp=0;Mep=0;G1e=0;Er4=0;Cfue=0;Eryad=0;Cfum=0;Imk=0;Gmp=0;Mon=0;Neu=0;"""
ATACseqContainment = 0.5 #minimum containment or sequence overlap for the annotation to be added to the loc dictionary list for that genome location
for cell_type in cell_types:
    bedtools_out = subprocess.check_output("bedtools intersect -loj -a ~/taylorRotation/preprocessing/data/ideasVisionV20p8Seg{}getfa.bed -b {}".format(cell_type, file3), shell=True).decode("utf-8").splitlines()
    for line in bedtools_out:
        fields=line.strip("\r\n;").split("\t")
        chrom, startL, endL, startf, endf = fields[0], int(fields[1]), int(fields[2]), int(fields[11]), int(fields[12])
        if int(fields[3]) == 0: #quiescent
            pass
        elif fields[10] == ".":
            loc[cell_type, chrom, startL, endL].append(0) #no containment/or overlap between IDEAS and ATAC
        else:
            cellTypeIndex = fields[13].split(";")
            aonab = dict([x.split('=') for x in cellTypeIndex])[cell_type]
            containment = (min(endL, endf) - max(startL, startf))/(endL-startL)
            if containment >= ATACseqContainment:
                loc[cell_type,chrom,startL,endL].append(int(aonab))
            else:
                loc[cell_type,chrom,startL,endL].append(0) #below specified containment level

"""separating annotated data into matched arrays"""
cellTypeIndex = []
labels = []
sequences = []
RNA_seq = []
ATAC_seq = []

cellType2Index = {
    "Cfue":0,
    "Cfum":1,
    "Cmp":2,
    "Er4":3,
    "Eryad":4,
    "G1e":5,
    "Gmp":6,
    "Imk":7,
    "Lsk":8,
    "Mep":9,
    "Mon":10,
    "Neu":11,
    "B":12,
    "Cd4":13,
    "Cd8":14,
    "Clp":15,
    "Eryfl":16,
    "Hpc7":17,
    "Mk":18,
    "Nk":19,
}

sequenceDict={
    "a":0,
    "c":1,
    "g":2,
    "t":3,
    'n':[0,1,2,3],
}

#loc - key:cellType,chrom,start,end 
#loc - value=[label, sequence, RNAseq1, RNAseq2, RNAseq3, ATACseq1, ATACseq2,...,ATACseqi] where i is unknown
for (cellType, chrom, start, end) in loc:
    cellTypeIndex.append(cellType2Index.setdefault(cellType, 20))
    labels.append(loc[cellType, chrom, start, end][0])
    RNA_seq.append(loc[cellType,chrom,start,end][2:5]) #Append a list of 3
    if 1 in loc[cellType,chrom,start,end][5:]: #necessity for only 1 for the whole thing to be 1
        ATAC_seq.append(1)
    else:
        ATAC_seq.append(0)
    sequence = loc[cellType,chrom,start,end][1]
    sequenceTensorial = np.zeros((4,42400)) #2D array of shape (4,42400)

    for i,n in enumerate(sequence.lower()):
        sequenceTensorial[sequenceDict[n],i]=1
        

    sequenceTensorial[:,len(sequence)+1:42400] = np.nan #for any column values that are blank because the sequence isn't 42400 nucleotides, NaN

    sequences.append(sequenceTensorial) #append the 2D array to the list

cellTypeIndex = np.array(cellTypeIndex, dtype=np.intp)
labels = np.array(labels, dtype=np.intp)
sequences = np.array(sequences) #make the list of 2D arrays a 3D array
RNA_seq = np.array(RNA_seq, dtype=np.float_)
ATAC_seq = np.array(ATAC_seq, dtype=np.bool_)    

f = open(outfile, 'wb')
np.savez(f, cellTypeIndex = cellTypeIndex, labels = labels, sequences=sequences, RNA_seq = RNA_seq, ATAC_seq = ATAC_seq)
f.close()
