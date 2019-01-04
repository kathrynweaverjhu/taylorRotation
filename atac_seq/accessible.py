#!/usr/bin/env python3

import sys

"""
Usage: ./accessible.py VISIONmusHem_ccREs.txt > VISIONmusHem_ccREs_filterkw.txt
"""

# for line in open(sys.argv[1]):
#     fields=line.strip("\r\n").split("\t")
#     newIndex = fields[3].split("_")
#     oldIndex = ["LSK", "HPC7", "CMP", "MEP", "G1E", "ER4", "CFUEad", "ERYad", "ERYfl", "CFUMK", "iMK", "GMP", "MON", "NEU", "NK", "B", "TCD4", "TCD8"]
#     newLine=""
#     for i in range(len(newIndex)):
#         newLine = newLine + (oldIndex[i]+"="+newIndex[i]+";")
#     print(fields[0], "\t", fields[1], "\t", fields[2], "\t", newLine, sep='')
    
"""
Usage: ./accessible.py VISIONmusHem_ccREs.txt > VISIONmusHem_ccREs_filter2kw.txt
"""
    
for line in open(sys.argv[1]):
    fields=line.strip("\r\n").split("\t")
    newIndex = fields[3].split("_")
    oldIndex = ["Lsk", "Hpc7", "Cmp", "Mep", "G1e", "Er4", "Cfue", "Eryad", "Eryfl", "Cfum", "Imk", "Gmp", "Mon", "Neu", "Nk", "B", "Cd4", "Cd8"]
    newLine=""
    for i in range(len(newIndex)):
        if oldIndex[i] in ["Lsk", "Cmp", "Gmp", "Mep", "Cfue", "Eryad", "Cfum", "Imk", "Mon", "Neu", "G1e", "Er4" ]:
            newLine = newLine + (oldIndex[i]+"="+newIndex[i]+";")
    print(fields[0], "\t", fields[1], "\t", fields[2], "\t", newLine, sep='')