#!/usr/bin/env python3


"""
Usage: ./geneIDs.py scriptseq3.v3.tab 
        ./geneIDs.py scriptseq3.v3.filterkw.tab
         ./geneIDs.py scriptseq3.v3.filterkw.tab mart_export-2.txt
            ./geneIDs.py scriptseq3.v3.filterChrLockw.tab

"""
import sys

"""
Usage: ./geneIDs.py scriptseq3.v3.tab > scriptseq3.v3.filterkw.tab
"""
# for line in open(sys.argv[1]):
#     if line.startswith("E"):
#     print(line.strip("\r\n")) #filter the file

"""
Usage: ./geneIDs.py scriptseq3.v3.filterkw.tab > scriptseq3.geneIDs.txt
"""
# for line in open(sys.argv[1]):
#     fields=line.strip("\r\n").split("\t")
#     print(fields[0][:18]) #get just the geneID for use in BioMart to get chromosome and location

"""
Usage: ./geneIDs.py scriptseq3.v3.filterkw.tab mart_export-2.txt > scriptseq3.v3.filterChrLockw2.tab
"""
# filename1 = open(sys.argv[1])
# filename2 = open(sys.argv[2])
# for (i), (line1, line2) in enumerate(zip(filename1, filename2)):
#     if i == 0:
#         # fields2=line2.strip("\r\n").split("\t")
# #         print(fields2[1], "\t", fields2[2], "\t", fields2[3], "\t",
# #             "Lsk", "\t", "Lsk", "\t", "Cmp", "\t", "Cmp", "\t", "Gmp", "\t", "Gmp", "\t", "Mep", "\t", "Mep", "\t", "Cfue", "\t", "Cfue", "\t",
# #             "Eryad", "\t", "Eryad", "\t", "Cfum", "\t", "Cfum", "\t", "Imk", "\t", "Imk", "\t", "Mono", "\t", "Mono", "\t", "Neu", "\t",
# #             "Neu", "\t", "G1e", "\t", "G1e", "\t", "Er4", "\t", "Er4", sep='')
#         filename1.seek(0)
#     else:
#         fields1=line1.strip("\r\n").split("\t")
#         fields2=line2.strip("\r\n").split("\t")
#         #print(len(fields1))
#         print(fields2[1], "\t", fields2[2], "\t", fields2[3], "\t",
#             fields1[2], "\t", fields1[3], "\t", fields1[4], "\t", fields1[5], "\t", fields1[6], "\t", fields1[7], "\t", fields1[8], '\t',
#             fields1[9], '\t', fields1[10], '\t', fields1[11], '\t', fields1[12], '\t', fields1[13], '\t', fields1[14], '\t', fields1[15],
#             '\t', fields1[16], '\t', fields1[17], '\t', fields1[18], '\t', fields1[19], '\t', fields1[20], '\t', fields1[21], '\t',
#             fields1[22], '\t', fields1[23], '\t', fields1[24], '\t', fields1[25], sep='')

"""
Usage: ./geneIDs.py scriptseq3.v3.filterChrLockw2.tab > scriptseq3.v3.filter3ChrLocAvgkw2.tab
"""

# for i, line in enumerate(open(sys.argv[1])):
#     fields=line.strip("\r\n").split("\t")
# #     # if i == 0:
# # #         print(fields[0], "\t", fields[1], "\t", fields[2], "\t", fields[3], "\t", fields[4], "\t", fields[5], "\t", fields[7], "\t", fields[9],
# # #             '\t', fields[11], '\t', fields[13], '\t', fields[15], '\t', fields[17], '\t', fields[19], '\t', fields[21], '\t', fields[23], '\t',
# # #             fields[25], '\t', fields[27], sep='')
# # #     else:
# #         # print(fields[0], "\t", fields[1], "\t", fields[2], "\t", fields[3], "\t", fields[4], "\t", round(((float(fields[5])+float(fields[6]))/2),3), '\t',
# #         # round(((float(fields[7])+float(fields[8]))/2),3), '\t', round(((float(fields[9])+float(fields[10]))/2),3), '\t', round(((float(fields[11])+float(fields[12]))/2),3), '\t',
# #         # round(((float(fields[13])+float(fields[14]))/2),3), '\t', round(((float(fields[15])+float(fields[16]))/2),3), '\t', round(((float(fields[17])+float(fields[18]))/2),3), '\t',
# #         # round(((float(fields[19])+float(fields[20]))/2),3), '\t', round(((float(fields[21])+float(fields[22]))/2),3), '\t', round(((float(fields[23])+float(fields[24]))/2),3), '\t',
# #         # round(((float(fields[25])+float(fields[26]))/2),3), '\t', round(((float(fields[27])+float(fields[28]))/2),3), sep='')
# #
#     if fields[0].endswith("PATCH"):
#         pass
#     elif fields[0] != "MT":
#         print("chr"+fields[0], "\t", fields[1], "\t", fields[2], "\t", round(((float(fields[3])+float(fields[4]))/2),3), '\t',
#         round(((float(fields[5])+float(fields[6]))/2),3), '\t', round(((float(fields[7])+float(fields[8]))/2),3), '\t', round(((float(fields[9])+float(fields[10]))/2),3), '\t',
#         round(((float(fields[11])+float(fields[12]))/2),3), '\t', round(((float(fields[13])+float(fields[14]))/2),3), '\t', round(((float(fields[15])+float(fields[16]))/2),3), '\t',
#         round(((float(fields[17])+float(fields[18]))/2),3), '\t', round(((float(fields[19])+float(fields[20]))/2),3), '\t', round(((float(fields[21])+float(fields[22]))/2),3), '\t',
#         round(((float(fields[23])+float(fields[24]))/2),3), '\t', round(((float(fields[25])+float(fields[26]))/2),3), sep='')

for line in open(sys.argv[1]):
    fields=line.strip("\r\n").split("\t")
    if fields[0].endswith("PATCH"):
        pass
    elif fields[0] != "MT":
        print("chr"+fields[0], "\t", fields[1], "\t", fields[2], "\t", "Lsk="+str(round(((float(fields[3])+float(fields[4]))/2),3))+";Cmp="+
        str(round(((float(fields[5])+float(fields[6]))/2),3))+";Gmp="+str(round(((float(fields[7])+float(fields[8]))/2),3))+";Mep="+str(round(((float(fields[9])+float(fields[10]))/2),3))+
        ";Cfue="+str(round(((float(fields[11])+float(fields[12]))/2),3))+";Eryad="+str(round(((float(fields[13])+float(fields[14]))/2),3))+";Cfum="+str(round(((float(fields[15])+float(fields[16]))/2),3))+
        ";Imk="+str(round(((float(fields[17])+float(fields[18]))/2),3))+";Mon="+str(round(((float(fields[19])+float(fields[20]))/2),3))+";Neu="+str(round(((float(fields[21])+float(fields[22]))/2),3))+
        ";G1e="+str(round(((float(fields[23])+float(fields[24]))/2),3))+";Er4="+str(round(((float(fields[25])+float(fields[26]))/2),3)), sep='')
        
"""
Usage: ./geneIDs.py scriptseq3.v3.filterChrLocAvgkw2.tab > scriptseq3.v3.filter2ChrLocAvgkw2.tab       
"""

# for i, line in enumerate(open(sys.argv[1])):
#     fields = line.strip("\r\n").split("\t")
#     # if i == 0:
# #         print (line.strip("\r\n"))
# #     else:
# #         if fields[2] != "chrMT":
# #             print (line.strip("\r\n"))
#     if fields[0] != "MT" or fields[0].endswith("PATCH"):
#         print (line.strip("\r\n"))