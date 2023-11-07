"""F = frontal polymerization
SP = spontaneous polymerization (no output values)
NH = no holes (no polymerization"""

import csv
import numpy as np
#data columns in order:
# 0: sample (sample number, no numerical meaning)
# 1: solvent, 2: concentration, 3: curing time (input variables)
# 4: anisotropy average, 5: volume fraction, 6: modulus (output variables)
# 7: polymerization type (f is good, SP or NH should be moved elsewhere)

#solvent types:
#n-Pentane, cyclopentane, n-hexane, cyclohexane, n-heptane


SPcount, NHcount = 0, 0
nPentaneCount, cyclopentaneCount, nHexaneCount, cyclohexaneCount, nHeptaneCount = 0, 0, 0, 0, 0
#will have a different list for each type of solvent

if __name__ == '__main__':
    with open('chemData.csv') as file:
        csv_reader = csv.reader(file, delimiter=';')
        data_list = list(csv_reader)

    for l in data_list: # counting how large different arrays need to be
        if l[1] == "n-Pentane":
            nPentaneCount += 1
        elif l[1] == "cyclopentane":
            cyclopentaneCount += 1
        elif l[1] == "n-hexane":
            nHexaneCount += 1
        elif l[1] == "cyclohexane":
            cyclohexaneCount += 1
        elif l[1] == "n-heptane":
            nHeptaneCount += 1
        if l[7] == "SP":
            SPcount += 1
        elif l[7] == "NH":
            NHcount += 1

    # creating separate lists for SP and NH
    SPList = [['0' for i in range(3)] for j in range(SPcount)] #length of inner list is 3 because we are only adding input variables
    NHList = [['0' for i in range(3)] for j in range(NHcount)]
    #currently there are no values with NH

    # putting rows with SP and NH in respective lists, removing from main list
    SPMarker, NHMarker = 0, 0
    for l in data_list:
        if l[7] == "SP":
            SPList[SPMarker] = [l[1], l[2], l[3]]
            SPMarker += 1
            data_list.remove(l)
        elif l[7] == "NH":
            NHList[NHMarker] = [l[1], l[2], l[3]]
            NHMarker += 1
            data_list.remove(l)
    # test print
    # print(f"SP List: {SPList}")
    # print(f"NH List: {NHList}")

    # creating each solvent's list
    # each will contain concentration, curing time, anisotropy average, volume fraction, modulus
    nPentaneList = [['0' for i in range(5)] for j in range(nPentaneCount)]
    cyclopentaneList = [['0' for i in range(5)] for j in range(cyclopentaneCount)]
    nHexaneList = [['0' for i in range(5)] for j in range(nHexaneCount)]
    cyclohexaneList = [['0' for i in range(5)] for j in range(cyclohexaneCount)]
    nHeptaneList = [['0' for i in range(5)] for j in range(nHeptaneCount)]

    # separating main list into different ones by solvent
    solventMarkers = [0, 0, 0, 0, 0] #nPentane, cyclopentane, nHexane, cyclohexane, nHeptane
    for l in data_list:
        listToAdd = [l[2], l[3], l[4], l[5], l[6]]
        #concentration, curing time, anisotropy average, volume fraction, modulus in order
        if l[1] == "n-Pentane":
            nPentaneList[solventMarkers[0]] = listToAdd
            solventMarkers[0] += 1
        elif l[1] == "cyclopentane":
            cyclopentaneList[solventMarkers[1]] = listToAdd
            solventMarkers[1] += 1
        elif l[1] == "n-hexane":
            nHexaneList[solventMarkers[2]] = listToAdd
            solventMarkers[2] += 1
        elif l[1] == "cyclohexane":
            cyclohexaneList[solventMarkers[3]] = listToAdd
            solventMarkers[3] += 1
        elif l[1] == "n-heptane":
            nHeptaneList[solventMarkers[4]] = listToAdd
            solventMarkers[4] += 1

    print(f"nPentane List: {nPentaneList}")
    print(f"cyclopentane List: {cyclopentaneList}")
    print(f"nHexane List: {nHexaneList}")
    print(f"cyclohexane List: {cyclohexaneList}")
    print(f"nHeptane List: {nHeptaneList}")





