"""F = frontal polymerization
SP = spontaneous polymerization (no output values)
NH = no holes (no polymerization)"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#data columns in order:
# 0: sample (sample number, no numerical meaning)
# 1: solvent, 2: concentration, 3: curing time (input variables)
# 4: anisotropy average, 5: volume fraction, 6: modulus (output variables)
# 7: polymerization type (F is good, SP has its own list)
# index 7 can be F and outputs are still NH, this has its own list too

# scatter plot, linear regression, and forest regression
def regress(xList, yList):
    lRegressor = LinearRegression()
    fRegressor = RandomForestRegressor(100, random_state=0)
    xList2d = np.array(xList).reshape((-1, 1)) #needed for regressors
    lRegressor.fit(xList2d, yList) #trains linear regressor
    fRegressor.fit(xList2d, yList) #trains forest regressor

    xGrid = np.arange(min(xList), max(xList), 0.01)
    xGrid = xGrid.reshape((len(xGrid)), 1) #creates a suitable x axis

    yPred = lRegressor.predict(xGrid)
    yPredForest = fRegressor.predict(xGrid) #trains regressors on the x axis

    plt.scatter(list(xList), list(yList)) #scatter plot
    plt.plot(xGrid, yPred, color='red') #linear regression
    plt.plot(xGrid, yPredForest, color='green') #forest regression
    plt.show()


def extract(l, index):
    #relevant sets of data are in columns not rows, this makes them easier to parse
    return (item[index] for item in l)

def numericCheck(l):
    # certain lines of data are not being detected as having NH or SP, hope this helps
    #CURRENTLY UNUSED and likely nonfunctional
    return not all(str(s).strip('-').replace('.', '').isdigit() for s in l)



#solvent types:
#n-Pentane, cyclopentane, n-hexane, cyclohexane, n-heptane
#will have a different list for each type of solvent

if __name__ == '__main__':
    with open('chemData.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        data_list = list(csv_reader)


    # creating separate lists for SP and NH
    SPList = []
    NHList = []
    #NHList is the ones with frontal polymerization but no numerical outputs
    #SPList is the ones with spontaneous polymerization and no numerical outputs


    # putting rows with SP and NH in respective lists, removing from main list
    #print(len(data_list))

    linesToRemove = []
    dataLen = len(data_list)
    for i in range(dataLen):
        l = data_list[i]
        if "SP" in l:
            print(f"SP found with id {l[0]}")
            SPList.append([l[1], l[2], l[3]])
            #data_list.remove(l)
            linesToRemove.append(l)
        elif "NH" in l:
            print(f"NH found with id {l[0]}")
            NHList.append([l[1], l[2], l[3]])
            #data_list.remove(l)
            linesToRemove.append(l)
    # test print
    # print(f"SP List: {SPList}")
    # print(f"NH List: {NHList}")

    for i in linesToRemove:
        data_list.remove(i)


    # creating each solvent's list
    # each will contain concentration, curing time, anisotropy average, volume fraction, modulus
    #nPentaneList = [[0 for i in range(5)] for j in range(nPentaneCount)]
    nPentaneList = []
    cyclopentaneList = []
    nHexaneList = []
    cyclohexaneList = []
    nHeptaneList = []

    # separating main list into different ones by solvent
    #print(data_list)
    for l in data_list[1:]:
        #print(l)
        listToAdd = [float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6])]
        #concentration, curing time, anisotropy average, volume fraction, modulus in order
        if l[1] == "n-Pentane":
            #nPentaneList[solventMarkers[0]] = listToAdd
            nPentaneList.append(listToAdd)
        elif l[1] == "cyclopentane":
            cyclopentaneList.append(listToAdd)
        elif l[1] == "n-hexane":
            nHexaneList.append(listToAdd)
        elif l[1] == "cyclohexane":
            cyclohexaneList.append(listToAdd)
        elif l[1] == "n-heptane":
            nHeptaneList.append(listToAdd)

    #test prints:
    """print(f"nPentane List: {nPentaneList}")
    print(f"cyclopentane List: {cyclopentaneList}")
    print(f"nHexane List: {nHexaneList}")
    print(f"cyclohexane List: {cyclohexaneList}")
    print(f"nHeptane List: {nHeptaneList}")"""


    #make sure to wrap both input lists in list()
    #extract(list, index) to get relevant index - sets of data are in columns
    #concentration, curing time, anisotropy average, volume fraction, modulus in order
    #neat but unlikely to give relevant results, we need results from input variables in tandem

    regress(list(extract(nPentaneList, 1)), list(extract(nPentaneList, 2)))





