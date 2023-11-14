"""F = frontal polymerization
SP = spontaneous polymerization (no output values)
NH = no holes (no polymerization)"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import scipy
import inspect
#data columns in order:
# 0: sample (sample number, no numerical meaning)
# 1: solvent, 2: concentration, 3: curing time (input variables)
# 4: anisotropy average, 5: volume fraction, 6: modulus (output variables)
# 7: polymerization type (F is good, SP has its own list)
# index 7 can be F and outputs are still NH, this has its own list too

#excelData.csv is the csv without trailing lines

# hardcoded for axis labels
inputNames = ["concentration", "curing time"]
outputNames = ["anisotropy average", "volume fraction", "modulus"]

def retrieve_name(var):
    # used to get variable name so I can put it in the title
    # takes in a variable, returns its name
    # ex: retrieve_name(solventList) = 'solventList'
    # CURRENTLY UNUSED
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# scatter plot, linear regression, and forest regression
# called in regressAll
def regress(inputList, solventName, xIndex, yIndex, xLabel=" ", yLabel =" "):
    xList = list(extract(inputList, xIndex)) # x axis stuff
    yList = list(extract(inputList, yIndex)) # y axis stuff
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
    # plot axis labels & title
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(f"{xLabel} against {yLabel} in {solventName}")
    plt.show()

def forestRegress(inputList, targetColumn):
    #targetColumn is the name of desired output, must be exact
    inputList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    convert_dict = {'anisotropy-average': float,
                    'volume-fraction': float,
                    'modulus': float
                    }

    # inputList = inputList.astype(convert_dict)

    inputListEmpty = inputList.isnull().sum()
    NAs = pd.concat([inputListEmpty], axis=1, keys=["Train"])
    NAs[NAs.sum(axis=1) > 0]



    """for col in inputList.dtypes[inputList.dtypes == "string"].index:
        for_dummy = inputList.pop(col)
        inputList = pd.concat([inputList, pd.get_dummies(for_dummy, prefix=col)], axis=1)"""

    inputList.head()

    labels = inputList.pop(targetColumn)
    x_train, x_test, y_train, y_test = train_test_split(inputList, labels, test_size=0.25)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    print(excelData.to_string())

    print(f"{inputList['concentration'].head()}")
    print(f"{inputList['curing-time'].head()}")
    #print(f"{inputList['anisotropy-average'].head()}")
    print(f"{inputList['modulus'].head()}")
    print(f"{inputList['volume-fraction'].head()}")

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(f"roc_auc value: {roc_auc}")

def extract(l, index):
    # relevant sets of data are in columns not rows, this makes them easier to parse
    return (item[index] for item in l)
def regressAll(solventList, solventName):
    # index 0 and 1 are inputs, 2, 3, 4 are outputs
    # essentially just a loop to do all regressions for one solvent
    for x in range(2):
        for y in range(2, 5):
            regress(solventList, solventName, x, y,
                    inputNames[x], outputNames[y-2])
            xList = list(extract(solventList, x))
            yList = list(extract(solventList, y))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xList, yList)
            r2 = r_value ** 2
            r2Rounded = round(r2, 2)
            r2Significance = ""
            if r2Rounded < 0.4:
                r2Significance = "indicating little to no statistical significance"
            elif r2Rounded < 0.7:
                r2Significance = "indicating a possible correlation"
            else:
                r2Significance = "indicating a likely correlation"
            print(f"R squared value of x value {inputNames[x]} and y value {outputNames[y-2]} using {solventName} is {round(r2, 3)}, {r2Significance}")






#solvent types:
#n-Pentane, cyclopentane, n-hexane, cyclohexane, n-heptane
#will have a different list for each type of solvent

if __name__ == '__main__':
    with open('chemData.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        data_list = list(csv_reader)

    excelData = pd.read_csv("excelData.csv", encoding='latin-1')
    #print(excelData.shape)

    excelData = excelData[excelData['anisotropy-average'] != "NH"]

    # removes all lines that would go into SP or NH


    # creating separate lists for SP and NH
    SPList = []
    NHList = []
    #NHList is the ones with frontal polymerization but no numerical outputs
    #SPList is the ones with spontaneous polymerization and no numerical outputs

    linesToRemove = [] #list of lines with SP or NH, to be removed later

    # putting rows with SP and NH in respective lists, removing from main list
    # HAVE to store lines to remove and then remove it later
    # removing them in the loop causes untold issues
    dataLen = len(data_list)
    for l in data_list:
        if "SP" in l:
            #print(f"SP found with id {l[0]}")
            SPList.append([l[1], l[2], l[3]])
            linesToRemove.append(l)
        elif "NH" in l:
            #print(f"NH found with id {l[0]}")
            NHList.append([l[1], l[2], l[3]])
            linesToRemove.append(l)
    for i in linesToRemove:
        data_list.remove(i) # removing every "marked" line






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

    # both of these variables are unused
    xRInput = 1 #x regression input
    yRInput = 4 #y regression input

    """
    regressAll(nPentaneList, "n-pentane")
    regressAll(cyclopentaneList, "cyclopentane")
    regressAll(nHexaneList, "n-hexane")
    regressAll(cyclohexaneList, "cyclohexane")
    regressAll(nHeptaneList, "n-heptane")"""

    forestRegress(excelData, "anisotropy-average")

    #print(excelData.to_string())


    """
    r^2 of one variable vs one variable 
    (one variable vs one variable is likely lacking context)
    (data set is also too small to glean anything statistically significant)
R squared value of x value concentration and y value anisotropy average is 0.39051461418053873
R squared value of x value concentration and y value volume fraction is 0.3224349525999838
R squared value of x value concentration and y value modulus is 0.7453394462010468
R squared value of x value curing time and y value anisotropy average is 0.40488764955475404
R squared value of x value curing time and y value volume fraction is 0.002546139200539298
R squared value of x value curing time and y value modulus is 0.09609195023779288
    
    
    
    """





