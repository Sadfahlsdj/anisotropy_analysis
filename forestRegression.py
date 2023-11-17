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
from sklearn import metrics

import scipy
import inspect
#data columns in order:
# 0: sample (sample number, no numerical meaning)
# 1: solvent, 2: concentration, 3: curing time (input variables)
# 4: anisotropy average, 5: volume fraction, 6: modulus (output variables)
# 7: polymerization type (F is good, SP has its own list)
# index 7 can be F and outputs are still NH, this has its own list too

"""def forestRegress(inputList, targetColumn):
    #targetColumn is the name of desired output, must be exact
    inputListEmpty = inputList.isnull().sum()
    NAs = pd.concat([inputListEmpty], axis=1, keys=["Train"])
    NAs[NAs.sum(axis=1) > 0]

    # following 3 lines should be commented
    for col in inputList.dtypes[inputList.dtypes == "string"].index:
        for_dummy = inputList.pop(col)
        inputList = pd.concat([inputList, pd.get_dummies(for_dummy, prefix=col)], axis=1)

    inputList.head()

    labels = inputList[targetColumn]
    x_train, x_test, y_train, y_test = train_test_split(inputList, labels, test_size=0.25)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    # print(inputList.to_string())

    # following 5 lines are test prints, should be commented
    print(f"{inputList['concentration'].head()}")
    print(f"{inputList['curing-time'].head()}")
    print(f"{inputList['anisotropy-average'].head()}")
    print(f"{inputList['modulus'].head()}")
    print(f"{inputList['volume-fraction'].head()}")
    
    #roc is meant for binary or otherwise categorical outputs
    #not fit for this model

    #false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #roc_auc = auc(false_positive_rate, true_positive_rate)
    #print(f"roc_auc value: {roc_auc}")
    """

def forest_regression(df, outputName, solventName):
    # xVars is concentration & curing time; yVars is anisotropy, volume fraction, and modulus
    # use first yVars for data split by solvent, second one for boiling point data
    xVars = df.drop(['anisotropy-average', 'volume-fraction', 'modulus', 'sample'], axis=1)
    # yVars = df.drop(['concentration', 'curing-time', 'sample'], axis=1)
    yVars = df.drop(['boiling-point', 'concentration', 'curing-time', 'sample'], axis=1)
    # test_size being 0.3 means that 70% of the data goes into train, which is needed
    xTrain, xValid, yTrain, yValid = train_test_split(xVars, yVars, test_size=0.3, random_state=42)

    # create regressor and train
    rg = RandomForestRegressor(n_estimators=100, random_state=42)
    rg.fit(xTrain, yTrain)
    yPred = rg.predict(xValid)
    # change datatype to make it easier to use
    yPred = pd.DataFrame(yPred, columns=['anisotropy-average', 'volume-fraction', 'modulus'])

    """
    anisotropyActual = yValid['anisotropy-average'].tolist()
    volumeFractionActual = yValid['volume-fraction'].tolist()
    modulusActual = yValid['modulus'].tolist()

    anisotropyPred = yPred['anisotropy-average'].tolist()
    volumeFractionPred = yPred['volume-fraction'].tolist()
    modulusPred = yPred['modulus'].tolist()
    """
    # print(anisotropyPred)

    # what output is used for the graph
    yValidGraph = yValid[outputName].tolist()
    yPredGraph = yPred[outputName].tolist()

    plt.figure(figsize=(10, 10))
    plt.scatter(yValidGraph, yPredGraph, color="red", label=f"Comparison between Actual and Predicted Data in {solventName}")
    plt.legend()
    plt.grid()
    plt.title(f"Actual vs Predicted Values for {outputName} in {solventName}")
    plt.xlabel("Predicted data")
    plt.ylabel("Actual data")
    plt.show()

    # r^2, multioutput='raw_values' prints each separately which is what i want
    r2 = metrics.r2_score(yValid, yPred, multioutput='raw_values')
    r2total = metrics.r2_score(yValid, yPred)
    print(f"R squared score for this data set is {r2} for anisotropy, volume fraction, and modulus respectively in {solventName}")
    print(f"Overall R squared score for this data set is {r2total}")

if __name__ == "__main__":
    # excelData has solvent names, bPointData(boilingPointData) has boiling points
    excelData = pd.read_csv("excelData.csv", encoding='latin-1')
    bPointData = pd.read_csv("boilingPointData.csv", encoding='latin-1')
    # print(excelData.shape)

    # separating NH and SP data
    # ideally i'd separate the two but there just isn't enough data

    columnNames = excelData.columns
    columnNamesBoilingPoint = bPointData.columns

    NHList = pd.DataFrame(columns=columnNames)
    NHListBoilingPoint = pd.DataFrame(columns=columnNamesBoilingPoint)

    # some modulus values are very small, will try removing them and running again
    smallModulusListBP = pd.DataFrame(columns=columnNamesBoilingPoint)

    for index, row in excelData.iterrows():
        if 'NH' in row['anisotropy-average']:
            tempdf = pd.DataFrame([row])
            # print(tempdf.to_string())
            NHList = pd.concat([NHList, tempdf], ignore_index=True)

    for index, row in bPointData.iterrows():
        if 'NH' in row['anisotropy-average']:
            tempdf = pd.DataFrame([row])
            # print(tempdf.to_string())
            NHListBoilingPoint = pd.concat([NHList, tempdf], ignore_index=True)


    excelData = excelData[excelData['anisotropy-average'] != "NH"]
    bPointData = bPointData[bPointData['anisotropy-average'] != "NH"]


    # print(bPointData.to_string())

    # separating input list by solvent
    columnNamesNew = excelData.columns
    nPentaneList = pd.DataFrame(columns=columnNamesNew)
    cycloPentaneList = pd.DataFrame(columns=columnNamesNew)
    nHexaneList = pd.DataFrame(columns=columnNamesNew)
    cycloHexaneList = pd.DataFrame(columns=columnNamesNew)
    nHeptaneList = pd.DataFrame(columns=columnNamesNew)

    for index, row in excelData.iterrows():
        if row['solvent'] == 'n-Pentane':
            tempdf = pd.DataFrame([row])
            nPentaneList = pd.concat([nPentaneList, tempdf], ignore_index=True)
        if row['solvent'] == 'cyclopentane':
            tempdf = pd.DataFrame([row])
            cycloPentaneList = pd.concat([cycloPentaneList, tempdf], ignore_index=True)
        if row['solvent'] == 'n-hexane':
            tempdf = pd.DataFrame([row])
            nHexaneList = pd.concat([nHexaneList, tempdf], ignore_index=True)
        if row['solvent'] == 'cyclohexane':
            tempdf = pd.DataFrame([row])
            cycloHexaneList = pd.concat([cycloHexaneList, tempdf], ignore_index=True)
        if row['solvent'] == 'n-heptane':
            tempdf = pd.DataFrame([row])
            nHeptaneList = pd.concat([nHeptaneList, tempdf], ignore_index=True)

    # drops nonnumeric variables
    # probably a better way to do this to be honest

    excelData.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    bPointData.drop(["polymerization-type"], axis=1, inplace=True)

    nPentaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    cycloPentaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    nHexaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    cycloHexaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    nHeptaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)



    #print(excelData.to_string())

    # cast all to float since output columns that had strings are type Object
    excelData = excelData.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    bPointData = bPointData.astype({'anisotropy-average': float, 'modulus': float, 'volume-fraction': float})
    nPentaneList = nPentaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    cycloPentaneList = cycloPentaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    nHexaneList = nHexaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    cycloHexaneList = cycloHexaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    nHeptaneList = nHeptaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})

    # some modulus values are much smaller than the others
    # this modifies the data to get rid of them for testing
    bPointData = bPointData[(bPointData['modulus']) >= 1]
    # print(bPointData.to_string())

    # second argument is the output to generate a predicted vs actual graph for
    # third argument is solvent name, doesn't need to be exact it's only used for titling
    # if using bPointData, it uses all solvents
    forest_regression(bPointData, 'modulus', 'all solvents overall')