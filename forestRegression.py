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
from sklearn import preprocessing

import scipy
import inspect
#data columns in order:
# 0: sample (sample number, no numerical meaning)
# 1: solvent, 2: concentration, 3: curing time (input variables)
# 4: anisotropy average, 5: volume fraction, 6: modulus (output variables)
# 7: polymerization type (F is good, SP has its own list)
# index 7 can be F and outputs are still NH, this has its own list too


def forest_regression(df, outputName, solventName, normalized=False, predictionList=None):
    # if predictionList exists, the entire input dataframe will be used to train
    # and prediction will be ran on predictionList
    # else, it will proceed as normal

    if predictionList is not None:
        testRatio = 0.01
    else:
        testRatio = 0.3 #determining how much of the input list is used to train

    # normalized is false by default, if it's true then the data set is modified to be normalized
    # and the x and y data sets will draw from the normalized set
    # rest of analysis happens as normal once the sets to be used are acquired
    if normalized:
        dfNormalized = preprocessing.normalize(df, axis=0) # axis=0 means it's done by column
        dfscaled = pd.DataFrame(dfNormalized, columns=df.columns)
        xVars = dfscaled.drop(['anisotropy-average', 'volume-fraction', 'modulus', 'sample'], axis=1)
        yVars = dfscaled.drop(['boiling-point', 'concentration', 'curing-time', 'sample'], axis=1)
    else:
        # xVars is concentration & curing time; yVars is anisotropy, volume fraction, and modulus
        # use first yVars for data split by solvent, second one for boiling point data
        xVars = df.drop(['anisotropy-average', 'volume-fraction', 'modulus', 'sample'], axis=1)
        # yVars = df.drop(['concentration', 'curing-time', 'sample'], axis=1)
        yVars = df.drop(['boiling-point', 'concentration', 'curing-time', 'sample'], axis=1)
    # test_size being 0.3 means that 70% of the data goes into train, which is needed
    xTrain, xValid, yTrain, yValid = train_test_split(xVars, yVars, test_size=testRatio, random_state=42)

    # create regressor and train
    rg = RandomForestRegressor(n_estimators=100, random_state=42)
    rg.fit(xTrain, yTrain)
    if predictionList is not None:
        predictionX = predictionList.drop(['anisotropy-average', 'volume-fraction', 'modulus', 'sample'], axis=1)
        yPred = rg.predict(predictionX)
        # predicts on predictionList's inputs
    else:
        yPred = rg.predict(xValid)
    # change datatype to make it easier to use
    yPred = pd.DataFrame(yPred, columns=['anisotropy-average', 'volume-fraction', 'modulus'])

    # only generates graph and r2 if the same dataframe is used for test & train
    # if a different one is used, exports a csv with outputs
    if predictionList is not None:
        yOverall = predictionX.join(yPred) # merges input/output into one dataframe for easy visualization
        yOverall = yOverall.round(4) #rounds to 4 to avoid floating point imprecision
        yOverall.to_csv("no_holes_predicted_value.csv", encoding='utf-8', index=False)
        # exports to csv
    else:
        # what output is used for the graph
        yValidGraph = yValid[outputName].tolist()
        yPredGraph = yPred[outputName].tolist()

        # below is used for titles and the r2 print
        normalizedString = ""
        if normalized:
            normalizedString = "with normalization"

        plt.figure(figsize=(10, 10))
        plt.scatter(yValidGraph, yPredGraph, color="red", label=f"Comparison between Actual and Predicted Data in {solventName}")
        plt.legend()
        plt.grid()
        plt.title(f"Actual vs Predicted Values for {outputName} in {solventName} {normalizedString}")
        plt.xlabel("Predicted data")
        plt.ylabel("Actual data")
        plt.show()

        # r^2, multioutput='raw_values' prints r2 for each output separately
        r2 = metrics.r2_score(yValid, yPred, multioutput='raw_values')
        np.around(r2, 5)
        r2total = metrics.r2_score(yValid, yPred)
        np.around(r2total, 5)

        print(f"R squared score for this data set is {r2} for anisotropy, volume fraction, and modulus respectively in {solventName}")
        print(f"Overall R squared score for this data set is {r2total} {normalizedString}")

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
            NHList = pd.concat([NHList, tempdf[tempdf.columns[1:]]], ignore_index=True)

    for index, row in bPointData.iterrows():
        if 'NH' in row['anisotropy-average']:
            tempdf = pd.DataFrame([row])
            # print(tempdf.to_string())
            NHListBoilingPoint = pd.concat([NHListBoilingPoint, tempdf[tempdf.columns[1:]]], ignore_index=True)

    excelData = excelData[excelData['anisotropy-average'] != "NH"]
    bPointData = bPointData[bPointData['anisotropy-average'] != "NH"]

    # NH values done removing now

    # drops nonnumeric variables
    # probably a better way to do this to be honest
    excelData.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    bPointData.drop(["polymerization-type"], axis=1, inplace=True)
    NHListBoilingPoint.drop(['polymerization-type'], axis=1, inplace=True)

    # cast all to float since output columns that had strings are type Object
    excelData = excelData.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    bPointData = bPointData.astype({'anisotropy-average': float, 'modulus': float, 'volume-fraction': float})

    # some modulus values are much smaller than the others
    # this modifies the data to get rid of them for testing
    # comment/uncomment the next line for different analyses--it's optional more or less
    # bPointData = bPointData[(bPointData['modulus']) >= 1]

    # second argument is the output to generate a predicted vs actual graph for
    # it needs to exactly match an output column name
    # third argument is solvent name, doesn't need to be exact it's only used for titling
    # if using bPointData, it uses all solvents
    # normalized is whether to normalize or not, default is false;
    # testing says it'll greatly decrease accuracy on anisotropy but slightly improve the other 2 outputs
    # predictionList, if provided, will run predicted values on its inputs
    # providing this will not generate a graph or r2 values (they won't be relevant)
    forest_regression(bPointData, 'modulus', 'all solvents overall', predictionList=NHListBoilingPoint)

    # INDIVIDUAL SOLVENT LISTS ARE DEPRECATED
    # if you really want to use them, find code for separating them in unusedCode.py