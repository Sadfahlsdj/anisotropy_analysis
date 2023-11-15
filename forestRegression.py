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

def forest_regression(df, outputName):
    # xVars is concentration & curing time; yVars is anisotropy, volume fraction, and modulus
    xVars = df.drop(['anisotropy-average', 'volume-fraction', 'modulus', 'sample'], axis=1)
    yVars = df.drop(['concentration', 'curing-time', 'sample'], axis=1)
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
    plt.scatter(yValidGraph, yPredGraph, color="red", label="Comparison between Actual and Predicted Data")
    plt.legend()
    plt.grid()
    plt.title(f"Actual vs Predicted Values for {outputName}")
    plt.xlabel("Predicted data")
    plt.ylabel("Actual data")
    plt.show()

    # r^2, multioutput='raw_values' prints each separately which is what i want
    r2 = metrics.r2_score(yValid, yPred, multioutput='raw_values')
    print(f"R squared score for this data set is {r2} for anisotropy, volume fraction, and modulus respectively")


if __name__ == "__main__":
    excelData = pd.read_csv("excelData.csv", encoding='latin-1')
    # print(excelData.shape)

    # drops nonnumeric variables and any lines with NH or SP
    excelData = excelData[excelData['anisotropy-average'] != "NH"]
    excelData.drop(["solvent", "polymerization-type"], axis=1, inplace=True)

    #print(excelData.to_string())

    # all this below is because the regression accepts only ints, not float
    # (float is continuous, which poses an issue apparently)

    # cast to float so i can multiply
    excelData = excelData.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})

    # multiply select columns by 1000
    """excelData['concentration'] = excelData['concentration'] * 1000
    excelData['curing-time'] = excelData['curing-time'] * 1000
    excelData['anisotropy-average'] = excelData['anisotropy-average'] * 1000
    excelData['modulus'] = excelData['modulus'] * 1000
    excelData['volume-fraction'] = excelData['volume-fraction'] * 1000

    # int cast to remove trailing 0s
    excelData = excelData.astype({"anisotropy-average": int, "modulus": int, "volume-fraction": int})
    """
    # print(excelData.to_string())


    excelData.to_csv("chemDataParsed", encoding='utf-8', index=False)
    # exports parsed csv to a new csv, which is then read from
    # encoding is (probably) needed to make it work, index means that row index isn't included (good)

    parsedData = pd.read_csv("chemDataParsed", encoding='latin-1')
    # forestRegress(parsedData, "anisotropy-average")

    # second argument is the output to generate a predicted vs actual graph for
    # probably need to separate by solvent to get better result
    forest_regression(parsedData, 'modulus')