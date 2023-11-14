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


if __name__ == "__main__":
    excelData = pd.read_csv("excelData.csv", encoding='latin-1')
    # print(excelData.shape)

    excelData = excelData[excelData['anisotropy-average'] != "NH"]
    forestRegress(excelData, "anisotropy-average")