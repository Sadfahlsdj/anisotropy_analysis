# forestRegression.py (the functional python file) has a lot of previous code
# that is currently unused but which I don't want to completely delete
# so I will stash it here


# starter code for a roc curve model (suited for binary or categorical number output)
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

# code to separate by solvent
"""
# separating input list by solvent
# only used for non boiling point data
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
            nHeptaneList = pd.concat([nHeptaneList, tempdf], ignore_index=True)"""

# code to drop nonnumerical inputs from solvent lists
"""
nPentaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    cycloPentaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    nHexaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    cycloHexaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    nHeptaneList.drop(["solvent", "polymerization-type"], axis=1, inplace=True)
    #print(excelData.to_string())"""

# code to cast all numerical columns to float for solvent lists
"""
nPentaneList = nPentaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    cycloPentaneList = cycloPentaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    nHexaneList = nHexaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    cycloHexaneList = cycloHexaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
    nHeptaneList = nHeptaneList.astype({"anisotropy-average": float, "modulus": float, "volume-fraction": float})
"""