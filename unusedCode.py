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