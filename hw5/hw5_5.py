import numpy as np
import csv
from sklearn.model_selection import KFold
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_name = "/home/snazyman/stat_ds/data_science/hw5/poly_data_space.csv"

with open(file_name) as csvfile:
    i = 0
    data_reader = csv.reader(csvfile)
    predictor = np.empty([200,1], dtype=float)
    target = np.empty([200,1], dtype=float)

    for row in data_reader:
        predictor[i] = float(row[0])
        target[i] = float(row[1])
        i = i + 1

    # pick k here ~ n_splits
    k = 5
    kf = KFold(n_splits = k)

    # pick a set of polynomial models
    degree_list = [1, 2, 3, 4, 5]

    CV_error = []
    
    # loop over them
    for poly_model in degree_list:
        i = 0
        
        # construct model
        # constant (for intercept)
        features = np.ones([200,1], dtype=float)

        MSE = np.zeros([k,1], dtype=float)
        
        # append polynomial powers of the predictor
        for power in range(1,poly_model+1):
            feature_power = np.power(predictor, power)
            features = np.column_stack([features, feature_power])

        # train and evaluate model
        for train_index, test_index in kf.split(predictor):

            # get current folds index(s)
            features_train, features_test = features[train_index], features[test_index]
            target_train, target_test = target[train_index], target[test_index]

            
            # train polynomial model
            model = sm.OLS(target_train,features_train).fit()

            # evaluate polynomial model on test fold
            target_predicted = model.predict(features_test)
            
            # compute MSE
            MSE[i] = mean_squared_error(target_test, target_predicted)
            i = i + 1

        # average MSE over k folds to get k fold cross validation error
        CV_error.append(MSE.mean())

    # plot results
    print(CV_error)
    plt.plot(degree_list, CV_error)
    plt.title(f"{k}-fold cross validation error (MSE) vs polynomial degree")
    plt.show()

    # (b) choose best model (model with lowest {k}-fold cross validation error (MSE))
    # run model on entire dataset. Report the coefficients. Plot the data and fitted polynomial
    idx = np.argmin(CV_error)
    degree = degree_list[idx]

    # recreate best model with best degree
    features = np.ones([200,1], dtype=float)
    for power in range(1,degree+1):
        feature_power = np.power(predictor, power)
        features = np.column_stack([features, feature_power])

    # train model
    model_final = sm.OLS(target,features).fit()
    print(model_final.summary())

    # print data vs predictions
    fig, ax = plt.subplots(figsize=(12, 8))
    fig = sm.graphics.plot_fit(model_final,1, ax=ax)

    ax.legend(loc='best');
    plt.title("Data and OLS polynomial model predictions")
    plt.show()
