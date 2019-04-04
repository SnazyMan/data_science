
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.stats.outliers_influence as st

# load the boston house price dataset from sklearn
boston = load_boston()
boston_price = boston.target
boston_nox = boston.data[:,4]
boston_rm = boston.data[:,5]
boston_dis = boston.data[:,7]
boston_lstat = boston.data[:,12]
print(boston.feature_names[4])
print(boston.feature_names[5])
print(boston.feature_names[7])
print(boston.feature_names[12])
# (a)
# Using perason correlation, compute and report the correlation matrix for our four predictor variables
# 
corr_matrix = np.corrcoef([boston_nox, boston_rm, boston_dis, boston_lstat])
print(corr_matrix)
#VIF = np.linalg.inv(corr_matrix)
#print(VIF.diagonal())

# (b)
# Compute variance inflation factor (VIF) for each of the predictor variables
# Multi-collinearity is a problem because coefficient estimates of multiple regression can change erraticaly in response to small changes in the model or data (quoting wikipedia). In perfect multicolinearity, the OLS estimator optimal solutions do not exist
features = np.column_stack([boston_nox, boston_rm, boston_dis, boston_lstat])
features = sm.add_constant(features)
vif = [st.variance_inflation_factor(features, i) for i in range(features.shape[1])]
print(vif)


# (c)
# Drop predictors with VIF values over 10
# run linear regression on remaining variables to predict the target

# no predictors had a VIF over 10

model = sm.OLS(boston_price,features).fit()
print(model.summary())

## (d) Compute F-statistic and its corresponding p-value in this model.
# This is done in model.summary()
# F-statistic: 241.6
# P-value 2.09e-115
# This means we can reject the null hypothesis since the p-value is almost 0.
# The null hypothesis is that the intercept only model (no predictor variables) is equal to the model with the predictor variables
# In other words, our model with the predictor variables gives a better fit than a model with just a y-intercept or at least one predictor coeficient, Bj is nonzero


