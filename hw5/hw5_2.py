
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


# (b)
# Compute variance inflation factor (VIF) for each of the predictor variables
features = np.column_stack([boston_nox, boston_rm, boston_dis, boston_lstat])
features = sm.add_constant(features)
vif = [st.variance_inflation_factor(features, i) for i in range(features.shape[1])]
print(vif)


# (c)
# Drop predictors with VIF values over 10
# run linear regression on remaining variables to predict the target

model = sm.OLS(boston_price,features).fit()
print(model.summary())

