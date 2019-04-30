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
boston_chas = boston.data[:,3]

# (a) fit OLS model to medv_i = B_0 + B_1*chas_i + B2*nox + e_i
features = np.column_stack([boston_chas, boston_nox])
features = sm.add_constant(features)
model = sm.OLS(boston_price,features).fit()
print(model.summary())

