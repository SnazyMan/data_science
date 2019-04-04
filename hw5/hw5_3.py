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


# (b) What is the interpretation of B_0.
# It explains average price of houses not bordering the Charles River

# (c) What is the interpretation of B_1
# B_1 explains the average difference between houses that border the Charles river and those that don't

# (d) Why do we not include two dummy vars?
# Then there is a problem with multi-colinearity. If one variable is 0, the other is 1. There is a perfect relationship between them. Extremly correlated.
# The constant term is a vector of ones. Then by construction the two dummy variables will add to be a vector of 1s. When a variable is 0, the other is 1 and vice versa. Therefore, the intercept is a linear combination of the dummy variables, creating the co-linearity.
