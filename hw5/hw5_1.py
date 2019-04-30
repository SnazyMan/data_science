
from sklearn.datasets import load_boston
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# load the boston house price dataset fro sklearn
boston = load_boston()
boston_age_org = boston.data[:,6]
boston_price = boston.target

# (a)
# find how the variable age relates to housing prices in Boston
# assume model medv_i = B_0 + B_1*age_i + e_i where e_i ~ N(0,sigma^2)

# add intercept for B_0
boston_age = sm.add_constant(boston_age_org)
model = sm.OLS(boston_price,boston_age).fit()

print(model.summary())

# print data vs predictions
#fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(boston_age_org, boston_price, 'o', label="data")
#ax.plot(boston_age_org, model.fittedvalues, 'r--.', label="OLS")
#ax.legend(loc='best');
#plt.show()

# (d) Recall that the residuals are the difference between the true response and the predicted response of your model, plot residuuals against the predictor. How do you expect this to look if the model is true?

plt.scatter(boston_age_org, model.resid)
plt.title("Residuals for every prediction of price from age")
plt.show()

# (e) generate probability plot
stats.probplot(model.resid, plot= plt)
plt.title("Model Residuals Probability Plot")
plt.show()

print(stats.kstest(model.resid, 'norm'))
