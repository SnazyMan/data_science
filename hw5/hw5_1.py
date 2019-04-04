
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
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(boston_age_org, boston_price, 'o', label="data")
ax.plot(boston_age_org, model.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best');
plt.show()

# Summary shows const = 30.9787; x1 = -.12, x1 is just the slope of the line in the linear model. It explains the rate of change of the price with age. or the average effect on price of one unit increase in age. The hypothesis test corresponding to B_1 has the null hypothesis that B_1 = 0 (no relationship between age and price). With a p-value of 0, we reject this hypothesis with any significance, meaning there is certainly a linear relationship.

# (b) Interpret the R^2 value for this regression in a sentence or two
# The R^2 value is .363, meaning that the model did not approximate the real data very well. There is a lot of variance in the data not explained in the model. I believe this also means there was high residuals.


# (c) How many parameters does this model have? What is the residual degrees of freedom?
# Two parameters B_0 and B_1
# Residual degrees of freedom is n-2 (2 comes from 2 parameters). It is the dimension of the space of the residuals. Since the estimated residuals must sum to zero (e1 + ... + en = 0) and x1e1 + ... + x_n * e_n = 0, then we have n-2 degrees of freedom. The residuals form a space of dimension n-2.

# (d) Recall that the residuals are the difference between the true response and the predicted response of your model, plot residuuals against the predictor. How do you expect this to look if the model is true?

# I expect the residuals to be without a strong pattern for a linear (true) model. A pattern in the residuals would indicate some non-linearity in the data. I do not see much of a patern, but the residuals have high variance.

#fig = plt.figure(figsize=(12,8))
#fig = sm.graphics.plot_regress_exog(model, 1, fig=fig)
plt.scatter(boston_age_org, model.resid)
plt.title("Residuals for every prediction of price from age")
plt.show()

# (e) generate probability plot
# If the model is true, I expect the probability plot to be directly on the linear line. However, we see that is not quite the case here
stats.probplot(model.resid, plot= plt)
plt.title("Model Residuals Probability Plot")
plt.show()


# This is a formal test to test if the residuals are normally generated. The pvalue is very small, meaning we should reject the null hypothesis (the null hypothesis is that the distributions are identical
# it gives a small p-value telling us to reject the null hypothesis that the distributions are equal
print(stats.kstest(model.resid, 'norm'))


# (f) Is this model true?
# Yes and no. We have a significantly significant predictor for age and price, however, we have a low r-squared which suggests that we are not explaining the variance in the model very well. Also, the probability plot and kstest suggests that the model does not have normally distributed residuals.

#I think that the low r-squared could be OK in this scenario, meaning that the housing price has a lot of variance or complexity, and a linear model can't capture this. I would use this predictor over nothing, but would seek out a better predictor.
