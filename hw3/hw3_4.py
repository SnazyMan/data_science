## Stastics for Data Science
# HW3 #4
# Tyler Olivieri

from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import numpy as np

# For X1, ... , Xn iid ; n = 15 drawn N(mu, sigma^2 = 4)
##############################################################################################################################
# (a)
# plot the power of the test Ho: mu = 0 versus Ha: mu != 0

n = 15
var = 4
alpha = .05
mu_0 = 0
threshold = norm.ppf(1 - alpha/2)
scaling_factor = math.sqrt(var/n)
mu_vals = np.linspace(-5,5,100)
power = []

# calculate power
for mu in mu_vals:
    power.append( ( (1 - norm.cdf(threshold - ((mu - mu_0)/scaling_factor))) + norm.cdf(-threshold - ((mu - mu_0)/scaling_factor))) )


# plot results
plt.title("Power of 2-sided hypothesis z test as a funciton of u: Ho: mu = 0")
plt.xlabel("mu")
plt.ylabel("Power")
plt.plot(mu_vals,power)
plt.show()

#####################################################################################################################################
# (b)
# now vary alpha and fix mu = 3
#

alpha_vals = np.linspace(0,.2,100)
mu = 3
power.clear()

# calculate power
for alpha_iter in alpha_vals:
    threshold = norm.ppf(1 - alpha_iter/2)
    power.append( (1 - norm.cdf(threshold - ((mu - mu_0)/scaling_factor))) + norm.cdf(-threshold - ((mu - mu_0)/scaling_factor)) )

# plot results
plt.title("Power of 2-sided hypothesis z test as a funciton of alpha with mu = 3: Ho: mu = 0")
plt.xlabel("alpha")
plt.ylabel("Power")
plt.semilogy(alpha_vals,power)
plt.show()

########################################################################################################################################
# (c)
# fix alpha = .05 and plot the power of the test as n varies
#


threshold = norm.ppf(1 - alpha/2)
power.clear()

# calculate power
for n_iter in range(1,25):
    scaling_factor_n = math.sqrt(var/n_iter)
    power.append( (1 - norm.cdf(threshold - ((mu - mu_0)/scaling_factor_n))) + norm.cdf(-threshold - ((mu - mu_0)/scaling_factor_n)) )

# plot results
plt.title("Power of 2-sided hypothesis z test as a funciton of alpha with mu = 3: Ho: mu = 0")
plt.xlabel("n")
plt.ylabel("Power")
plt.plot(range(1,25),power)
plt.show()
