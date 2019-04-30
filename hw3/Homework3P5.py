# # Numpy Practice

import numpy as np
from urllib.request import urlopen
import math

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Below are the names corresponding to each column, e.g. 2nd column has values of sepalwidth.
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# TODO 1: Import data from the above url into a numpy ndarray named 'iris'.
tuple_list = []

f = urlopen(url)
data = f.read().decode("utf-8")
for line in data.splitlines():
    split_line = line.split(',')
    tuple_list.append(tuple(split_line))

iris = np.array(tuple_list)

# remove last erroneous line
iris = iris[:-1]

# TODO 2: Print out the data. Observe the different kinds of species of irises.
print(iris)

# Now let's separate the data into 2 numpy arrays. 
# The first array (iris_1d) will contain the species name (5th column).
# The second array (irid_2d) will contain the other 4 columns. Convert this into a 'dtype = float' array.
# TODO 3: Create iris_1d and iris_2d.
iris_1d = np.zeros(len(iris), dtype=object)
iris_2d = np.zeros((len(iris), 4), dtype=float)

for i in range(0, len(iris)):
    el0, el1, el2, el3, el4 = iris[i]
    iris_1d[i] = el4
    iris_2d[i,0] = el0
    iris_2d[i,1] = el1
    iris_2d[i,2] = el2
    iris_2d[i,3] = el3

# ### Wrangling

# Now that you have imported data, you must check to see if the data is fit for analysis. 
# First, let's modify the data a bit to put in some NaN(not a number) values in it
# TODO 4: Randomly convert 20 entries in 'iris_2d' into NaNs.

for i in range(1, 20):
    # randomly select row index
    r_idx = np.random.uniform(0,len(iris_2d)-1)
    r_idx = round(r_idx)
    
    # randomly select col index
    c_idx = np.random.uniform(0,3)
    c_idx = round(c_idx)

    iris_2d[r_idx,c_idx] = np.nan

# There are several ways to deal with corrupted data (NaNs).
# One way to do this is to replace NaNs with the average of the other values in that column.
# TODO 5: Replace the NaN entries in the data with the average values of each column.

# take average of each column
col_mean = np.nanmean(iris_2d, axis=0)

# find where the nans are
idxs = np.where(np.isnan(iris_2d))

# replace nan with mean
iris_2d[idxs] = np.take(col_mean, idxs[1])

# TODO 6: Write a function to check if there are NaN values in the data. You may use built-in functions as necessary.
print(np.isnan(iris_2d).any())

# ### Filtering

# TODO 7: Filter the rows of 'iris_2d' that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0.
filtered_iris = []
for i in range(0,len(iris_2d)):
    if iris_2d[i,0] < 5 and iris_2d[i,2] > 1.5:
        # not sure what to do with the filtered rows  ... I stored them in a list
        filtered_iris.append(iris_2d[i,:])
        
print(filtered_iris)

# Create a deep copy of 'iris_2d'.
iris_2d_c = iris_2d.copy()

# TODO 8: Convert the column 'petallength' (3rd col) to a string according to the following rules in the copied array.
# - <3 --> 'small'
# - 3-5 --> 'medium'
# - >=5 --> 'large'


# I was not sure if the strings should replace the entries in the numpy array or to create a new array
# loop over 3rd column
string_list = []
for i in range(0, len(iris_2d_c)):
    if iris_2d[i,2] < 3:
        string_list.append("small")
    elif iris_2d[i,2] >= 3 and iris_2d[i,2] < 5:
        string_list.append("medium")
    else:
        string_list.append("large")

print(string_list)
        
# ### Statisticsl

# You may use scipy.stats package as you desire, although you can get through all the assignments without it
import scipy.stats as st

# Please re-import the iris dataset (the same way as TODO 1)
# TODO 9: Compute the means of the sepalwidth of the different species of irises and rank the species from highest to lowest 
tuple_list = []

f = urlopen(url)
data = f.read().decode("utf-8")
for line in data.splitlines():
    split_line = line.split(',')
    tuple_list.append(tuple(split_line))

iris = np.array(tuple_list)

# remove last erroneous line
iris = iris[:-1]
iris_1d = np.zeros(len(iris), dtype=object)
iris_2d = np.zeros((len(iris), 4), dtype=float)

for i in range(0, len(iris)):
    el0, el1, el2, el3, el4 = iris[i]
    iris_1d[i] = el4
    iris_2d[i,0] = el0
    iris_2d[i,1] = el1
    iris_2d[i,2] = el2
    iris_2d[i,3] = el3

# find idices of corresponding species
species = np.unique(iris_1d)
means = np.zeros(len(species))
for i in range(0,len(species)):
    idxs = np.where(iris_1d[:] == species[i])
    means[i] = np.mean(iris_2d[idxs,1])

# reverse sort or highest to lowest means
ordered_means = np.sort(means)[::-1]

# Hypothesis testing
# TODO 10: Compute the p-value for the hypothesis test:
# H_0 = mean(sepallength of Iris-setosa) < 5.0
# H_a = mean(sepallength of Iris-setosa) >= 5.0
# and determine whether to accept or reject H_0 with significance level alpha = 0.05
Ho_mu = 5
alpha = .05

# first compute x_bar or sample mean of sepallength of Iris-setosa, which is column 0
idxs = np.where(iris_1d[:] == "Iris-setosa")
n = len(idxs[0])
x_bar = np.mean(iris_2d[idxs,0])
var = np.var(iris_2d[idxs,0])

# use t test because we don't know population std dev, and student t distribtuion converges to normal with n -> inf
t = (x_bar - Ho_mu)/(math.sqrt(var/n))

# calculate p-value
p_value = 1 - st.t.cdf(t,n-1)
print(p_value)

# determine threshold at significance level alpha = 0.05
threshold = st.t.ppf(1-alpha, n-1)

if t > threshold:
    print("reject Ho")
else:
    print("accept Ho")

# ### Plotting

from matplotlib import pyplot as plt

# TODO 11: Construct a plot that enables us to compare the 4 different measurements (Columns 1~4) of each species.
# You are free to choose whichever plot that will fit this goal. The grading criteria would be 
# 1) accurateness of the plots
# 2) choice of an appropriate type of plot
# 3) labels and scaling of the axis

# 16 subplots where the ith row axis and jth column correspond to a measurement
# extract indices
setosa_idx = np.where(iris_1d[:] == "Iris-setosa")
versicolor_idx = np.where(iris_1d[:] == "Iris-versicolor")
virginica_idx = np.where(iris_1d[:] == "Iris-virginica")

# create subplot
fig, axes = plt.subplots(4 ,4, sharex = 'col', sharey = 'row')
for i in range(0,4):
    for j in range(0,4):
        axes[i,j].scatter(iris_2d[setosa_idx, j], iris_2d[setosa_idx, i], c = 'b', label = 'setosa')
        axes[i,j].scatter(iris_2d[versicolor_idx, j], iris_2d[versicolor_idx, i], c = 'r', label = 'versicolor')
        axes[i,j].scatter(iris_2d[virginica_idx, j], iris_2d[virginica_idx, i], c = 'g', label = 'virginica')

# label appropriately
axes[0,0].legend()
fig.suptitle('Comparing measurements in the Iris dataset')
axes[0,0].set_title('sepallength')
axes[0,0].set_ylabel('sepallength')
axes[0,1].set_title('sepalwidth')
axes[1,0].set_ylabel('sepalwidth')
axes[0,2].set_title('petallength')
axes[2,0].set_ylabel('petallength')
axes[0,3].set_title('petalwidth')
axes[3,0].set_ylabel('petalwidth')
#plt.savefig('hw3_5.pdf', format='pdf')
plt.show()
