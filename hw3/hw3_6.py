## Stastics for Data Science
# HW3 #6
# Tyler Olivieri

from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import scipy.stats as st

####################################################################
# (a) for experiment i, state your null and alternative hypothesis
#
# Since we will accept the change if the experiment increases performance,
# we don't care about a two-sided test
# We will be confident the experiment increases performance if the null hypothesis is
# rejected
# Ho: average number of minutes in experiment i = 49.75
# Ha: average number of minutes in experiment i > 49.75
#
# p-value = Pr{observation contradicts null hypothesis more} = Pr{Z >= z | Ho true}
# where t is t test (current obeservation)
# p-value = 1 - Pr{Z <= z | Ho true} = 1 - std_gaussian_cdf(z)
# we can use the z-test/ standard gaussian because we have large n for each experiment
#

####################################################################
# (b) Find the experiments that have a p-value less than alpha = .05
#
alpha = .05
time_mean = 49.75
action_mean = 31.3
dir_str = "/home/snazyman/stat_ds/data_science/hw3/experiments"
directory = os.fsencode(dir_str)
null_rej_time = []

# loop over experiments directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # open experiment*.csv file
    if filename.endswith(".csv"):
        with open(dir_str + '/' + filename) as csvfile:
            data_time_list = []
            data_reader = csv.reader(csvfile)

            # read in the data from the experiment*.csv
            for row in data_reader:
                data_time_list.append(row[1])

            # remove first entry - it is not data but descriptor
            del data_time_list[0]

            data_time_array = np.array(data_time_list,dtype='float')

            # compute z-score
            # ddof=1 uses 1/n-1 instead of 1/n in variance calculation
            z_time = (np.mean(data_time_array) - time_mean) / math.sqrt(np.var(data_time_array)/data_time_array.size)
            
            # compute p value
            p_value_time = 1 - st.norm.cdf(z_time)

            # compare to alpha
            # save the experiments that reject Ho
            if p_value_time < alpha:
                null_rej_time.append(filename)

# compute the probability that you find a significant result due to chance under an assumption that the hypothesis are independent
# Pr{finding at least one signifcant result} = 1 - Pr{finding no signicant result}
# Pr{finding no signifcant result} = Pr{first test gives no significant result and second test gives no significant result} = Pr{first test gives no significant resut}Pr{second test gives no signicant result}...Pr{nth test gives no significant result} = (1- alpha)^n
# Pr{finding at least one signifcant result} = 1 - (1 - alpha)^n
num_hypothesis = 70
p_sig_chance = 1 - (1-alpha)**num_hypothesis

print(null_rej_time)
print(p_sig_chance)

# This poses a problem because the probability of finding a significant result is now greater than alpha with 2 tests, we now have a probability of finding a significant result that is greater than our desired significance level

#########################################################################
# (c)
# Apply the bonferroni correction, set cut-off significance to alpha/m
# find significant experiments

bonferroni_alpha = .05/num_hypothesis
bonferroni_null_rej_time = []

# loop over experiments directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # open experiment*.csv file
    if filename.endswith(".csv"):
        with open(dir_str + '/' + filename) as csvfile:
            data_time_list = []
            data_reader = csv.reader(csvfile)

            # read in the data from the experiment*.csv
            for row in data_reader:
                data_time_list.append(row[1])

            # remove first entry - it is not data but descriptor
            del data_time_list[0]

            data_time_array = np.array(data_time_list,dtype='float')

            # compute z-score
            # ddof=1 uses 1/n-1 instead of 1/n in variance calculation
            z_time = (np.mean(data_time_array) - time_mean) / math.sqrt(np.var(data_time_array)/data_time_array.size)
            
            # compute p value
            p_value_time = 1 - st.norm.cdf(z_time)

            # compare to alpha
            # save the experiments that reject Ho
            if p_value_time < bonferroni_alpha:
                bonferroni_null_rej_time.append(filename)

p_sig_chance = 1 - (1-bonferroni_alpha)**num_hypothesis
print(bonferroni_null_rej_time)
print(p_sig_chance)

#####################################################################################################################
# (d)
# Implement Holm-Bonferroni procedure
# find significant experiments

hb_null_rej_time = []

exp_idx = 0
p_value_array = np.zeros(num_hypothesis)
exp_array = []

# loop over experiments directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # open experiment*.csv file
    if filename.endswith(".csv"):
        with open(dir_str + '/' + filename) as csvfile:
            data_time_list = []

            data_reader = csv.reader(csvfile)

            # read in the data from the experiment*.csv
            for row in data_reader:
                data_time_list.append(row[1])


            # remove first entry - it is not data but descriptor
            del data_time_list[0]

            data_time_array = np.array(data_time_list,dtype='float')

            # compute z-score
            # ddof=1 uses 1/n-1 instead of 1/n in variance calculation
            z_time = (np.mean(data_time_array) - time_mean) / math.sqrt(np.var(data_time_array)/data_time_array.size)
            
            # compute p value
            p_value_time = 1 - st.norm.cdf(z_time)
            p_value_array[exp_idx] = p_value_time
            exp_array.append(filename)
            
            exp_idx = exp_idx + 1

# order p values
p_value_ordered_idx = p_value_array.argsort()

# apply Holm-Bonferroni procedure to check if we should reject Ho[k]
k = 0
while (p_value_array[p_value_ordered_idx[k]] <= (alpha/(num_hypothesis+1-k))):

    # reject null hypothesis k
    hb_null_rej_time.append(exp_array[p_value_ordered_idx[k]])

    k = k + 1

    # if k = num_hypothesis we rejected all null hypothesis and can break out of loop
    # removes accessing (p_value_array[p_value_ordered_idx[num_hypothesis]] which will never exist
    if k == num_hypothesis:
        break
            
print(hb_null_rej_time)
