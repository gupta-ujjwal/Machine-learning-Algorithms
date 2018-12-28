
#importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import math

#importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
N = 10000
d = 10
ads_selected = []
num_of_selection = [0] * d
sums_of_reward = [0] * d
total = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if num_of_selection[i] > 0 :
            average_reward = sums_of_reward[i]/num_of_selection[i]
            delta = math.sqrt(3/2*math.log(n+1)/num_of_selection[i])
            upper_bound = average_reward + delta
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    num_of_selection[ad] = num_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total = total + reward

#plotting the result
plt.hist(ads_selected)
plt.title('Histogram of each add')
plt.show()