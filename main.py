'''
This script loads in a csv file of template home data.
Using those homes, a city of homes is simulated using the block bootstrap method.

By: Kostas Hatalis
'''

import lehighdsm.functions as dsm
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
np.random.seed(1)

# load data in
df = dsm.load('data/home_all.csv')


N = 100 # number of homes to simulate
homes = ['A','B','C','D','E','F','G'] # list of template homes
city = df['A'] # we start simulating a city with 1 home


# pick N random homes from list
templates = np.random.choice(homes, size=N, replace=True)


# simulate 1 month of data for each home, save to city dataframe
for i in range(N):
    city = dsm.create_homes(df,templates[i],i).join(city, how='inner')


# sum the data together to get total load
total = city.sum(axis=1)
total.to_csv('city.csv', sep=',')


# plot data
# plt.plot(total.loc[(df.index.day == 1)]) # plot 1 day
plt.plot(total) # plot whole month
plt.ylabel('Total Load')
plt.xlabel('Time')

# print runtime to console
print('Runtime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()