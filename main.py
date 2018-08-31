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


# simulate initial city of N homes without DSM
experiment = {}
experiment['N'] = 100 # number of homes to simulate
experiment['filename'] = 'data/home_all.csv'
experiment['alpha'] = 100 # max load
experiment['beta'] = 100 # max price
experiment['epsilon_D'] = -0.6 # elasticity of load
experiment['epsilon_P'] = -0.4 # elasticity of price

# % of load of each home participating in DSM
experiment['kappa'] = np.random.uniform(low=0.0, high=1.0, size=experiment['N'])


experiment = dsm.load(experiment)
experiment = dsm.simulate_city(experiment)
experiment = dsm.simulate_load_price(experiment)

P = experiment['P']
L = experiment['L']
total_L = experiment['total_L']

plt.figure(1)
plt.plot(P)
plt.figure(2)
plt.plot(total_L)







# plot data
# plt.plot(total.loc[(df.index.day == 1)]) # plot 1 day
# plt.plot(total_load) # plot whole month
# plt.ylabel('Total Load')
# plt.xlabel('Time')

# print runtime to console
print('Runtime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()