'''
This script loads in a csv file of template home data.
Using those homes, a city of homes is simulated using the block bootstrap method.
This simulation is used as the phi distribution.
Then load and price data is simulated for DSM.

By: Kostas Hatalis
'''

import lehighdsm.functions as dsm
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
np.random.seed(1)

# simulation parameters
experiment = {}
experiment['filename'] = 'data/home_all.csv'
experiment['N'] = 200 # number of homes to simulate
experiment['T'] = 24*7 # total time to simulate
experiment['kappa'] = 0 #np.random.uniform(low=0.0, high=1.0, size=experiment['N'])

# Parameters for SO to determine electricity prices
experiment['alpha'] = 800 # max load tolerated by SO
experiment['beta'] = 10_000 # price for max load in cents

experiment['L_target'] = 100 # target load

experiment['epsilon_D'] = -1 # elasticity of load

# run simulation
experiment = dsm.load(experiment) # load template homes
experiment = dsm.simulate_city(experiment) # simulate phi
experiment = dsm.simulate_load_price(experiment)
dsm.output_results(experiment)

# print runtime to console
print('Runtime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()