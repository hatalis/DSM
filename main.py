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
experiment['alpha'] = 40_000 # max load tolerated by SO
experiment['beta'] = 1_000 # price for max load
experiment['omega'] = 100 # max load allowed by DSM
experiment['epsilon_D'] = -0.6 # elasticity of load
experiment['epsilon_P'] = -0.4 # elasticity of price
experiment['T'] = 24*7 # total time to simulate

# % of load of each home participating in DSM
experiment['kappa'] = np.random.uniform(low=0.0, high=1.0, size=experiment['N'])

# run simulation
experiment = dsm.load(experiment) # load template homes
experiment = dsm.simulate_city(experiment) # simulate phi
experiment = dsm.simulate_load_price(experiment)
dsm.output_results(experiment)

# print runtime to console
print('Runtime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()