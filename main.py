'''
This script loads in a csv file of template home data.prin
Using those homes, a city of homes is simulated using the block bootstrap method.
This simulation is used as the phi distribution.
Then load and price data is simulated for DSM.

By: Kostas Hatalis
'''

import lehighdsm as dsm
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()
# np.random.seed(1)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#--------------------------------------------------------------------------
# simulation parameters
experiment = {}
experiment['filename'] = 'data/home_all.csv'
window = 0 #24*3+1 # training size window for SARIMA in simulation
experiment['N_train'] = 24*14
experiment['N_test'] = 24*2
experiment['lags'] = 24 # num of lags for feature eng.
experiment['scaling'] = None
experiment['N'] = 200 # number of homes to simulate
experiment['T'] = experiment['N_train'] + experiment['N_test'] + window # total time to simulate
experiment['kappa'] = 0.1 #np.random.uniform(low=0.0, high=1.0, size=experiment['N'])

# Parameters for ISO to determine electricity prices
experiment['alpha'] = 800 # max load tolerated by SO
experiment['beta'] = 10_000 # price for max load in cents
experiment['L_target'] = 100 # target load
experiment['L_hat_period'] = 300
experiment['epsilon_D'] = -1 # elasticity of load
experiment['epsilon_P'] = None # if None then elasticity is determined by ISO
experiment['method'] = 0 # 0 = pers, 1 = SARIMA, 2 = SARIMAX
experiment['window'] = window
#--------------------------------------------------------------------------
# QARNET parameters
experiment['optimizer'] = 'Adam' # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
experiment['activation'] = 'relu' # relu, sigmoid, tanh, softplus, elu, softsign, sigmoid, linear
experiment['smooth_loss'] = 1 # 0 = pinball, 1 = smooth pinball loss
experiment['maxIter'] = 2000
experiment['batch_size'] = 200
experiment['hidden_dims'] = [60] # number of nodes per hidden layer
experiment['smoothing_alpha'] = 0.01 # smoothing rate
experiment['Lambda'] = 0.5 # regularization term
experiment['n_tau'] = 2
experiment['tau'] = np.array([0.005, 0.995])
experiment['kappa'] = 0 # penalty term
experiment['margin'] = 0 # penalty margin
experiment['print_cost'] = 0 # 1 = plot cost
experiment['plot_results'] = 1 # 1 = plot results
#--------------------------------------------------------------------------

# run simulation
experiment = dsm.functions.load(experiment) # load template homes
experiment = dsm.functions.simulate_city(experiment) # simulate phi
experiment = dsm.functions.simulate_load_price(experiment)
dsm.functions.output_results(experiment)

# get SARIMA forecast
experiment = dsm.detection.SARIMA_forecast(experiment)

# get featurs and split data
experiment = dsm.features.lagged_load(experiment)
experiment = dsm.features.lagged_price(experiment)
experiment = dsm.detection.split_data(experiment, scaling= None)

# run detection
experiment = dsm.detection.QARNET_detector(experiment,test_method=0) # 0 = QARNET, 1 = QAR

# evaliation of detection results
experiment = dsm.evaluation.evaluate_PI_results(experiment)
dsm.evaluation.output_PI_results(experiment)

# print runtime to console
print('Runtime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()