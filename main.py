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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)

#--------------------------------------------------------------------------
# simulation parameters
experiment = {}
experiment['filename'] = 'data/home_all.csv'
window = 0 #24*3+1 # training size window for SARIMA in simulation
experiment['N_train'] = 24*14
experiment['N_test'] = 24*2
experiment['lags'] = 48 # num of lags for feature eng.
experiment['scaling'] = None
experiment['N'] = 200 # number of homes to simulate
experiment['T'] = 24 #experiment['N_train'] + experiment['N_test'] + window # total time to simulate
experiment['kappa'] = 0.5 #np.random.uniform(low=0.0, high=1.0, size=experiment['N'])

# Parameters for ISO to determine electricity prices
# experiment['alpha'] = 800 # max load tolerated by SO
# experiment['beta'] = 10_000 # price for max load in cents
experiment['L_target'] = 200 # target load
experiment['L_hat_period'] = 300
experiment['epsilon_D'] = -1 # elasticity of load
experiment['epsilon_P'] = None # if None then elasticity is determined by ISO
experiment['method'] = 0 # 0 = pers, 1 = SARIMA, 2 = SARIMAX
experiment['window'] = window
#--------------------------------------------------------------------------
# QARNET parameters
# tau = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
#        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# tau = [0.025, 0.975]
# experiment['optimizer'] = 'Adam' # SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
# experiment['activation'] = 'relu' # relu, sigmoid, tanh, softplus, elu, softsign, sigmoid, linear
# experiment['smooth_loss'] = 1 # 0 = pinball, 1 = smooth pinball loss
# experiment['maxIter'] = 10_000
# experiment['batch_size'] = 200
# experiment['hidden_dims'] = [200] # number of nodes per hidden layer
# experiment['smoothing_alpha'] = 0.01 # smoothing rate
# experiment['Lambda'] = 0.01 # regularization term
# experiment['tau'] = np.array(tau)
# experiment['n_tau'] = len(tau)
# experiment['kappa'] = 0 # penalty term
# experiment['margin'] = 0 # penalty margin
# experiment['print_cost'] = 0 # 1 = plot cost
# experiment['plot_results'] = 1 # 1 = plot results
#--------------------------------------------------------------------------

experiment = dsm.functions.load(experiment)  # load template homes
experiment = dsm.functions.simulate_city(experiment)  # simulate phi
experiment = dsm.functions.simulate_load_price(experiment)
# dsm.functions.output_results(experiment)

experiment = dsm.detection.split_data_online(experiment)
experiment = dsm.detection.SARIMA_forecast(experiment)
# experiment = dsm.attack_model.attack_model(experiment)

dsm.detection.residual_analysis(experiment)
# experiment = dsm.detection.GRLT_detector(experiment)
# experiment = dsm.detection.CUSUM_detector(experiment)

# L = experiment['L_total']
# L_test_prediction = experiment['L_test_prediction']
# L_train_prediction = experiment['L_train_prediction']
# N_train = experiment['N_train']
# L[N_train:, 0] = L_test_prediction.ravel()
# L_processed = np.ravel(L)
# experiment['L_processed'] = L_processed[24:]
#
# experiment = dsm.features.lagged_load(experiment)
# experiment = dsm.features.lagged_price(experiment)
# experiment = dsm.detection.split_data(experiment, scaling=None)
# experiment = dsm.detection.QARNET_detector(experiment, test_method=0)  # 0 = QARNET, 1 = QAR
# experiment = dsm.evaluation.evaluate_PI_results(experiment)
# dsm.evaluation.output_PI_results(experiment)


'''
for i in range(boot):
    print(i)
    try:
        del experiment['X']
    except:
        None
    experiment = dsm.functions.load(experiment) # load template homes
    experiment = dsm.functions.simulate_city(experiment) # simulate phi
    experiment = dsm.functions.simulate_load_price(experiment)
    experiment = dsm.detection.SARIMA_forecast(experiment)
    experiment = dsm.features.lagged_load(experiment)
    experiment = dsm.features.lagged_price(experiment)
    experiment = dsm.detection.split_data(experiment, scaling= None)
    experiment = dsm.detection.QARNET_detector(experiment,test_method=0) # 0 = QARNET, 1 = QAR
    experiment = dsm.evaluation.evaluate_PI_results(experiment)
    QS_boot[i] = experiment['QS']
    IS_boot[i] = experiment['IS']
    ACE_boot[i] = experiment['ACE']
    SHARP_boot[i] = experiment['SHARP']

experiment['QS_boot'] = QS_boot
experiment['IS_boot'] = IS_boot
experiment['ACE_boot'] = ACE_boot
experiment['SHARP_boot'] = SHARP_boot
# dsm.functions.output_results(experiment)
# dsm.evaluation.output_PI_results(experiment)
dsm.evaluation.output_PI_CI_results(experiment)
'''

# print runtime to console
print('\nRuntime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()