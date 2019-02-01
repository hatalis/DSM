'''
This script loads in a csv file of template home data.
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
np.random.seed(1)
start_time = time.time()

#--------------------------------------------------------------------------
# Parameters for data simulation
experiment = {}
experiment['N_train'] = 24*28
experiment['N_test'] = 24*2
experiment['window'] = 24*2 # num of lags for features for SARIMA in simulation

# Parameters for price-load feedback simulation
experiment['kappa'] = 0.9
experiment['N'] = 200 # number of homes to simulate
experiment['L_target'] = 200 # target load
experiment['epsilon_D'] = -1 # elasticity of load
experiment['L_hat_method'] = 0 # 0 = persistence, 1 = SARIMA
experiment['goal'] = 0 # = L', 1 = L' + delta (see paper)
lw = 1 # ROC line width
#--------------------------------------------------------------------------
# # Gradual or Sudden Attack:
N_test = experiment['N_test']
attack = np.zeros((N_test, 1))
for t in range(23, N_test):
    attack[t] = attack[t - 1] + 0*5  # Gradual Attack
    # attack[t] = 150 # Sudden Attack
# Point Attack:
attack[24] = 250
attack[29] = 200
attack[34] = 300
attack[37] = 100
attack[46] = 150
experiment['attack'] = attack
# --------------------------------------------------------------------------

experiment = dsm.simulation.load_data(experiment)  # load template homes
experiment = dsm.simulation.simulate_city(experiment)  # simulate city of homes (phi)
experiment = dsm.simulation.simulate_load_price(experiment) # simulate feedback
# dsm.evaluation.output_simulation_results(experiment)

'''
Supervised Learning Detection
method(int): 0: k-Nearest Neighbors
             1: Logistic Regression
             2: Random Forest
             3: Support Vector Classifier
             4: Gaussian Naive Bayes
             5: Decision Trees
             6: AdaBoost Classifier
             7: Gradient Boosting Classifier
             8: Neural Network Classifier
'''
experiment['lags'] = 24 # includes itself lag 0
experiment = dsm.learning_detection.create_training_data(experiment)

# TEST LR
experiment = dsm.learning_detection.sklearn(experiment, method = 1)
experiment = dsm.evaluation.confusion_metrics(experiment)
plt.plot(experiment['FPR'], experiment['TPR'], color='blue', lw=lw, label='LR Detector')
#
# # TEST LR
# experiment = dsm.learning_detection.sklearn(experiment, method = 2)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# plt.plot(experiment['FPR'], experiment['TPR'], color='brown', lw=lw, label='RF Detector')
#
# # TEST GNB
# experiment = dsm.learning_detection.sklearn(experiment, method = 4)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# plt.plot(experiment['FPR'], experiment['TPR'], color='green', lw=lw, label='GNB Detector')
#
# # TEST GBC
# experiment = dsm.learning_detection.sklearn(experiment, method = 7)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# plt.plot(experiment['FPR'], experiment['TPR'], color='purple', lw=lw, label='GBC Detector')
#
# # TEST kNN
# experiment = dsm.learning_detection.sklearn(experiment, method = 8)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# plt.plot(experiment['FPR'], experiment['TPR'], color='pink', lw=lw, label='ANN Detector')


'''
Change Point Detection
method(int): 0: Windowed-GLRT
             1: CUSUM Test
'''
# dsm.sequantial_detection.exploratory_data_analysis(experiment) # ACF analysis for SARIMA parameters
experiment = dsm.sequantial_detection.SARIMA_forecast(experiment, plot_fit = False)
dsm.sequantial_detection.residual_analysis(experiment) # to see if residuals are iid Gaussian

# experiment = dsm.sequantial_detection.GLRT_detector(experiment)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# plt.plot(experiment['FPR'], experiment['TPR'], color='red', lw=lw, label='GLRT Detector')

# experiment = dsm.sequantial_detection.CUSUM_detector(experiment)
# experiment = dsm.evaluation.confusion_metrics(experiment)
# dsm.evaluation.plot_ROC_curve(experiment['FPR'], experiment['TPR'], label = 'CUSUM Detector',lw=lw)

#--------------------------------------------------------------------------
# print runtime to console
print('\nRuntime in seconds = {:,.4f}'.format(float(time.time() - start_time)))
plt.show()