# -*- coding: utf-8 -*-
"""

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np

def output_PI_CI_results(experiment):

    # SHARP = experiment['SHARP']
    # QS = experiment['QS']
    # IS = experiment['IS']
    # ACE = experiment['ACE']
    # PINC = experiment['PINC']
    # plot_results = experiment['plot_results']
    # print_cost = experiment['print_cost']
    # costs = experiment['costs']
    # q_hat = experiment['q_hat']
    # y_test = experiment['y_test']
    # N_test = experiment['N_test']
    # n_tau = experiment['n_tau']
    # N_train = experiment['N_train']

    QS_boot = experiment['QS_boot']
    IS_boot = experiment['IS_boot']
    ACE_boot = experiment['ACE_boot']
    SHARP_boot = experiment['SHARP_boot']


    # CIs = np.zeros((18,3))
    # for i in range(18):
    print('QS = ',confidence_intervals(QS_boot))
    print('IS = ', confidence_intervals(IS_boot))
    print('ACE = ',confidence_intervals(ACE_boot))
    print('SG = ', confidence_intervals(SHARP_boot))

    # plt.plot(PINC,SHARP)
    #
    # if plot_results:
    #     plt.figure(1)
    #     x = (range(N_train,N_train+N_test))
    #     plt.plot(x,y_test, 'r*')
    #     n_PIs = n_tau // 2
    #     for i in range(n_PIs):
    #         y1 = q_hat[:,i]
    #         y2 = q_hat[:,-1-i]
    #         plt.fill_between(x, y1, y2, color='blue', alpha=0.4) # alpha=str(1/n_PIs)
    #     plt.ylabel('Normalized Load')
    #     plt.xlabel('Time (hour)')
    return None

def confidence_intervals(stats):
    alpha = 0.95
    p = int(((1.0-alpha)/2.0) * 100)
    lower = max(0.0, np.percentile(stats, p))

    p = int((alpha+((1.0-alpha)/2.0)) * 100)
    upper = min(1.0, np.percentile(stats, p))

    median = np.percentile(stats, 50)

    return lower, median, upper