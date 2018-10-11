# -*- coding: utf-8 -*-
"""

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np

def output_PI_results(experiment):

    SHARP = experiment['SHARP']
    QS = experiment['QS']
    IS = experiment['IS']
    ACE = experiment['ACE']
    PINC = experiment['PINC']
    plot_results = experiment['plot_results']
    print_cost = experiment['print_cost']
    costs = experiment['costs']
    q_hat = experiment['q_hat']
    y_test = experiment['y_test']
    N_test = experiment['N_test']
    n_tau = experiment['n_tau']
    N_train = experiment['N_train']

    print('PINC = ',PINC)
    print('Sharpness = ',SHARP)
    print('ACE = ', ACE)
    print('QS = ',QS)
    print('IS = ',IS)

    # plt.plot(PINC,SHARP)
    if print_cost:
        plt.figure(0)
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('epcoh')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.yscale('log')

    if plot_results:
        plt.figure(1)
        x = (range(N_train,N_train+N_test))
        plt.plot(x,y_test, 'r*')
        n_PIs = n_tau // 2
        for i in range(n_PIs):
            y1 = q_hat[:,i]
            y2 = q_hat[:,-1-i]
            plt.fill_between(x, y1, y2, color='blue', alpha=0.4) # alpha=str(1/n_PIs)
        plt.ylabel('Normalized Load')
        plt.xlabel('Time (hour)')
    return None