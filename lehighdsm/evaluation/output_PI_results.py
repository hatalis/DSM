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
    PICP = experiment['PICP']
    plot_results = experiment['plot_results']
    print_cost = experiment['print_cost']
    costs = experiment['costs']
    q_hat = experiment['q_hat']
    y_test = experiment['y_test']
    N_test = experiment['N_test']
    n_tau = experiment['n_tau']
    N_train = experiment['N_train']
    L_test_attack = experiment['L_test_attack']

    print('PICP = ',PICP)
    print('Sharpness = ',SHARP)
    print('ACE = ', ACE)
    print('QS = ',QS)
    print('IS = ',IS)

    # PIs = np.arange(0.1,1,0.1)
    # plt.figure()
    # plt.plot(PIs,PICP.T)
    # plt.xlim(0,1)

    # plt.plot(PINC,SHARP)
    if print_cost:
        plt.figure()
        plt.plot(np.squeeze(costs))
        plt.ylabel('loss')
        plt.xlabel('epcoh')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.yscale('log')

    if plot_results:
        plt.figure()
        x = (range(N_train,N_train+N_test))
        plt.plot(x,y_test, 'r*')
        n_PIs = n_tau // 2
        for i in range(n_PIs):
            y1 = q_hat[:,i]
            y2 = q_hat[:,-1-i]
            plt.fill_between(x, y1, y2, color='blue', alpha=0.4) # alpha=str(1/n_PIs)
        plt.ylabel('Normalized Load')
        plt.xlabel('Time (hour)')
        plt.plot(x,L_test_attack)
    return None