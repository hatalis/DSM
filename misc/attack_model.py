import numpy as np
import matplotlib.pyplot as plt

def attack_model(experiment):

    N_test = experiment['N_test']
    L_test = experiment['L_test']


    attack = np.zeros((N_test,1))

    # Gradual or Sudden Attack:
    for t in range(N_test/2,N_test):
        attack[t] = attack[t-1] + 0

    # Point Attack:
    # attack[24] = 250
    # attack[30] = 200
    # attack[46] = 300


    L_test_attack = L_test + attack


    # binary vector classifying attacks
    y_true =(attack > 0)*1
    # y_actual = y_actual.ravel()
    # y_actual = y_actual.reshape((N_test,1))
    # print(y_actual)
    experiment['y_true'] = y_true
    experiment['attack'] = attack
    experiment['L_test_attack'] = L_test_attack



    return experiment