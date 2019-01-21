import numpy as np
import matplotlib.pyplot as plt

def attack_model(experiment):

    N_test = experiment['N_test']
    L_test_actual = experiment['L_test_actual']

    # Gradual Attack
    # a = np.zeros((24,1)).T
    # b = np.linspace(0,100,num = 24).reshape((24,1)).T
    # attack = np.hstack((a, b)).T

    # Sudden Attack
    # a = np.zeros((1,24))
    # b = np.ones((1,24))*100
    # attack = np.hstack((a, b)).T

    # Point Attack
    attack = np.zeros((48,1))
    attack[12] = 75
    attack[24] = 50
    attack[36] = 100


    L_test_attack = L_test_actual + attack


    # binary vector classifying attacks
    y_true =(attack > 0)*1
    # y_actual = y_actual.ravel()
    # y_actual = y_actual.reshape((N_test,1))
    # print(y_actual)

    experiment['y_true'] = y_true
    experiment['attack'] = attack
    experiment['L_test_attack'] = L_test_attack

    # plt.figure()
    # plt.plot(attack,'r')
    # plt.xlabel('Time (hrs)')
    # plt.ylabel('Attack Level')
    # plt.title('Point Attack')

    # plt.plot(L_test_actual)
    # plt.plot(experiment['L_test_prediction'])

    return experiment