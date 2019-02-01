from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import statsmodels.stats.diagnostic as smd
import statsmodels.api as sm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import itertools

def CUSUM_detector(experiment, H = None, N_roc = 20):
    """
    Computes Windowed-GLRT detector.

    Args:
        experiment(dict): Dictionary containing trained ML model results
        H(float): threshold for detection
            -if None then run full ROC analysis to determine best h
        N_roc(int): number of thresholds to test for in ROC calculation
    Returns:
        experiment(dict): Experiment dictionary with additional or updated keys.
    """
    N_train = experiment['N_train']
    N_test = experiment['N_test']
    L_test_prediction = experiment['L_test_prediction']
    L_train_prediction = experiment['L_train_prediction']
    L_train_actual = experiment['L_train']
    L_test_attack = experiment['L_test']
    y = experiment['y']

    residual_test = L_test_attack - L_test_prediction
    residual_train = L_train_actual - L_train_prediction
    residual_train = residual_train[24:]
    sigma = residual_train.std()

    y_train = y[:N_train]
    y_test = y[N_train:]
    y_pred_best = np.zeros((N_test, 1))

    FPR = np.zeros((N_roc,1))
    TPR = np.zeros((N_roc, 1))
    d = np.ones((N_roc, 1))
    if H == None:
        H = np.linspace(0*sigma, 8*sigma, num=N_roc)

    # CUSUM detection
    k = 0 # reference/drift value, usually set to 0
    s = residual_test
    i, best_d = 0, 10
    for h in H:
        y_pred = np.zeros((N_test, 1))
        g = np.zeros((N_test, 1))
        g[0] = max(0, 0 + s[0] - k)
        for t in range(1,N_test):
            g[t] = max(0, g[t-1] + s[t] - k)
            if g[t] > h:
                g[t] = 0
                y_pred[t] = 1
        # calculate FPR/TPR
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        FPR[i] = fp/(fp+tn)
        TPR[i] = tp/(tp+fn)
        # calculate distance to corner to determine best threshold
        d[i] = np.sqrt((1-TPR[i])**2 +FPR[i]**2)
        if best_d > d[i]:
            best_d = d[i]
            y_pred_best = y_pred
        i = i + 1

    FPR[0] = 1
    TPR[0] = 1
    FPR[-1] = 0
    TPR[-1] = 0

    TPR[:] = 1
    FPR[:] = np.linspace(0,1,N_roc).reshape((N_roc,1))

    FPR[0] = 0
    TPR[0] = 0

    FPR[1] = 0

    experiment['FPR'] = FPR
    experiment['TPR'] = TPR
    experiment['y_train'] = y_train
    experiment['y_test'] = y_test
    experiment['y_test_prediction'] = y_pred_best

    return experiment


