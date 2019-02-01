
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import numpy as np

def GLRT_detector(experiment, P_FA = None, N_roc = 20, N_window = 24):
    """
    Computes Windowed-GLRT detector.

    Args:
        experiment(dict): Dictionary containing trained ML model results
        P_FA(float): probability of false alarm, 0.05 is a standard value
            -if None, it runs full ROC analysis to determine best P_FA
        N_window(int): size of window to calculate test statistic
        N_roc(int): number of thresholds to test for in ROC calculation
    Returns:
        experiment(dict): Experiment dictionary with additional or updated keys.
    """

    N_test = experiment['N_test']
    L_test_prediction = experiment['L_test_prediction']
    L_train_prediction = experiment['L_train_prediction']
    L_train_actual = experiment['L_train']
    L_test_attack = experiment['L_test']
    y = experiment['y']
    N_train = experiment['N_train']

    residual_train = L_train_actual - L_train_prediction
    residual_test = L_test_attack - L_test_prediction
    residual_train = residual_train[24:] # ignore first period, typically bad fit by SARIMA

    var = residual_train.var() # population variance used for threshold
    y_pred = np.zeros((N_test, 1))
    y_train = y[:N_train]
    y_test = y[N_train:]
    y_pred_best = y_pred

    FPR = np.zeros((N_roc,1))
    TPR = np.zeros((N_roc, 1))
    d = np.ones((N_roc, 1))
    if P_FA == None:
        P_FA_range = np.linspace(0, 1, num=N_roc)
    else:
        P_FA_range = P_FA

    i, best_d = 0, 10
    for P_FA in P_FA_range:
        # plt.figure()
        y_pred = np.zeros((N_test, 1))
        for t in range(1,N_test):
            if N_window>t:
                a = residual_train[-1-(N_window-t)+1:].ravel()
                b = residual_test[:t].ravel()
                window = np.concatenate((a,b))
            else:
                window = residual_test[t-N_window:t]
            window = window.reshape((N_window,1))
            # x = np.arange(t-N_window,t)
            # plt.plot(x,window+t*0)
            threshold = np.sqrt(var / N_window) * norm.isf(P_FA)
            test_statistic = np.mean(window)
            if test_statistic > threshold:
                y_pred[t] = 1 # H1: attack detected
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

    FPR[-1] = 1

    experiment['FPR'] = FPR
    experiment['TPR'] = TPR
    experiment['y_train'] = y_train
    experiment['y_test'] = y_test
    experiment['y_test_prediction'] = y_pred_best

    return experiment


