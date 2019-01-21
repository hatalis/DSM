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

def CUSUM_detector(experiment):

    L_total = experiment['L_total']
    N_train = experiment['N_train']
    N_test = experiment['N_test']
    L_test_prediction = experiment['L_test_prediction']
    L_train_prediction = experiment['L_train_prediction']
    L_train_actual = experiment['L_train_actual']
    L_test_actual = experiment['L_test_actual']
    L_test_attack = experiment['L_test_attack']

    residual_train = L_train_actual - L_train_prediction
    residual_test = L_test_attack - L_test_prediction
    residual_noattack = L_test_actual - L_test_prediction
    residual_train = residual_train[24:]

    var = residual_train.var()
    sigma = residual_train.std()
    mu = residual_train.mean()

    residual_train = (residual_train - mu) / 1
    residual_test = (residual_test - mu) / 1
    residual_noattack = (residual_noattack - mu) / 1

    # plt.figure()
    # plt.plot(residual_test, 'r',label='Attack')
    # plt.plot(residual_noattack,'g',label='No Attack')
    # plt.xlabel('Time (hrs)')
    # # plt.ylabel('Attack Level')
    # plt.legend()

    h = 2*sigma
    k = 0

    # CUSUM detection
    y_pred = np.zeros((N_test, 1))
    g = np.zeros((N_test, 1))
    s = residual_test

    temp = 0 + s[0] - k
    g[0] = max(0, temp)

    for t in range(1,N_test):
        temp = g[t-1] + s[t] - k
        g[t] = max(0,temp)

        if g[t] > h:
            g[t] = 0
            y_pred[t] = 1
            print(t)


    y_true = experiment['y_true']
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, ['$H_0$', '$H_1$'])

    return experiment




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()