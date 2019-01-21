
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

def SARIMA_forecast(experiment):

    L = experiment['L_total']
    N_train =  experiment['N_train']
    N_test = experiment['N_test']

    # L_diff = np.diff(L,n=1,axis = 0)
    # plt.figure()
    # plot_acf(L_diff,lags=48)
    # plot_pacf(L_diff, lags=48)
    # plt.figure()
    # plt.plot(L_diff)
    # plt.show()

    model = SARIMAX(L[:N_train], order=(1, 1, 0), seasonal_order=(1, 0, 0, 24),
                    enforce_invertibility=False, enforce_stationarity=False)
    model_fit = model.fit(disp=False)
    L_test_prediction = model_fit.forecast(steps = N_test)

    L_test_prediction = L_test_prediction.reshape((N_test, 1))
    L_train_prediction = model_fit.fittedvalues
    L_train_prediction = L_train_prediction.reshape((N_train, 1))





    experiment['L_test_prediction'] = L_test_prediction
    experiment['L_train_prediction'] = L_train_prediction



    return experiment