
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

def SARIMA_forecast(experiment):

    L_total = experiment['L_total']
    N_train =  experiment['N_train']
    N_test = experiment['N_test']

    model = SARIMAX(L_total[:N_train], order=(1, 1, 1), seasonal_order=(1, 1, 0, 24),
                    enforce_invertibility=False, enforce_stationarity=False)
    model_fit = model.fit(disp=False)
    X_test_SARIMA = model_fit.forecast(steps = N_test)

    experiment['X_test_SARIMA'] = X_test_SARIMA
    x = (range(N_train, N_train + N_test))

    L_total = np.reshape(L_total,(len(L_total),1))
    L_total[N_train:,0] = X_test_SARIMA
    L_processed = np.ravel(L_total)

    experiment['L_processed'] = L_processed

    experiment['X_test_SARIMA'] = X_test_SARIMA
    plt.plot(x,X_test_SARIMA)
    return experiment