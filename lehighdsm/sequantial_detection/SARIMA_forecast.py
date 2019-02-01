
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt

def SARIMA_forecast(experiment, plot_fit = False):
    """
    Fit SARIMA model on L_train data and forecast N_test steps ahead

    Args:
        experiment(dict): Experiment dictionary.

    Returns:
        experiment(dict): Experiment dictionary with additional keys.
    """
    L_train = experiment['L_train']
    N_train =  experiment['N_train']
    N_test = experiment['N_test']

    # Fit SARIMA model
    model = SARIMAX(L_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 24),
                    enforce_invertibility=False, enforce_stationarity=False)
    model_fit = model.fit(disp=False)
    L_train_prediction = model_fit.fittedvalues
    L_test_prediction = model_fit.forecast(steps = N_test)

    # reshape arrays
    L_test_prediction = L_test_prediction.reshape((N_test, 1))
    L_train_prediction = L_train_prediction.reshape((N_train, 1))

    # plot fit on training data and prediction
    if plot_fit:
        plt.figure()
        plt.plot(L_train)
        plt.plot(L_train_prediction,'red')
        plt.figure()
        plt.plot(L_test_prediction,'red')

    experiment['L_test_prediction'] = L_test_prediction
    experiment['L_train_prediction'] = L_train_prediction

    return experiment