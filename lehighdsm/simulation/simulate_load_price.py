
import numpy as np

def simulate_load_price(experiment):
    """
    Classifier models for binary prediction, as defined in the sklearn-module.

    Args:
        experiment(dict): Experiment dictionary.

    Returns:
        dict: Experiment dictionary with additional keys.
    """
    T = experiment['N_total']
    N_train = experiment['N_train']
    N_test = experiment['N_test']
    N = experiment['N']
    phi = experiment['city'].values
    epsilon_D = experiment['epsilon_D']
    window = experiment['window']
    L_hat_method = experiment['L_hat_method']
    L_target = experiment['L_target']
    kappa = experiment['kappa']
    goal = experiment['goal']
    attack = experiment['attack']

    y = np.zeros((T,1))
    P = np.zeros((T,1))
    L = np.zeros((T,N))
    L_total = np.zeros((T,1))
    Phi_hat = np.zeros((T, 1))
    L_target = np.ones((T,1))*L_target

    # Run price and load feedback simulation
    Phi_total = np.sum(phi,axis = 1)[:T]
    L_total[0] = Phi_total[0]
    i = 0
    for t in range(1,T):
        # 1. DEFINE PRICE
        if t > 0: # change to window if using SARIMA forecast!
            # Make Phi forecast.
            Phi_hat[t] = load_forecast(t, Phi_total[:t], L_hat_method, window)
            # Pick load goal.
            L_adjusted = L_target[t] + goal*(L_target[t-1]-L_total[t-1])
            if L_adjusted < 0:
                L_adjusted = 10
            # Set prices.
            P[t] = (L_adjusted/Phi_hat[t])**(1/epsilon_D)
        # 2. DEFINE LOAD
        L[t, :] =  kappa*(phi[t, :]*(P[t]**epsilon_D)) + (1 - kappa) * phi[t, :]
        L_total[t] = np.sum(L[t, :])

        # Add attack to load on testing set only.
        if t >= N_train:
            L_total[t] = L_total[t] + attack[i]
            y[t] = (attack[i]>0)*1
            i = i + 1

    P = P / 100  # convert cents to USD

    experiment['Phi_total'] = Phi_total
    experiment['Phi_train'] = Phi_total[:N_train]
    experiment['Phi_test'] = Phi_total[N_train:]
    experiment['Phi_hat'] = Phi_hat
    experiment['L_target'] = L_target
    experiment['L_total'] = L_total
    experiment['L_train'] = L_total[:N_train]
    experiment['L_test'] = L_total[N_train:]
    experiment['attack'] = attack
    experiment['P_total'] = P
    experiment['P_train'] = P[:N_train]
    experiment['P_test'] = P[N_train:]
    experiment['L'] = L # individual home loads

    # binary vector classifying attacks
    experiment['y'] = y

    return experiment

# Set Phi (base load) 1-step ahead forecast
from statsmodels.tsa.statespace.sarimax import SARIMAX
def load_forecast(t, L_total, method, window):

    L_hat = 0
    if method == 0: # persistance forecast
        L_hat = float(L_total[-1])
    elif method > 0:
        if t > window:
            if method == 1: # SARIMA
                # print(np.shape(L_total[-1-window:]),t)
                model = SARIMAX(L_total[-1-window:], order=(1,1,1), seasonal_order=(1,1,0,24),
                                enforce_invertibility=False,enforce_stationarity=False)
                model_fit = model.fit(disp=False)
                L_hat = model_fit.forecast()
            # else: # SARIMAX
            #     # print(t)
            #     model = SARIMAX(L_total[-1-window:],  exog=P[-1-window:], order=(1,1,1), seasonal_order=(1,1,0,24),
            #                     enforce_invertibility=False,enforce_stationarity=False)
            #     model_fit = model.fit(disp=False)
            #     L_hat = model_fit.forecast(exog=P[-1].reshape((1, 1)))
        else:
            L_hat = float(L_total[-1]) # persistance forecast

    return L_hat