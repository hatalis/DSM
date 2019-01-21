import pandas as pd
import numpy as np

def lagged_price(experiment):
    """
    Computes lagged price features.

    Arguments:
        experiment(dict): Dictionary containing P
        n(int): Lag-time

    Returns:
        experiment(dict): Experiment dictionary with additional or updated key: X
    """
    n = experiment['lags']

    L_processed = experiment['L_processed']
    alpha = experiment['alpha']
    beta = experiment['beta']
    epsilon_P = experiment['epsilon_P']
    P = beta * ((alpha - L_processed) ** epsilon_P)
    Prices = pd.DataFrame(data={'P': np.ravel(P)})

    experiment['Prices_SARIMA'] = Prices

    # create feature columns
    data = {}
    if n > 0:
        for i in range(1, n+1):
            name = 'P_lag_'+str(i)
            data[name] = Prices['P'].shift(periods=i)
    else:
        print('Error: lags must be greater then 0!')

    feature = pd.DataFrame(data)

    try:
        X = experiment['X']
        X = X.join(feature, how='outer')
    except KeyError:
        X = feature

    experiment['X'] = X

    return experiment