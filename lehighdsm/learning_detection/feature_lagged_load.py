import pandas as pd
import numpy as np

def feature_lagged_load(experiment, name_start = 'L_lag_', L = None):
    """
    Computes lagged load features

    Arguments:
        experiment(dict): Dictionary containing L_total
        n(int): Lag-time

    Returns:
        experiment(dict): Experiment dictionary with additional or updated key: X
    """
    lags = experiment['lags']
    Load = pd.DataFrame(data = {'L': np.ravel(L)})

    # create feature columns
    data = {}
    for i in range(lags):
        name = name_start+str(i)
        data[name] = Load['L'].shift(periods=i)
    feature = pd.DataFrame(data)

    try:
        X = experiment['X']
        X = X.join(feature, how='outer')
    except KeyError:
        X = feature

    experiment['X'] = X

    return experiment