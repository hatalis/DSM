import pandas as pd
import numpy as np

def lagged_load(experiment):
    """
    Computes lagged load features

    Arguments:
        experiment(dict): Dictionary containing L_total
        n(int): Lag-time

    Returns:
        experiment(dict): Experiment dictionary with additional or updated key: X
    """
    L_processed = experiment['L_processed']
    n = experiment['lags']
    Load = pd.DataFrame(data = {'L': np.ravel(L_processed)})

    # create feature columns
    data = {}
    if n > 0:
        for i in range(1, n+1):
            name = 'L_lag_'+str(i)
            data[name] = Load['L'].shift(periods=i)
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