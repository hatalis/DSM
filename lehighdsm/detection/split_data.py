from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def split_data(experiment, scaling = None):
    """
    Splits data into train- and test-set and (optionally) applies scaling to all features.

    Args:

    """
    # load in data from dictionary
    X = experiment['X']
    experiment['scaling'] = scaling
    L_processed = experiment['L_processed']
    X_test_SARIMA = experiment['X_test_SARIMA']

    N_test = experiment['N_test']
    y = pd.DataFrame(data={'target': np.ravel(L_processed)})

    # combine covariates and labels, drop rows with NaN values; do the same thing for returns
    Xy = X.join(y, how='outer')
    Xy = Xy.dropna()

    # redefine X and y from Xy with remaining dates
    X = Xy.drop(columns=['target'])
    y = Xy[['target']]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N_test, shuffle=False)
    split_index = X_test.index.values[0]  # first index of test set (important for evaluation)

    # convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test_true = y_test.values

    y_test = X_test_SARIMA


    # apply scaling and convert scaled data back to pandas
    if scaling is not None:
        X_train = scaling.fit_transform(X_train)
        X_test = scaling.transform(X_test)
        y_train = scaling.fit_transform(y_train)
        y_test = scaling.transform(y_test)

    # save everything to dictionary
    experiment['X_train'] = X_train
    experiment['y_train'] = y_train
    experiment['X_test'] = X_test
    experiment['y_test'] = y_test
    experiment['y_test_true'] = y_test_true
    experiment['split_index'] = split_index
    experiment['N_features'] = np.size(X_train, axis = 1)

    return experiment
