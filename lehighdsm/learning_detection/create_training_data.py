import pandas as pd
import numpy as np
import lehighdsm as dsm

def create_training_data(experiment):

    # first pass of data to create class y = 0
    experiment = dsm.learning_detection. \
        feature_lagged_load(experiment, name_start='L_lag_', L=experiment['L_total'])
    experiment = dsm.learning_detection.split_data(experiment)

    # create second pass of data to add simulated attacks to training data
    L_train = experiment['L_train']
    N_train = experiment['N_train']

    # ---------------------
    # Gradual or Sudden Attack:
    attack = np.zeros((N_train, 1))
    for t in range(int(N_train/2)):
        attack[t] = 100 # Sudden Attack
    for t in range(int(N_train / 2),N_train):
        attack[t] = attack[t-1] + 10 # Gradual Attack

    # Point Attack:
    # attack[24] = 250
    # attack[30] = 200
    # attack[46] = 300
    L_train_attack = L_train + attack
    # ---------------------

    lags = experiment['lags']
    Load = pd.DataFrame(data = {'L': np.ravel(L_train_attack)})

    # create feature columns
    data = {}
    for i in range(lags):
        name = 'L_attack_lags_'+str(i)
        data[name] = Load['L'].shift(periods=i)
    X = pd.DataFrame(data)
    y = np.ones((N_train,1))

    # combine covariates and labels, drop rows with NaN values; do the same thing for returns
    y = pd.DataFrame(data={'target': np.ravel(y)})
    Xy = X.join(y, how='outer')
    Xy = Xy.dropna()

    # redefine X and y from Xy with remaining dates
    X_train_attack = Xy.drop(columns=['target'])
    y_train_attack = Xy[['target']]
    X_train_attack = X_train_attack.values
    y_train_attack = y_train_attack.values


    # load in previous training data and add new data
    X_train = experiment['X_train']
    y_train = experiment['y_train']


    X_train = np.concatenate((X_train, X_train_attack), axis=0)
    y_train = np.concatenate((y_train, y_train_attack), axis=0)

    experiment['X_train'] = X_train
    experiment['y_train'] = y_train
    # experiment['N_train'] = np.shape(y_train)[0]

    return experiment