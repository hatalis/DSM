'''
By Kostas Hatalis
'''
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras import regularizers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import itertools
K.set_floatx('float64')

def QARNET_detector(experiment,test_method=0):

    # load in training data (x,y)
    X_train = experiment['X_train']  # data, numpy array of shape (number of features, number of examples)
    y_train = experiment['y_train']  # true observations vector of shape (1, number of examples)
    X_test = experiment['X_test']
    smooth_loss = experiment['smooth_loss']
    kappa = experiment['kappa']
    alpha = experiment['smoothing_alpha']
    margin = experiment['margin']
    Lambda = experiment['Lambda']
    tau = experiment['tau']
    hidden_dims = experiment['hidden_dims']
    maxIter = experiment['maxIter']
    batch_size = experiment['batch_size']
    activation = experiment['activation']
    optimizer = experiment['optimizer']
    N_features = experiment['N_features']
    n_tau = experiment['n_tau']
    layers_dims = [N_features]+hidden_dims+[n_tau]

    # -------------------------------------- build the model
    model = Sequential()
    if test_method == 0: # QARNN
        for i in range(0,len(layers_dims)-2):
            model.add(Dense(layers_dims[i+1], input_dim=layers_dims[i], kernel_regularizer=regularizers.l2(Lambda),
                            kernel_initializer='normal', activation=activation))
        model.add(Dense(layers_dims[-1], kernel_initializer='normal'))
    elif test_method == 1: # QAR
        model.add(Dense(layers_dims[-1], input_dim=layers_dims[0], kernel_regularizer=regularizers.l2(Lambda),
                        kernel_initializer='normal'))

    # -------------------------------------- compile and fit model
    model.compile(loss=lambda Y, Q: pinball_loss(tau,Y,Q,alpha,smooth_loss,kappa,margin), optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=maxIter, verbose=0, batch_size=batch_size)

    # -------------------------------------- estimate quantiles of testing data
    q_hat = model.predict(X_test)

    LB = q_hat[:,0]
    UB = q_hat[:,1]


    y_true = experiment['y_true']
    N_test = experiment['N_test']
    L_test_attack = experiment['L_test_attack']

    y_pred = np.zeros((N_test, 1))
    for t in range(N_test):
        if (L_test_attack[t] > UB[t]):
            y_pred[t] = 1
            print(t)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, ['$H_0$','$H_1$'], normalize=True)


    experiment['q_hat'] = q_hat
    experiment['costs'] = history.history['loss']

    return experiment




# pinball loss function with penalty
def pinball_loss(tau, y, q, alpha = 0.01, smooth_loss = 1, kappa=0, margin=0):
    error = (y - q)
    diff = q[:, 1:] - q[:, :-1]

    quantile_loss = 0
    if smooth_loss == 0: # pinball function
        quantile_loss = K.mean(K.maximum(tau * error, (tau - 1) * error))
    elif smooth_loss == 1: # smooth pinball function
        quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha))
    elif smooth_loss == 2: # huber norm approximation
        epsilon = 2 ** -8

        # if K.abs(error) > epsilon:
        #     u = K.abs(error) - epsilon / 2
        # else:
        #     u = (error**2) / (2 * epsilon)

        logic = K.cast((K.abs(error) > epsilon),dtype='float64')

        u = (K.abs(error)-epsilon/2)*logic + ((error**2) / (2 * epsilon))*(1-logic)

        quantile_loss = K.mean(K.maximum(tau * u, (tau - 1) * u))


    # penalty = -kappa * K.mean(alpha2*K.softplus(-diff / alpha2))
    # penalty = K.mean(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)) * kappa
    penalty = kappa * K.mean(tf.square(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)))

    return quantile_loss + penalty


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
