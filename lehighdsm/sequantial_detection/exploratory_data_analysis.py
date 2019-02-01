
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def exploratory_data_analysis(experiment):
    """
    Plots L_train, 1-lag L_train, and ACF/PACF of differenced L_train.
    These plots can help in picking SARIMA parameters.

    Args:
        experiment(dict): Experiment dictionary.

    Returns:
        None
    """
    L_train = experiment['L_train']
    L_train_diff = np.diff(L_train,n=2,axis=0)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(L_train)
    plt.subplot(2, 1, 2)
    plt.plot(L_train_diff)

    fig, axes = plt.subplots(2, 1)
    plot_acf(L_train_diff, lags=48, ax=axes[0])
    plot_pacf(L_train_diff, lags=48, ax=axes[1])

    return None
