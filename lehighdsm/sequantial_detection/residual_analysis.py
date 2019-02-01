
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import numpy as np

def residual_analysis(experiment):
    """
     Fit SARIMA model on L_train data and forecast N_test steps ahead

     Args:
         experiment(dict): Experiment dictionary.

     Returns:
         None.
    """
    L_train_prediction = experiment['L_train_prediction']
    L_train_true = experiment['L_train']

    residual_train = L_train_prediction - L_train_true
    residual_train = residual_train[25:]

    print(np.mean(residual_train))
    print(np.var(residual_train))
    plt.figure()
    plt.plot(L_train_prediction,'red')
    plt.plot(L_train_true)

    # plot residual series
    plt.figure()
    plt.plot(residual_train)
    plt.title('Training Residual')
    plt.xlabel('Time (hr)')
    plt.ylabel('Residual')

    # ACF analysis for independence
    fig, axes = plt.subplots(2, 1)
    plot_acf(residual_train, lags=48, ax=axes[0])
    plot_pacf(residual_train, lags=48, ax=axes[1])
    fig.tight_layout()

    # Q-Q Plot
    plt.figure()
    sigma = residual_train.std()
    stats.probplot(residual_train.ravel(), dist="norm", plot=plt, sparams=(0, sigma))
    plt.title('Q-Q Plot of Training Residuals')
    plt.grid()

    '''
    ADF TEST:
       p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
       p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    D’Agostino-Pearson Test:
        p-value > 0.05: Fail to reject the null hypothesis (H0), x comes from a normal distribution.
        p-value <= 0.05: Reject the null hypothesis (H0), x does not comes from a normal distribution.
    Jarque-Bera test:
        The null hypothesis of the Jarque-Bera test is a joint hypothesis of the skewness 
        being zero and the excess kurtosis being zero. 
        p-value > 0.05: Fail to reject the null hypothesis (H0); data is Gaussian.
        p-value <= 0.05: Reject the null hypothesis (H0); data is not Gaussian.
        Note: a large JB Statistic indicates that the data are not normally distributed. 
        Note: Note that this test usually works for a large enough number of data samples (>2000) 
    '''
    print('Analyzing TRAINING residuals...')
    print(' ADF Stationarity Test: ')
    adf_result = adfuller(residual_train.flatten())
    # print(adf_result[1])
    if (adf_result[1] <= 0.05):
        print("     > The residuals are stationary: YES")
    else:
        print("     > The residuals are stationary: NO")

    print(' D’Agostino-Pearson Test: ')
    DAP_result = stats.normaltest(residual_train.ravel())
    if (DAP_result[1] <= 0.05):
        print("     > The residuals are Gaussian: NO")
    else:
        print("     > The residuals are Gaussian: YES")

    print(' Jarque-Bera Test: ')
    JB_result = stats.jarque_bera(residual_train)
    if (JB_result[1] <= 0.05):
        print("     > The residuals are Gaussian: NO")
    else:
        print("     > The residuals are Gaussian: YES")

    return None


