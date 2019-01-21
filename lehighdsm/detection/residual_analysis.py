
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import statsmodels.stats.diagnostic as smd
import statsmodels.api as sm

def residual_analysis(experiment):

    L_total = experiment['L_total']
    N_train = experiment['N_train']
    N_test = experiment['N_test']
    L_test_prediction = experiment['L_test_prediction']
    L_train_prediction = experiment['L_train_prediction']
    L_train_actual = experiment['L_train_actual']
    L_test_actual = experiment['L_test_actual']


    residual_train = L_train_actual - L_train_prediction
    residual_test  = L_test_actual - L_test_prediction
    residual_train = residual_train[48:]
    var = residual_train.var()
    mu = residual_train.mean()
    sigma = residual_train.std()
    residual_train = (residual_train-mu)/sigma

    plt.figure()
    plt.plot(residual_train)
    plt.title('Standarized Training Residuals')
    plt.xlabel('Time (hr)')
    plt.ylabel('Standarized Load Residual')

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
    print(adf_result[1])
    if (adf_result[1] <= 0.05):
        print("     > The residuals are stationary.")
    else:
        print("     > The residuals are NOT stationary.")

    print(' D’Agostino-Pearson Test: ')
    DAP_result = stats.normaltest(residual_train.ravel())
    if (DAP_result[1] <= 0.05):
        print("     > The residuals are NOT Gaussian.")
    else:
        print("     > The residuals are Gaussian.")

    print(' Jarque-Bera Test: ')
    JB_result = stats.jarque_bera(residual_train)
    if (JB_result[1] <= 0.05):
        print("     > The residuals are NOT Gaussian.")
    else:
        print("     > The residuals are Gaussian.")



    print('\nAnalyzing TESTING residuals...')
    print(' ADF Stationarity Test: ')
    adf_result = adfuller(residual_test.ravel())
    if (adf_result[1] <= 0.05):
        print("     > The residuals are stationary.")
    else:
        print("     > The residuals are NOT stationary.")

    print(' D’Agostino-Pearson Test: ')
    DAP_result = stats.normaltest(residual_test.ravel())
    if (DAP_result[1] <= 0.05):
        print("     > The residuals are NOT Gaussian.")
    else:
        print("     > The residuals are Gaussian.")

    print(' Jarque-Bera Test: ')
    JB_result = stats.jarque_bera(residual_test)
    if (JB_result[1] <= 0.05):
        print("     > The residuals are NOT Gaussian.")
    else:
        print("     > The residuals are Gaussian.")

    # Running and interpreting a Ljung-Box test
    # ljung_box = smd.acorr_ljungbox(residual_train, lags=24)
    # if any(ljung_box[1] < 0.05):
    #     print("     > The residuals are autocorrelated.")
    # else:
    #     print("     > The residuals are not autocorrelated.")

    # y_train, l1 = stats.boxcox(100 + residual_train.ravel())
    # y_test, l2 = stats.boxcox(100 + residual_test.ravel())
    # plot_acf(y_train, lags=48)
    # plot_pacf(y_test, lags=48)

    plot_acf(residual_train-mu,lags=72)
    plot_pacf(residual_train-mu, lags=72)


    # temp=zca_whitening(residual_train.T)
    # # print(temp)
    # plt.figure()
    # plt.plot(temp)
    #
    # plot_acf(temp,lags=48)
    # plot_pacf(temp, lags=48)

    # Q-Q Plot
    plt.figure()
    plt.subplot(2, 1, 1)
    stats.probplot(residual_train.ravel(), dist="norm", plot=plt, sparams=(0, sigma))
    plt.title('Q-Q Plot of Training Residuals')
    plt.subplot(2, 1, 2)
    stats.probplot(residual_test.ravel(), dist="norm", plot=plt, sparams=(0, sigma))
    plt.title('Q-Q Plot of Test Residuals')







    return None


def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

