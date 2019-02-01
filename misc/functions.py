
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load(experiment):

    filename = experiment['filename']
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M')
    raw_data = pd.read_csv(filename, parse_dates=[0], index_col=0, date_parser=dateparse)
    kWh = raw_data.resample('H').apply('sum')/60 # convert data from kW to kWh (average the power within 1 hour)
    experiment['raw_data'] = kWh

    return experiment


def create_homes(df, template, k):
    days, new_home = np.random.randint(1, 31, size=31), None

    for j in days:
        temp = np.array(df.loc[(df.index.day == j), template])  # pick random day worth of data
        new_home = np.append(new_home, temp)  # add random day to new series
    new_home = pd.DataFrame(data=new_home[1:], index=df.index, columns=[str(k)])  # convert to dataframe

    # print('Finished simulating home #{}'.format(k))

    return new_home


def simulate_city(experiment):

    raw_data = experiment['raw_data']
    N = experiment['N']
    homes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # list of template homes
    city = raw_data['A']  # we start simulating a city with 1 home

    # pick N random homes from list
    templates = np.random.choice(homes, size=N, replace=True)

    # simulate 1 month of data for each home, save to city dataframe
    for i in range(N-1):
        city = create_homes(raw_data, templates[i], i).join(city, how='inner')

    # sum the data together to get total load
    total_load = city.sum(axis=1)

    # total_load.to_csv('total_load.csv', sep=',')

    experiment['city'] = city
    experiment['total_load'] = total_load

    return experiment

# from statsmodels.graphics.tsaplots import plot_acf

def simulate_load_price(experiment):

    N = experiment['N']
    phi = experiment['city'].values
    epsilon_D = experiment['epsilon_D']
    window = experiment['window']

    method = experiment['method']
    L_target = experiment['L_target']
    kappa = experiment['kappa']
    T = experiment['T']

    goal = 0

    L_target = 200
    kappa = 0.0
    T = 48


    P = np.zeros((T,1))
    L = np.zeros((T,N))
    L_total = np.zeros((T,1))
    L_hat = np.zeros((T, 1))
    L_target = np.ones((T,1))*L_target

    # attack parameters
    # pa = 0.0
    # attack, attack_prob, = [0, 1], [1-pa, pa]
    # attack_outcome = np.random.choice(attack, size=[T,N], p=attack_prob)

    attack = np.zeros((T,1))
    for t in range(23,T):
        attack[t] = attack[t-1] + 10

    # attack[24] = 250
    # attack[30] = 200
    # attack[46] = 300

    Phi_Total = np.sum(phi,axis = 1)[:T]
    L_total[0] = Phi_Total[0]
    for t in range(1,T):
        # 1. DEFINE PRICE
        if t > window:
            # L_hat[t] = load_forecast(t, L_total[:t], P[:t], method, window)

            L_adjusted = L_target[t] + goal*(L_target[t-1]-L_total[t-1])
            if L_adjusted < 0:
                L_adjusted = 10# L_target[t]
            P[t] = (L_adjusted/Phi_Total[t-1])**(1/epsilon_D)

            # P[t] = (L_target[t] / Phi_Total[t]) ** (1 / epsilon_D)

        # 2. DEFINE LOAD
        L[t, :] =  kappa*(phi[t, :]*(P[t]**epsilon_D)) + (1 - kappa) * phi[t, :]

        # # random attack
        # amount = 0.00
        # L[t, :] = L[t, :] * (1+attack_outcome[t, :]*amount)
        # # attack on midnight
        # if t % 24 == 0:
        #     L[t, :] = L[t, :] * 0

        # if total load > alpha, shed load
        L_total[t] = np.sum(L[t, :])
        # if L_total[t] > alpha:
        #     L_total[t] = alpha-1
        L_total[t] = L_total[t]  + attack[t]

    print('kappa = ', kappa)
    print('goal = ',goal)
    print('L_target = ', np.mean(L_target))
    print('Mean L = ', np.mean(L_total))

    experiment['Phi_Total'] = Phi_Total
    experiment['L_target'] = L_target
    experiment['P'] = P/100 # convert cents to USD
    experiment['L'] = L
    experiment['L_total'] = L_total
    experiment['L_hat'] = L_hat
    experiment['attack'] = attack
    return experiment

'''
# OLD MODEL
def simulate_load_price(experiment):

    N = experiment['N']
    phi = experiment['city'].values
    alpha = experiment['alpha']
    epsilon_D = experiment['epsilon_D']
    epsilon_P = experiment['epsilon_P']
    L_hat_period = experiment['L_hat_period']
    L_target = experiment['L_target']
    kappa = experiment['kappa']
    beta = experiment['beta']
    T = experiment['T']
    total_load = experiment['total_load']
    method = experiment['method']
    window = experiment['window']

    # plot_acf(total_load.diff(periods=1).values[1:], alpha=0.05, lags=48)
    # plt.show()


    P_target = L_hat_period/L_target
    if epsilon_P == None:
        epsilon_P = np.log(P_target/beta) / np.log(alpha - L_hat_period)

    P = np.ones((T,1))*0
    L = np.zeros((T,N))
    L_total = np.zeros((T,1))
    L_hat = np.zeros((T, 1))
    P_perfect = np.ones((T, 1))*P_target

    # P[100:150] = 0.01
    # attack parameters
    pa = 0.4
    attack, attack_prob, = [0, 1], [1-pa, pa]
    attack_outcome = np.random.choice(attack, size=[T,N], p=attack_prob)

    for t in range(1,T):
        # SO defines price first
        if t > window:
            L_hat[t] = load_forecast(t, L_total[:t], P[:t], method, window)
            P[t] = beta * ((alpha - L_hat[t])**epsilon_P)

        # individual household defines load from DSM+phi
        L[t, :] =  kappa*(phi[t, :]*(P[t]**epsilon_D)) + (1 - kappa) * phi[t, :]
        # L[t, 0] =  kappa*(L_hat[t]*(P[t]**epsilon_D)) + (1 - kappa) *L_hat[t]

        # # random attack
        amount = 0.00
        L[t, :] = L[t, :] * (1+attack_outcome[t, :]*amount)
        # # attack on midnight
        # if t % 24 == 0:
        #     L[t, :] = L[t, :] * 0

        # if total load > alpha, shed load
        L_total[t] = np.sum(L[t, :])
        if L_total[t] > alpha:
            L_total[t] = alpha-1

    for t in range(window,T):
        P_perfect[t] = beta * ((alpha - L_total[t])**epsilon_P)

    experiment['P_target'] = P_target
    experiment['L_target'] = L_target
    experiment['epsilon_P'] = epsilon_P
    experiment['P'] = P # convert cents to USD
    experiment['L'] = L # individual user loads
    experiment['L_total'] = L_total
    experiment['L_hat'] = L_hat
    experiment['P_perfect'] = P_perfect

    return experiment
'''

from statsmodels.tsa.statespace.sarimax import SARIMAX
def load_forecast(t, L_total, P, method, window):

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
            else: # SARIMAX
                # print(t)
                model = SARIMAX(L_total[-1-window:],  exog=P[-1-window:], order=(1,1,1), seasonal_order=(1,1,0,24),
                                enforce_invertibility=False,enforce_stationarity=False)
                model_fit = model.fit(disp=False)
                L_hat = model_fit.forecast(exog=P[-1].reshape((1, 1)))
        else:
            L_hat = float(L_total[-1]) # persistance forecast

    return L_hat



from sklearn.metrics import mean_squared_error
def output_results(experiment):

    Phi_Total = experiment['Phi_Total']
    L_total = experiment['L_total']
    L_target = experiment['L_target']
    L_hat = experiment['L_hat']
    P = experiment['P']
    # P_target = experiment['P_target']/100
    # P_perfect = experiment['P_perfect']/100
    window = experiment['window']
    attack = experiment['attack']

    # print('=========================')
    # print('Target Load = {:,.2f}'.format(np.mean(L_target)))
    # print('Actual Load = {:,.2f}'.format(np.mean(L_total)))
    # print('RMSE of Load Forecast vs Observed = ',np.sqrt(mean_squared_error(L_total[window:], L_hat[window:])))
    # print('=========================')
    # print('Target Price = {:,.4f}'.format(P_target))
    # print('Elasticity of Price = {:,.2f}'.format(epsilon_P))
    # print('Mean RTP = = {:,.4f}'.format(np.mean(P)))
    # print('RMSE of RTP vs Ideal Price = ',np.sqrt(mean_squared_error(P_perfect[window:], P[window:])))
    # print('=========================')

    '''
    Plot attack
    '''
    plt.figure(figsize=(4.5,6.5))
    plt.subplot(211)
    plt.plot(attack, color='red')
    plt.ylabel('Attack Level')
    plt.grid(alpha=0.4)


    plt.tight_layout()
    plt.subplot(212)
    plt.subplots_adjust(bottom=0.1)
    plt.plot(L_total,'red')
    plt.plot(Phi_Total,'black')


    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    plt.ylim(1, 1_000)



    plt.legend(['Attacked Load','Nominal Load'],fancybox=True,framealpha=1.0,
               shadow=True,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.01))
    #  plt.axvline(x=window, color='black')

    plt.grid(alpha=0.4)






    plt.figure(figsize=(4.5,6.5))
    plt.subplot(211)
    plt.plot(P, color='darkorange')
    plt.ylabel('Price (USD/kWh)')
    plt.grid(alpha=0.4)
    # plt.legend(['Estimated Price','Ideal Price'],fancybox=True,framealpha=1.0,
    #            shadow=True,ncol=3,loc='upper center', bbox_to_anchor=(0.5, 1.08))
    # plt.ylim(0, 0.08)
    # plt.axvline(x=window, color='black')
    # plt.title('Elasticity of price = {:,.2f}'.format(epsilon_P))
    # plt.axhline(y=P_target, color='r')
    # plt.axhline(y=np.mean(P), linestyle='--', color='green')
    # plt.subplots_adjust(hspace=10)
    # plt.subplots_adjust(left=0.15)

    plt.tight_layout()
    plt.subplot(212)
    plt.subplots_adjust(bottom=0.1)
    plt.plot(L_total)
    plt.plot(Phi_Total,'black')
    # plt.plot(L_target, 'red')
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    plt.ylim(1, 1_000)

    plt.axhline(y=np.mean(L_target), color='r')
    plt.axhline(y=np.mean(L_total), linestyle='--', color='blue')
    plt.legend(['Observed Load','Base Load','Target Load','Mean Load'],fancybox=True,framealpha=1.0,
               shadow=True,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.01))
    #  plt.axvline(x=window, color='black')

    plt.grid(alpha=0.4)

    return None