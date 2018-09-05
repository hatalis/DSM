
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

    experiment['city'] = city
    experiment['total_load'] = total_load
    print(np.max(total_load))
    return experiment


def simulate_load_price(experiment):

    N = experiment['N']
    phi = experiment['city'].values
    alpha = experiment['alpha']
    epsilon_D = experiment['epsilon_D']
    L_target = experiment['L_target']
    kappa = experiment['kappa']
    beta = experiment['beta']
    T = experiment['T']

    L_hat_period = 300
    P_target = L_hat_period/L_target
    epsilon_P = np.log(P_target/beta) / np.log(alpha - L_hat_period)

    P = np.zeros((T,1))
    L = np.zeros((T,N))
    L_total = np.zeros((T,1))

    for t in range(1,T):
        # SO defines price first
        L_hat = load_forecast(t, L_total, P)
        P[t] = beta * ((alpha - L_hat)**epsilon_P)
        # individual household defines load from DSM+phi
        L[t,:] =  kappa*(phi[t, :]*(P[t]**epsilon_D)) + (1 - kappa) * phi[t, :]
        L_total[t] = np.sum(L[t,:])
        # if total load > alpha, shed load
        if L_total[t] > alpha:
            L_total[t] = alpha-1

    experiment['P_target'] = P_target
    experiment['L_target'] = L_target
    experiment['epsilon_P'] = epsilon_P
    experiment['P'] = P/100 # convert cents to USD
    experiment['L'] = L
    experiment['L_total'] = L_total

    return experiment


def load_forecast(t, L_total, P):
    L_hat = float(L_total[t-1]) # persistance forecast
    return L_hat


def output_results(experiment):

    P = experiment['P']
    L_total = experiment['L_total']
    epsilon_P = experiment['epsilon_P']
    L_target = experiment['L_target']
    P_target = experiment['P_target']

    print('Target Load =', L_target)
    print('Target Price =', P_target)
    print('Elasticity of price =',epsilon_P)
    print('=========================')
    print('Actual Price =', np.mean(P))
    print('Actual Load =', np.mean(L_total))


    plt.figure(1)
    plt.subplot(211)
    plt.plot(P, color='darkorange')
    # plt.ylim(1, 60)
    plt.ylabel('Price (USD/kWh)')
    plt.subplot(212)
    plt.plot(L_total)
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    # plt.ylim(1, 40000)

    return None