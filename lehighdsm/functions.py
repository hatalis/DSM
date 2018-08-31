'''
This function creates 1 month worth of new load data using block bootstrap from a template dataset.
'''

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

def load(experiment):

    filename = experiment['filename']

    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M')
    raw_data = pd.read_csv(filename, parse_dates=[0], index_col=0, date_parser=dateparse)

    kWh = raw_data.resample('H').apply('sum') # convert data from kW to kWh

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

    # total.to_csv('city.csv', sep=',')

    return experiment



def simulate_load_price(experiment):

    N = experiment['N']
    city = experiment['city']
    alpha = experiment['alpha']
    epsilon_D = experiment['epsilon_D']
    epsilon_P = experiment['epsilon_P']
    total_load = experiment['total_load']
    kappa = experiment['kappa']
    beta = experiment['beta']

    T = 24*7
    P = np.zeros((T,1))
    L = np.zeros((T,N))
    total_L = np.zeros((T,1))

    # Simulate u = AR(1) with phi=-0.6
    ma = np.array([1])
    ar = np.array([1, 0.6])
    AR_object = ArmaProcess(ar, ma)
    u = AR_object.generate_sample(nsample=T)


    city = city.values # convert from pandas to numpy array

    L[0,:] = city[0,:]
    total_L[0] = np.sum(L[0,:])

    kappa = 0
    for t in range(1,T):
        L_hat = total_L[t-1]
        P[t] = beta * (alpha - L_hat)**epsilon_P + u[t]*0

        L[t,:] =  kappa*(100*(P[t]**epsilon_D)) + (1 - kappa) * city[t, :]
        total_L[t] = np.sum(L[t,:])



    P[0] = P[1]
    experiment['P'] = P
    experiment['L'] = L
    experiment['total_L'] = total_L

    return experiment