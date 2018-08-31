
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    return experiment


def simulate_load_price(experiment):

    N = experiment['N']
    city = experiment['city']
    alpha = experiment['alpha']
    epsilon_D = experiment['epsilon_D']
    epsilon_P = experiment['epsilon_P']
    omega = experiment['omega']
    kappa = experiment['kappa']
    beta = experiment['beta']
    T = experiment['T']

    P = np.zeros((T,1))
    L = np.zeros((T,N))
    total_L = np.zeros((T,1))

    phi = city.values # convert from pandas to numpy array
    L[0,:] = phi[0,:] # set 1st L value from phi
    total_L[0] = np.sum(L[0,:])

    for t in range(1,T):
        L_hat = total_L[t-1] # persistance forecast
        P[t] = beta * (alpha - L_hat)**epsilon_P
        L[t,:] =  kappa*(omega*(P[t]**epsilon_D)) + (1 - kappa) * phi[t, :]
        total_L[t] = np.sum(L[t,:])
    P[0] = P[1]
    total_L[0] = total_L[1]

    experiment['P'] = P
    experiment['L'] = L
    experiment['total_L'] = total_L

    return experiment


def output_results(experiment):

    P = experiment['P']
    total_L = experiment['total_L']

    plt.figure(1)
    plt.subplot(211)
    plt.plot(P[3:], color='darkorange')
    plt.ylim(1, 60)
    plt.ylabel('Price (USD)')
    plt.subplot(212)
    plt.plot(total_L)
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    plt.ylim(1, 40000)

    return None