
import numpy as np
import pandas as pd

def create_homes(df, template, k, N_total):

    days, new_home = np.random.randint(1, 31, size=31), None

    for j in days:
        temp = np.array(df.loc[(df.index.day == j), template])  # pick random day worth of data
        new_home = np.append(new_home, temp)  # add random day to new series
    new_home = pd.DataFrame(data=new_home[1:], index=df.index, columns=[str(k)])  # convert to dataframe

    # print('Finished simulating home #{}'.format(k))

    return new_home


def simulate_city(experiment):

    N_total = experiment['N_total']
    raw_data = experiment['raw_data']
    N = experiment['N']
    homes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # list of template homes
    city = raw_data['A']  # we start simulating a city with 1 home

    # pick N random homes from list
    templates = np.random.choice(homes, size=N, replace=True)

    # simulate 1 month of data for each home, save to city dataframe
    for i in range(N-1):
        city = create_homes(raw_data, templates[i], i, N_total).join(city, how='inner')

    # sum the data together to get total load
    total_load = city.sum(axis=1)

    # total_load.to_csv('total_load.csv', sep=',')

    experiment['city'] = city
    experiment['total_load'] = total_load

    return experiment
