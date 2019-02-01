
import pandas as pd

def load_data(experiment):

    filename = 'data/home_all.csv'
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %H:%M')
    raw_data = pd.read_csv(filename, parse_dates=[0], index_col=0, date_parser=dateparse)
    kWh = raw_data.resample('H').apply('sum')/60 # convert data from kW to kWh (average the power within 1 hour)
    experiment['raw_data'] = kWh
    experiment['N_total'] = experiment['N_train'] + experiment['N_test']
    experiment['filename'] = filename

    return experiment