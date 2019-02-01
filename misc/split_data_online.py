
def split_data_online(experiment):

    L_total = experiment['L_total']
    N_train = experiment['N_train']
    N_test = experiment['N_test']

    L_total = L_total[3:]
    N_train = N_train -3

    experiment['L_total'] = L_total
    experiment['N_train'] = N_train

    L_test_actual = L_total[N_train:]
    L_test_actual = L_test_actual.reshape((N_test, 1))

    L_train_actual = L_total[:N_train]
    L_train_actual = L_train_actual.reshape((N_train, 1))

    experiment['L_train_actual'] = L_train_actual
    experiment['L_test_actual'] = L_test_actual

    return experiment