
import numpy as np
import matplotlib.pyplot as plt

def output_simulation_results(experiment):

    N_train = experiment['N_train']
    N_test = experiment['N_test']

    Phi_hat = experiment['Phi_hat']
    Phi_total = experiment['Phi_total']
    Phi_train = experiment['Phi_train']
    Phi_test = experiment['Phi_test']

    L_target = experiment['L_target']
    L_total = experiment['L_total']
    L_train = experiment['L_train']
    L_test = experiment['L_test']
    attack = experiment['attack']

    P_total = experiment['P_total']
    P_train = experiment['P_train']
    P_test = experiment['P_test']

    window = experiment['window']
    kappa = experiment['kappa']
    goal = experiment['goal']

    print('=========================')
    print('Target Load = {:,.2f}'.format(np.mean(L_target)))
    print('Actual Load = {:,.2f}'.format(np.mean(L_total)))
    # print('RMSE of Target vs Actual = ',np.sqrt(mean_squared_error(L_target, L_total)))
    print('=========================')
    print('kappa = ', kappa)
    print('goal = ', goal)
    print('=========================')

    '''
    Plot of attack on test data.
    '''
    # plt.figure(figsize=(4.5,6.5))
    # plt.subplot(211)
    # plt.plot(attack, color='red')
    # plt.ylabel('Attack Level')
    # plt.grid(alpha=0.4)
    # plt.tight_layout()
    # plt.title('Attacked Test Set')
    #
    # plt.subplot(212)
    # plt.subplots_adjust(bottom=0.1)
    # plt.plot(L_test,'red')
    # plt.plot(Phi_test,'black')
    # plt.xlabel('Time (hr)')
    # plt.ylabel('Total Load (kWh)')
    # plt.ylim(1, 1_000)
    # plt.legend(['Attacked Load','Nominal Load'],fancybox=True,framealpha=1.0,
    #            shadow=True,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.01))
    # plt.grid(alpha=0.4)

    '''
    Plot testing data simulation.
    '''
    plt.figure(figsize=(4.5,6.5))
    plt.subplot(211)
    plt.plot(P_test, color='darkorange')
    plt.ylabel('Price (USD/kWh)')
    plt.grid(alpha=0.4)
    plt.title('Testing Set')
    plt.tight_layout()

    plt.subplot(212)
    plt.subplots_adjust(bottom=0.1)
    plt.plot(L_test)
    plt.plot(Phi_test,'black')
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    plt.ylim(1, 1_000)
    plt.axhline(y=np.mean(L_target[N_train:]), color='r')
    plt.axhline(y=np.mean(L_test), linestyle='--', color='blue')
    plt.legend(['Observed Load','Base Load','Target Load','Mean Load'],fancybox=True,framealpha=1.0,
               shadow=True,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.01))
    plt.grid(alpha=0.4)

    '''
    Plot training data simulation.
    '''
    plt.figure(figsize=(12,6.5))
    plt.subplot(211)
    plt.plot(P_train, color='darkorange')
    plt.ylabel('Price (USD/kWh)')
    plt.grid(alpha=0.4)
    plt.title('Training Set')
    plt.tight_layout()

    plt.subplot(212)
    plt.subplots_adjust(bottom=0.1)
    plt.plot(L_train)
    plt.plot(Phi_train,'black')
    plt.xlabel('Time (hr)')
    plt.ylabel('Total Load (kWh)')
    plt.ylim(1, 1_000)
    plt.axhline(y=np.mean(L_target[:N_train]), color='r')
    plt.axhline(y=np.mean(L_train), linestyle='--', color='blue')
    plt.legend(['Observed Load','Base Load','Target Load','Mean Load'],fancybox=True,framealpha=1.0,
               shadow=True,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.01))
    plt.grid(alpha=0.4)

    return None