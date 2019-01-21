# -*- coding: utf-8 -*-
"""

@author: Kostas Hatalis
"""

import numpy as np


def evaluate_PI_results(experiment):
    # load in quantile estimates and evaluation parameters
    n_test = experiment['N_test']
    y_test = experiment['y_test_true']
    n_tau = experiment['n_tau']
    tau = experiment['tau']
    q_hat = experiment['q_hat']
    alpha = experiment['alpha']


    # calculate evaluation scores
    QS = quantileScore(q_hat, tau, n_tau, n_test, y_test)
    [IS, sharp_score, PINC] = intervalScore(q_hat, tau, n_tau, n_test, y_test)
    ACE,PICP = coverageScore(q_hat, tau, n_tau, n_test, y_test)

    # save scores to dictionary and return
    # Note: QS, IS, SHARP is normalized by alpha (max load)
    experiment['QS'] = QS/alpha
    experiment['IS'] = IS/alpha
    experiment['PINC'] = PINC
    experiment['ACE'] = ACE
    experiment['PICP'] = PICP
    experiment['SHARP'] = sharp_score/alpha

    return experiment


# ------------------------------------------------------------------------------

def crossScore(q_hat, n_tau, n_test):
    # count times q_{m} > q_{m+1}
    cross = 0
    for i in range(0, n_test):
        for m in range(0, n_tau - 1):
            if q_hat[i, m] > q_hat[i, m + 1]:
                cross += 1

    return cross


# ------------------------------------------------------------------------------

def coverageScore(q_hat, tau, n_tau, n_test, y_test):
    n_pi = int(n_tau / 2)  # number of prediction intervals
    PICP = np.zeros((n_pi, 1))
    ACE = np.zeros((n_pi, 1))

    # calculate PICP
    PINC = [0] * n_pi
    for m in range(0, n_pi):
        PINC[m] = tau[-(m + 1)] - tau[m]
        PINC[m] = PINC[m]

        # calculate PICP and then ACE
    for m in range(0, n_pi):
        LB = q_hat[:, m]
        UB = q_hat[:, -(m + 1)]
        c = 0
        for i in range(0, n_test):
            if y_test[i] <= UB[i] and y_test[i] >= LB[i]:
                c += 1
        PICP[m] = (1 / n_test) * c
        # ACE[m] = abs(PICP[m]-PINC[m])
        ACE[m] = abs(PICP[m] - PINC[m]) * 100

    # average q-scores from all PIs into a single score
    ACE = np.mean(ACE)

    # PICP = np.mean(PICP)

    return ACE,PICP


# ------------------------------------------------------------------------------

def intervalScore(q_hat, tau, n_tau, n_test, y_test):
    n_pi = int(n_tau / 2)  # number of prediction intervals
    interval_score = np.zeros((n_pi, 1))
    sharp_score = np.zeros((n_pi, 1))

    # calculate PICP
    PINC = [0] * n_pi
    for m in range(0, n_pi):
        PINC[m] = tau[-(m + 1)] - tau[m]

    # calculate interval score sharpness
    for m in range(0, n_pi):
        LB = q_hat[:, m]
        UB = q_hat[:, -(m + 1)]
        alpha = 1 - PINC[m]

        IS = np.zeros((n_test, 1))
        sharpness = np.zeros((n_test, 1))
        for i in range(0, n_test):
            L = LB[i]
            U = UB[i]
            delta = U - L
            y = y_test[i]
            sharpness[i] = delta
            if y < L:
                IS[i] = -2 * alpha * delta - 4 * (L - y)
            elif y > U:
                IS[i] = -2 * alpha * delta - 4 * (y - U)
            else:
                IS[i] = -2 * alpha * delta

        sharp_score[m] = np.mean(sharpness)
        interval_score[m] = np.mean(IS)

    # average q-scores from all PIs into a single score
    interval_score = np.mean(interval_score)
    # sharp_score = np.mean(sharp_score)

    return interval_score, sharp_score, PINC


# ------------------------------------------------------------------------------

def quantileScore(q_hat, tau, n_tau, n_test, y_test):
    qscore = np.zeros((n_tau, 1))

    # pinball function
    for m in range(0, n_tau):
        xq = np.zeros((n_test, 1))
        for i in range(0, n_test):
            if y_test[i] < q_hat[i, m]:
                xq[i] = (1 - tau[m]) * (q_hat[i, m] - y_test[i])
            else:
                xq[i] = tau[m] * (y_test[i] - q_hat[i, m])
        qscore[m] = np.mean(xq)

    # average q-scores from all quantiles into a single score
    qscore = np.mean(qscore)

    return qscore