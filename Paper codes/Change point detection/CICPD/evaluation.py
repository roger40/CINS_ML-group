#!/usr/local/bin/python
# -*-coding:utf-8 -*-

import numpy as np


def computeFvalue(pv, rv, beta):
    '''
    :param pv: precision
    :param rv: recall
    :param beta: trade-off weight value can be 0.5 ,1, 2
    :return: fvalue
    '''
    temp = np.power(beta, 2)
    return (temp + 1) * pv * rv * 1.0 / (temp * pv + rv)


def Evaluation(p_c, groundEvent, s, numSnapshots, **kwargs):
    """Evaluate the performance of the designed method
    With the ground truth(external event information),we use two evaluation
    index named precision and recall to evaluate the performance of our method
    and other baseline method.

    precision defined as the proportion of estimated change point that
    occur within a given delay
    s of a known events.

    recall defined as the proportion of known events that occur within
    a delay s of an estimated
    change point.

    precision or recall is as normal when the given delay s is 0.

    Parameters
    ----------
    p_c: the potential change points detected by our method
    groundEvent: the real changePoints related to the external events
    s:delay delta
    kwargs: the parameters to be tuned,such as k、alpha

    Returns
    ----------
    precision :

    recall:
    """
    if len(p_c) == 0:
        print kwargs, "there is no point be detected..."
        return

    print p_c
    g_c = groundEvent

    inters = p_c.intersection(g_c)
    pg_n = len(inters)

    p_c = list(p_c)
    g_c = list(g_c)

    g_n = len(g_c)
    p_n = len(p_c)

    tempM = np.zeros((p_n, g_n))
    for i in range(p_n):
        for j in range(g_n):
            tempM[i, j] = np.abs((p_c[i] - g_c[j]))

    tempM = np.matrix(tempM)

    tempP = np.min(tempM, 1)
    tempP = tempP.reshape(1, tempP.shape[0]).tolist()[0]
    precision = 1.0 * len([item for item in tempP if item <= s]) / p_n

    tempR = np.min(tempM, 0)
    tempR = tempR.tolist()[0]
    recall = 1.0 * len([item for item in tempR if item <= s]) / g_n

    fpr = (p_n - pg_n) * 1.0 / (numSnapshots - g_n)

    fvalue = 0 if precision == 0 and recall == 0  else computeFvalue(precision,
                                                                     recall,
                                                                     1)
    print kwargs, " realAbnormal  %d thoughtAbnormal %d tr_abnormal %d " \
                  "so accuracy: %f  and  recall : %f and fpr: %f and fvalue : %f " % (
                      g_n, p_n, pg_n, precision, recall, fpr,
                      fvalue), 'Attention Attention Attention' if fvalue > 0.62 else  None

    return precision, recall, fvalue, fpr
