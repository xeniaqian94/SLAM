"""
Class for computing various metrics on a data set with a BayesNet Node object.
"""
import logging

import numpy as np

EPSILON = 1e-16

MAP_ACCURACY_KEY = 'map_accuracy'
AUC_KEY = 'auc'
LOGLI_KEY = 'logli'
D_PRIME_KEY = 'd_prime'
NAIVE_KEY = 'naive'
METRICS_KEYS = {NAIVE_KEY, LOGLI_KEY, MAP_ACCURACY_KEY, AUC_KEY, D_PRIME_KEY}

LOGGER = logging.getLogger(__name__)


def auc_helper(data, prob_true):
    """ Compute AUC (area under ROC curve) as a function of binary data values and predicted
    probabilities.  If data includes only positive or only negative labels, returns np.nan.

    :param np.ndarray[bool] data: binary data values (positive/negative class labels).
    :param np.ndarray[float] prob_true: probability of positive label
    :return: area under ROC curve
    :rtype: float
    """
    if len(prob_true) != len(data):
        raise ValueError('prob_true and data must have the same length')

    prob_true, data = Metrics._check_finite(prob_true, data)
    sorted_idx = np.argsort(prob_true)[::-1]
    sorted_prob_true = prob_true[sorted_idx]
    unique_prob_true_idx = np.append(np.flatnonzero(np.diff(sorted_prob_true)),
                                     len(sorted_prob_true) - 1)
    x = data[sorted_idx]
    not_x = np.logical_not(x)

    # Compute cumulative sums of true positives and false positives.
    tp = np.cumsum(x)[unique_prob_true_idx].astype(float)
    fp = np.cumsum(not_x)[unique_prob_true_idx].astype(float)

    # The i'th element of tp (fp) is the number of true (false) positives
    # resulting from using the i'th largest rp as a threshold. That is,
    # we predict correct if a response's rp is >= sorted_prob_true[i].
    # We want the first element to correspond to a threshold sufficiently
    # high to yield no predictions of correct. The highest rp qualifies
    # as this highest threshold if its corresponding response is incorrect.
    # Otherwise, we need to add an artificial "highest threshold" at the
    # beginning that yields 0 true positives and 0 false positives.
    if tp[0] != 0.0:
        tp = np.append(0.0, tp)
        fp = np.append(0.0, fp)

    # Calculate true positive rate and false positive rate.
    # This requires at least 1 correct and 1 incorrect response.
    if not tp[-1]:
        return np.nan
    tpr = tp / tp[-1]

    if not fp[-1]:
        return np.nan
    fpr = fp / fp[-1]

    return np.trapz(tpr, fpr)


