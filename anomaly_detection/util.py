import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import average_precision_score


def evaluate_detector(detector, X, y):
    """Run all metrics on for detector results on X"""
    scores = get_predictions(detector, X)
    return scores, get_score_dict(scores=scores, y_true=y)


def get_score_dict(scores, y_true):
    return {**get_auc_pr_score(scores=scores, y=y_true),
            **get_auc_roc_score(scores=scores, y=y_true)}


def get_predictions(detector, X):
    """get anomaly score predictions for X"""
    return pd.Series(detector.score_samples(X), index=X.index)


def get_mean_rank(scores, y):
    """Calculates mean and std rank of all fraud cases from _scores.csv df"""
    data = pd.DataFrame({'scores': scores, 'y': y})
    if data['y'].dtype in ['str', 'O']:
        data['y'] = ~data['y'].dropna().str.startswith('NonFraud')
    data = data.sort_values(['scores', 'y']).dropna().reset_index(drop=True)
    fraud_ranks = data[data['y']].index.values + 1

    return {f'fraud_ranks': list(fraud_ranks),
            f'rank_mean': fraud_ranks.mean(),
            f'rank_std': fraud_ranks.std()}


def get_max_rank(scores, y):
    """
    Calculates the rank of the least anomalous fraud case from _scores.csv df
    Shows where threshold would need to be / how many values need to be labeled as anomalous, to find all frauds.
    """
    max_fraud_eval = scores[y[y != 'NonFraud'].index].max()  # calculate max fraud score for min_detect
    return {f'max_fraud': max_fraud_eval,
            f'min_detect': (pd.Series(scores) <= max_fraud_eval).value_counts()[1]}


def get_auc_pr_score(scores, y, bin_labels=True):
    """Calculates auc_pr_score for fraud cases from _scores.csv df"""
    if not bin_labels:
        bin_labels = lambda y: (y != 'NonFraud').astype(int)  # binarize label for auc scores
        y = bin_labels(y)
    return {f'auc_pr': average_precision_score(y_true=y, y_score=-1 * scores)}


def get_auc_roc_score(scores, y, bin_labels=True):
    """Calculates auc_pr_score for fraud cases from _scores.csv df"""
    if not bin_labels:
        bin_labels = lambda y: (y != 'NonFraud').astype(int)  # binarize label for auc scores
        y = bin_labels(y)
    return {f'auc_roc': roc_auc_score(y_true=y, y_score=-1 * scores)}


def gen_confusion_matrix(detector, X, y, n_anomalies):
    """
    Runs detector on X and converts anomaly scores to labels, so that n_anomalies samples
    with lowest (highest negative) anomaly scores are classified as anomaly
    """
    scores = get_predictions(detector, X)

    # classify top n_anomalies as anomalous
    anomaly_threshold = scores[np.argpartition(scores, n_anomalies)[n_anomalies]]
    pred = (scores <= anomaly_threshold).astype(int)

    return confusion_matrix((y != "NonFraud").astype("int"), pred)
