# !/usr/bin/env python


def features_length(features_list):
    sum = 0
    for _, size in features_list:
        sum += size
    return sum

def train(labeled_data, N):
    """

    :param labeled_data: labeled data
    :param N: number of iterations
    :return W: learnt weights
    """

