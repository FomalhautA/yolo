import numpy as np
import random
import time

from util import DataUtils


def distance(v1, v2, epsilon=1e-6):
    """

    :param v1: n-d numpy array
    :param v2: n-d numpy array, same shape with v1
    :param epsilon:
    :return:
    """
    sm = np.sum((v1-v2)**2)

    return (sm+epsilon)**0.5


def mean_clusters(clusters):
    """

    :param clusters: list of list of n-d numpy arrays
    :return: [k lists of centers]
    """
    res = []
    for cluster in clusters:
        res.append(np.mean(np.array(cluster), axis=0))

    return res


def kmeans(inputs, k=5, epsilon=1e-6, threshold=1e-1):
    """

    :param inputs: list of n-d numpy array
    :param k: how many clusters needed
    :param epsilon:
    :param threshold:
    :return: list of k n-d numpy array
    """
    epochs = 0
    centers = random.sample(inputs, k)
    print('Original Centers ...')
    for c in centers:
        print(list(c.astype(int)))

    while True:
        epochs += 1
        clusters = [[] for i in range(k)]
        dists = [0 for i in range(k)]

        for v in inputs:
            for i, c in enumerate(centers):
                dists[i] = distance(c, v)
            idx = np.argmin(dists)
            clusters[idx].append(v)

        temp = mean_clusters(clusters)

        diff = (np.sum((np.array(temp) - np.array(centers))**2) * 1./k + epsilon)**0.5
        if diff < threshold:
            break

        centers = temp

    print('{} epochs runned.'.format(epochs))

    return centers


def pipline(lebels):
    """

    :param lebels:
    :return:
    """

    inputs = [np.array([item[2] - item[0], item[3] - item[1]]) for item in labels]

    centers = kmeans(inputs, k=5, threshold=0.5)

    for c in centers:
        print(list(c.astype(int)))


if __name__ == '__main__':
    label_file = '../Data/object-detection-crowdai/labels.csv'
    data_folder = '../Data/object-detection-crowdai'
    data_utils = DataUtils(label_file=label_file, data_folder=data_folder)
    labels = data_utils.labels

    random.seed()
    for i in range(10):
        print('*************************************')
        start = time.time()
        pipline(labels)
        end = time.time()
        print('Time consumed {}s'.format(round(end-start, 0)))
