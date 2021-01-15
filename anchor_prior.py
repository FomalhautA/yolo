import numpy as np
import random
import time

from util import DataUtils, iou


def dist_func(v1, v2, epsilon=1e-6, mode='iou'):
    """

    :param v1: n-d numpy array
    :param v2: n-d numpy array, same shape with v1
    :param epsilon:
    :param mode:
    :return:
    """
    if mode == 'euclid':
        sm = np.sum((v1-v2)**2)

        return (sm+epsilon)**0.5
    elif mode == 'iou':
        x1, y1 = v1
        x2, y2 = v2

        box1 = [960-x1/2, 600-y1/2, 960+x1/2, 600+y1/2]
        box2 = [960-x2/2, 600-y2/2, 960+x2/2, 600+y2/2]

        return 1 - iou(box1, box2, encode=False)
    else:
        raise Exception('Unrecognized mode {}'.format(mode))


def merge_func(clusters):
    """

    :param clusters: list of list of n-d numpy arrays
    :return: [k lists of centers]
    """
    res = []
    for cluster in clusters:
        res.append(np.mean(np.array(cluster), axis=0))

    return res


def diff_func(centers1, centers2, epsilon=1e-6):
    """

    :param centers1:
    :param centers2:
    :param epsilon:
    :return:
    """
    k = len(centers1)
    return (np.sum((np.array(centers1) - np.array(centers2)) ** 2) * 1. / k + epsilon) ** 0.5


def kmeans(inputs, k=5, threshold=1e-1, mode='iou'):
    """

    :param inputs: list of n-d numpy array
    :param k: how many clusters needed
    :param threshold:
    :param mode:
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
                dists[i] = dist_func(c, v, mode=mode)
            idx = np.argmin(dists)
            clusters[idx].append(v)

        temp = merge_func(clusters)

        diffs = diff_func(temp, centers)
        if diffs < threshold:
            break

        centers = temp

    print('{} epochs runned.'.format(epochs))

    return centers


def pipline(labels, mode='iou'):
    """

    :param labels:
    :param mode:
    :return:
    """
    inputs = [np.array([item[2] - item[0], item[3] - item[1]]) for item in labels]
    centers = kmeans(inputs, k=5, threshold=0.5, mode=mode)

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
        pipline(labels, mode='iou')
        end = time.time()
        print('Time consumed {}s'.format(round(end-start, 0)))
