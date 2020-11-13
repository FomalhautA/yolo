import collections
import copy
import csv
import cv2
import os
import random

import numpy as np
import pandas as pd
from PIL import Image


# from threadsafe_iter import threadsafe_generator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataUtils:
    def __init__(self, label_file, data_folder, model_dir='./model', train_batch_size=2, pred_batch_size=1):
        """

        :param label_file:
        :param data_folder:
        """
        self.model_dir = model_dir
        self.ORIG_CHANNEL = 1
        self.grid_resolution = 24
        self.image_width = 1920
        self.image_height = 1200
        self.width_box_num = self.image_width // self.grid_resolution
        self.height_box_num = self.image_height // self.grid_resolution
        self.anchor_priors = [[754, 625], [434, 387], [257, 203], [141, 104], [59, 55]]
        self.box_num_per_grid = len(self.anchor_priors)
        self.classes = {'Car': 0, 'Truck': 1, 'Pedestrian': 2}
        self.reverse_classes = {0: 'Car', 1: 'Truck', 2: 'Pedestrian'}
        self.label_file = label_file
        self.data_folder = data_folder
        self.labels = load_data(label_file)
        self.dict_labels = self.labels_to_dict(self.labels)
        self.train_fnames, self.val_fnames = self.data_partition(self.dict_labels, tv_ratio=0.1, ratio=1,
                                                                 random_seed=1, save_to_file=False)
        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.train_data_gen = self.train_data_generator(self.train_fnames, self.data_folder,
                                                        batch_size=self.train_batch_size, shuffle=True)
        self.val_data_gen = self.eval_data_generator(self.val_fnames, self.data_folder,
                                                     batch_size=self.train_batch_size)
        self.pred_data_gen = self.pred_data_generator(self.val_fnames, self.data_folder,
                                                      batch_size=self.pred_batch_size)

        self.val_data_gt = self.fetch_groundtruth(self.val_fnames)
        if len(self.val_data_gt) % self.train_batch_size:
            counts = len(self.val_data_gt) // self.train_batch_size * self.train_batch_size
            self.val_data_gt = self.val_data_gt[0:counts]

    @staticmethod
    def labels_to_dict(labels):
        re = collections.defaultdict(list)
        for label in labels:
            re[label[4]].append(label)

        return re

    def get_image(self, fname, folder, aug=True):

        temp_name = fname
        if not aug:
            temp_name = fname.split('-aug')[0] if fname.__contains__('-aug') else fname

        im = Image.open(os.path.join(folder, temp_name), 'r')

        return np.array(im.convert('RGB'))

    def fetch_data_x(self, fname, folder):
        return self.get_image(fname, folder, aug=False) / 255.

    def fetch_data_y(self, fname):
        return self.label_transform(fname)

    def fetch_groundtruth(self, fnames):
        res = []
        for fname in fnames:
            res.append(self.fetch_data_y(fname))

        return np.array(res)

    def label_transform(self, fname):
        res = np.zeros((self.height_box_num, self.width_box_num, 5 + len(self.classes.keys())))
        for label in self.dict_labels[fname]:
            xmin, ymin, xmax, ymax, fname, label, url = label

            j = (xmax + xmin) // 2 // self.grid_resolution
            i = (ymin + ymax) // 2 // self.grid_resolution
            c = 1.
            x_ = (xmin + xmax) / 2. / self.image_width
            y_ = (ymin + ymax) / 2. / self.image_height
            w = (xmax - xmin) * 1. / self.image_width
            h = (ymax - ymin) * 1. / self.image_height
            p_c = [0, 0, 0]
            p_c[self.classes[label]] = 1
            temp = [c, x_, y_, w, h]
            temp.extend(p_c)
            res[i][j] = np.array(temp)

        return res

    def label_transform_v2(self, fname):
        res = np.zeros((self.height_box_num, self.width_box_num, self.box_num_per_grid, 5 + len(self.classes.keys())))
        for label in self.dict_labels[fname]:
            xmin, ymin, xmax, ymax, fname, label, url = label
            # ious = [iou(box1=, box2=)]

            j = (xmax + xmin) // 2 // self.grid_resolution
            i = (ymin + ymax) // 2 // self.grid_resolution
            c = 1.
            x_ = (xmin + xmax) / 2. / self.image_width
            y_ = (ymin + ymax) / 2. / self.image_height
            w = (xmax - xmin) * 1. / self.image_width
            h = (ymax - ymin) * 1. / self.image_height
            p_c = [0, 0, 0]
            p_c[self.classes[label]] = 1
            temp = [c, x_, y_, w, h]
            temp.extend(p_c)
            res[i][j] = np.array(temp)

    def data_partition(self, f_dict, tv_ratio=0.1, ratio=0.002, random_seed=None, save_to_file=False):
        """

        :param f_dict:
        :param tv_ratio: train validation ratio
        :param ratio: ratio of picked data
        :param random_seed:
        :param save_to_file:
        :return: list of train set and list of validation set
        """
        if type(random_seed) is int:
            random.seed(random_seed)

        total = len(f_dict.keys())
        picked = int(total * ratio)

        data_picked = rand_sample(f_dict, picked)
        temp = copy.deepcopy(data_picked)
        data_val = rand_sample(temp, int(picked * tv_ratio))
        data_train = dict_substract(temp, data_val)

        if save_to_file:
            df_val = pd.DataFrame(list(data_val.keys()))
            df_val.to_csv('../Data/val.csv', index=False)

            df_train = pd.DataFrame(list(data_train.keys()))
            df_train.to_csv('../Data/tra.csv', index=False)

            df_full = pd.DataFrame(list(data_picked.keys()))
            df_full.to_csv('../Data/full.csv', index=False)

        return list(data_train.keys()), list(data_val.keys())

    # @threadsafe_generator
    def train_data_generator(self, fnames, folder, batch_size, shuffle=False):
        temp = copy.deepcopy(fnames)
        while True:
            offset = 0
            total = len(temp)
            if shuffle:
                random.shuffle(temp)
            while offset + batch_size <= total:
                X, Y = [], []
                for fname in temp[offset:offset + batch_size]:
                    X.append(self.fetch_data_x(fname, folder))
                    Y.append(self.fetch_data_y(fname))

                offset += batch_size
                yield np.array(X), np.array(Y)

    # @threadsafe_generator
    def eval_data_generator(self, fnames, folder, batch_size):
        while True:
            offset = 0
            total = len(fnames)
            while offset + batch_size <= total:
                X, Y = [], []
                for fname in fnames[offset:offset + batch_size]:
                    X.append(self.fetch_data_x(fname, folder))
                    Y.append(self.fetch_data_y(fname))

                offset += batch_size
                yield np.array(X), np.array(Y)

    # @threadsafe_generator
    def pred_data_generator(self, fnames, folder, batch_size):
        offset = 0
        total = len(fnames)
        while offset + batch_size <= total:
            X = []
            for fname in fnames[offset:offset + batch_size]:
                X.append(self.fetch_data_x(fname, folder))

            offset += batch_size
            yield np.array(X)


def predict_refine(predicts, num_bbox=5, num_classes=3, score_threshold=(0.5, 0.5, 0.5), iou_threshold=0.5):
    """

    :param predicts: grid_y * grid_x * (5 * num_bbox + num_classes)
    :param num_bbox:
    :param num_classes:
    :param score_threshold:
    :param iou_threshold:
    :return: class specified predict box list, each class has a corresponding predict box list,
        each box like (class_specified_confidence, x, y, w, h)
        shape num_classes * N * 5
    """
    grid_y, grid_x, channel = np.shape(predicts)

    pred_bboxes = np.stack(np.split(predicts[:, :, 0:5*num_bbox], indices_or_sections=num_bbox, axis=-1), axis=2)
    pr_classes = predicts[:, :, 5*num_bbox:]
    confidences = np.squeeze(pred_bboxes[:, :, :, 0])
    temp = []
    for i in range(num_classes):
        multi = confidences * np.expand_dims(pr_classes[:, :, i], axis=-1)
        temp.append(np.expand_dims(multi, axis=-1))

    class_confs = np.concatenate(temp, axis=-1)   # grid_y * grid_x * num_box * num_classes
    res = []
    for i in range(num_classes):
        # print('class {}'.format(i))
        class_conf = np.expand_dims(class_confs[:, :, :, i], axis=-1)
        positions = pred_bboxes[:, :, :, 1:]
        bboxes = np.concatenate([class_conf, positions], axis=-1)
        candidate_lst = []
        for j in range(grid_y):
            for k in range(grid_x):
                candidate_lst.extend(bboxes[j][k])

        candidate_lst = list(filter(lambda x: x[0] > score_threshold[i], candidate_lst))
        # print('candidates: {}'.format(len(candidate_lst)))
        if len(candidate_lst) > 0:
            candidate_lst = np.array(candidate_lst)
            idxes = list(reversed(np.argsort(candidate_lst[:, 0])))
            candidates = candidate_lst[idxes]

            final_bboxes = non_max_suppression(list(candidates), iou_threshold)
        else:
            final_bboxes = []
        # print('final: {}'.format(len(final_bboxes)))
        res.append(final_bboxes)

    return res


def non_max_suppression(candidate_bboxs, iou_threshold):
    """

    :param candidate_bboxs: each box is like [conf, x, y, w, h]
    :param iou_threshold:
    :return: remained bboxs: each box is like [conf, x, y, w, h]
    """
    res = []
    while candidate_bboxs:
        picked = candidate_bboxs[0]
        res.append(list(picked))
        candidate_bboxs = list(filter(lambda x: iou(x[1:], picked[1:]) < iou_threshold, candidate_bboxs))

    return res


def iou(box1, box2, epsilon=1e-6):
    """

    :param box1: [x1, y1, w1, h1]
    :param box2: [x2, y2, w2, h2]
    :param epsilon:
    :return: scalar, iou of box1 and box2
    """
    W = 1920
    H = 1200
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xmin1 = (x1 - w1/2.) * W
    xmax1 = (x1 + w1/2.) * W
    ymin1 = (y1 - h1/2.) * H
    ymax1 = (y1 + h1/2.) * H

    xmin2 = (x2 - w2 / 2.) * W
    xmax2 = (x2 + w2 / 2.) * W
    ymin2 = (y2 - h2 / 2.) * H
    ymax2 = (y2 + h2 / 2.) * H

    xmin = max([xmin1, xmin2])
    ymin = max([ymin1, ymin2])
    xmax = min([xmax1, xmax2])
    ymax = min([ymax1, ymax2])

    if xmax <= xmin or ymax <= ymin:
        return 0.
    else:
        area_intsec = (xmax - xmin) * (ymax - ymin)
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        return area_intsec / (area1 + area2 - area_intsec + epsilon)


def save_to_file(rows, filename='result.csv'):
    """

    :param rows:
    :param filename:
    :return:
    """
    filepath = os.path.join("./performance", filename)

    csv_file = open(filepath, 'w+', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(rows)

    csv_file.close()


def f1_score(rr, pr):
    if rr == 0 and pr == 0:
        return 0
    else:
        return 2 * rr * pr / (rr + pr)


def df_to_dict(df):
    ans = dict()
    for key in df.keys():
        ans[key] = df[key].values

    return ans


def load_data(label_file='../Data/object-dataset/labels.csv'):
    labels = pd.read_csv(label_file)

    return labels.values


def rand_sample(f_dict, scale):
    """
    Randomly select samples from dict, filename as key, image data as values.
    Scale is sample numbers you want.
    """
    subset = dict()

    keys = random.sample(list(f_dict.keys()), scale)
    for item in keys:
        subset[item] = f_dict[item]

    return subset


def dict_substract(dict1, dict2):
    for key in dict2.keys():
        del dict1[key]

    return dict1


def binned(lst, column, start, end, bins):
    """

    :param lst: 2d list which to be binned
    :param column: binning according to which column of 2d list, desc ordered
    :param start: binning start value
    :param end: binning end value
    :param bins: how many bins to do
    :return:
    """
    intervals = np.linspace(start, end, bins+1, endpoint=True)

    res = []
    j = 0
    for i in range(bins):
        temp = []
        while lst[j][column] <= intervals[i]:
            temp.append(lst[j])
            j += 1
        res.append(list(np.mean(temp, axis=0)))

    return res
