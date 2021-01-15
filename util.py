import collections
import copy
import csv
import time
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
        self.grid_resolution = 48
        self.image_width = 1920
        self.image_height = 1200
        self.width_box_num = self.image_width // self.grid_resolution
        self.height_box_num = self.image_height // self.grid_resolution
        self.anchor_priors = [[676, 583], [377, 327], [205, 161], [113, 83], [51, 50]]
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
        self.train_data_gen = self.train_data_generator_v2(self.train_fnames, self.data_folder,
                                                           batch_size=self.train_batch_size, shuffle=True)
        self.val_data_gen = self.eval_data_generator_v2(self.val_fnames, self.data_folder,
                                                        batch_size=self.train_batch_size)
        self.pred_data_gen = self.pred_data_generator(self.val_fnames, self.data_folder,
                                                      batch_size=self.pred_batch_size)

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

    def fetch_data_y_v2(self, fname):
        return self.label_transform_v2(fname)

    def fetch_groundtruth(self, fnames):
        res = []
        for fname in fnames:
            res.append(self.fetch_data_y(fname))

        return np.array(res)

    def fetch_groundtruth_v2(self, fnames):
        res = []
        start = time.time()
        for i, fname in enumerate(fnames):
            if i % 1000 == 0:
                print('Filename index {}'.format(i))
            res.append(self.fetch_data_y_v2(fname))
        print('Single Label transform time consumed {}s'.format(int(time.time() - start)))
        return np.array(res)

    def label_transform(self, fname):
        """

        :param fname: picture name
        :return: nd array, with shape grid_y * grid_x * (5 + num_classes)
        """
        res = np.zeros((self.height_box_num, self.width_box_num, 5 + len(self.classes.keys())))
        for label in self.dict_labels[fname]:
            xmin, ymin, xmax, ymax, fname, cname, url = label

            j = (xmax + xmin) // 2 // self.grid_resolution
            i = (ymin + ymax) // 2 // self.grid_resolution
            c = 1.
            x_, y_, w, h = loc_encode([xmin, ymin, xmax, ymax])
            p_c = [0, 0, 0]
            p_c[self.classes[cname]] = 1
            temp = [c, x_, y_, w, h]
            temp.extend(p_c)
            res[i][j] = np.array(temp)

        return res

    def set_gt_bbox(self, y, fname):
        """

        :param y: np array, groundtruth storage
        :param fname: picture filename
        :return: np array, same shape with y,
        """
        for label in self.dict_labels[fname]:
            xmin, ymin, xmax, ymax, fname, cname, url = label
            if xmax - xmin < 0.5 or ymax-ymin < 0.5:
                continue
            x_, y_, w, h = loc_encode([xmin, ymin, xmax, ymax])

            j = (xmax + xmin) // 2 // self.grid_resolution
            i = (ymin + ymax) // 2 // self.grid_resolution
            c = 1.

            bboxes = self.reconstruct_bboxs(i, j)
            ious = [iou(bbox, [xmin, ymin, xmax, ymax], encode=False) for bbox in bboxes]
            idx = np.argmax(ious)

            p_c = [0, 0, 0]
            p_c[self.classes[cname]] = 1
            cx = j * self.grid_resolution
            cy = i * self.grid_resolution
            pw, ph = self.anchor_priors[idx]
            if pw < 1e-4 or ph < 1e-4 or h < 1e-4 or w < 1e-4:
                print(idx, pw, ph, ious, i, j, xmin, ymin, xmax, ymax)
                print(label)

            t_x, t_y, t_w, t_h = self.parameterize(x_, y_, w, h, cx, cy, pw, ph)
            # print([round(t_x, 2), round(t_y, 2), round(t_w, 2), round(t_h, 2)])
            temp = [c, t_x, t_y, t_w, t_h]
            temp.extend(p_c)
            y[i][j][idx] = np.array(temp)

    def set_noobj_bbox(self, y, fname, threshold=0.6, epsilon=1e-4):
        """

        :param y:
        :param fname:
        :param threshold: mark max iou < threshold bbox as no object bbox
        :param epsilon:
        :return:
        """
        for i in range(self.height_box_num):
            for j in range(self.width_box_num):
                bboxes = self.reconstruct_bboxs(i, j)
                for b in range(self.box_num_per_grid):
                    ious = [iou(bboxes[b], box[0:4], encode=False) for box in self.dict_labels[fname]]
                    if max(ious) < threshold and y[i][j][b][0] < epsilon:
                        y[i][j][b][-1] = 1

    def label_transform_v2(self, fname):
        """

        :param fname: picture name
        :return: nd array, with shape grid_y * grid_x * [num_boxes * (5+num_classes + sign_noobj)]
        """
        y = np.zeros((self.height_box_num, self.width_box_num, self.box_num_per_grid, 5 + len(self.classes.keys())))

        self.set_gt_bbox(y, fname)

        return np.reshape(y, (self.height_box_num, self.width_box_num, -1))

    def reconstruct_bboxs(self, i, j):
        """

        :param i: index for grid_y
        :param j: inded for grid_x
        :return: list of bboxs, bbox like [xmin, ymin, xmax, ymax]
        """
        x = self.grid_resolution * j + self.grid_resolution / 2
        y = self.grid_resolution * i + self.grid_resolution / 2

        res = []
        for anchor in self.anchor_priors:
            w, h = anchor
            res.append([max(0, x-w/2), max(0, y-h/2), min(x+w/2, self.image_width), min(y+h/2, self.image_height)])

        return res

    def parameterize(self, x, y, w, h, cx, cy, pw, ph):
        """

        :param x: 0-1, centroids on x axis
        :param y: 0-1, centroids on y axis
        :param w: 0-1, width of actual box
        :param h: 0-1, height of actual box
        :param cx: grid cell left top coordinate on x axis
        :param cy: grid cell left top coordinate on y axis
        :param pw: width of anchor box
        :param ph: height of anchor box
        :return: tx, ty, tw, th
        """
        tx = reverse_sigmoid((x*self.image_width - cx) * 1. / self.grid_resolution)
        ty = reverse_sigmoid((y*self.image_height - cy) * 1. / self.grid_resolution)
        tw = np.log(w*self.image_width / pw)
        th = np.log(h*self.image_height / ph)

        return [tx, ty, tw, th]

    def deparameterize(self, box, cx, cy, pw, ph):
        """

        :param box: tx, ty, tw, th
        :param cx: grid cell left top coordinate on x axis
        :param cy: grid cell left top coordinate on y axis
        :param pw: width of anchor box
        :param ph: height of anchor box
        :return: x, y, w, h
        """
        tx, ty, tw, th = box
        x = (sigmoid(tx) * self.grid_resolution + cx) / self.image_width
        y = (sigmoid(ty) * self.grid_resolution + cy) / self.image_height
        w = np.e**tw * pw / self.image_width
        h = np.e**th * ph / self.image_height

        return [x, y, w, h]

    def data_partition(self, f_dict, tv_ratio=0.1, ratio=0.002, random_seed=None,
                       save_to_file=False):
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

    def batch_loop(self, temp, folder):
        X, Y = [], []
        for fname in temp:
            X.append(self.fetch_data_x(fname, folder))
            Y.append(self.fetch_data_y(fname))

        return np.array(X), np.array(Y)

    def batch_loop_v2(self, temp, folder):
        X, Y = [], []
        for fname in temp:
            X.append(self.fetch_data_x(fname, folder))
            Y.append(self.fetch_data_y_v2(fname))

        return np.array(X), np.array(Y)

    # @threadsafe_generator
    def train_data_generator(self, fnames, folder, batch_size, shuffle=False):
        random.seed()
        temp = copy.deepcopy(fnames)
        while True:
            offset = 0
            total = len(temp)
            if shuffle:
                random.shuffle(temp)
            while offset + batch_size <= total:
                yield self.batch_loop(temp[offset:offset+batch_size], folder)
                offset += batch_size

    def train_data_generator_v2(self, fnames, folder, batch_size, shuffle=False):
        random.seed()
        temp = copy.deepcopy(fnames)
        while True:
            offset = 0
            total = len(temp)
            if shuffle:
                random.shuffle(temp)
            while offset + batch_size <= total:
                yield self.batch_loop_v2(temp[offset:offset+batch_size], folder)
                offset += batch_size

    # @threadsafe_generator
    def eval_data_generator(self, fnames, folder, batch_size):
        while True:
            offset = 0
            total = len(fnames)
            while offset + batch_size <= total:
                yield self.batch_loop(fnames[offset:offset+batch_size], folder)
                offset += batch_size

    def eval_data_generator_v2(self, fnames, folder, batch_size):
        while True:
            offset = 0
            total = len(fnames)
            while offset + batch_size <= total:
                yield self.batch_loop_v2(fnames[offset:offset+batch_size], folder)
                offset += batch_size

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

    def predict_refine(self, predicts, num_bbox=5, num_classes=3, score_threshold=(0.5, 0.5, 0.5), iou_threshold=0.5):
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

    def predict_refine_v2(self, predicts, num_bbox=5, num_classes=3, score_threshold=(0.5, 0.5, 0.5), iou_threshold=0.5):
        """

        :param predicts: grid_y * grid_x * (num_bbox * (5 + num_classes)), coordinate is parameterized, t_x, t_y, t_w, t_h
        :param num_bbox:
        :param num_classes:
        :param score_threshold:
        :param iou_threshold:
        :return: class specified predict box list, each class has a corresponding predict box list,
            each box like (class_specified_confidence, x, y, w, h)
            shape num_classes * N * 5
        """
        grid_y, grid_x, channel = np.shape(predicts)

        pred_bboxes = np.stack(np.split(predicts, indices_or_sections=num_bbox, axis=-1), axis=2)
        pr_classes = pred_bboxes[:, :, :, 5:]      # grid_y * grid_x * num_bbox *num_classes
        confidences = np.squeeze(pred_bboxes[:, :, :, 0])   # grid_y * grid_x * num_bbox

        pr_classes = soft_max(pr_classes, axis=-1)
        confidences = sigmoid(confidences)

        # calculate class specified confidences
        class_confs = np.expand_dims(confidences, axis=-1) * pr_classes     # grid_y * grid_x * num_box * num_classes

        res = []
        for i in range(num_classes):
            # print('class {}'.format(i))
            class_conf = np.expand_dims(class_confs[:, :, :, i], axis=-1)
            positions = pred_bboxes[:, :, :, 1:5]
            bboxes = np.concatenate([class_conf, positions], axis=-1)
            candidate_lst = []
            for j in range(grid_y):
                for k in range(grid_x):
                    for b in range(num_bbox):
                        bboxes[j, k, b, 1:] = self.deparameterize(bboxes[j, k, b, 1:],
                                                                  cx=k*self.grid_resolution,
                                                                  cy=j*self.grid_resolution,
                                                                  pw=self.anchor_priors[b][0],
                                                                  ph=self.anchor_priors[b][1])
                        candidate_lst.append(bboxes[j][k][b])

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

    def predict_pick(self, y, gt, num_bbox):
        gt_boxs = np.stack(np.split(gt, indices_or_sections=num_bbox, axis=-1), axis=3)
        inds = np.where(gt_boxs[..., 0] > 0.5)
        grid_y, grid_x, channel = np.shape(y)

        y_boxs = np.stack(np.split(y, indices_or_sections=num_bbox, axis=-1), axis=2)
        # picked = np.expand_dims(y_boxs, axis=0)[inds]
        for j in range(grid_y):
            for k in range(grid_x):
                for b in range(num_bbox):
                    y_boxs[j, k, b, 1:5] = self.deparameterize(y_boxs[j, k, b, 1:5],
                                                               cx=k * self.grid_resolution,
                                                               cy=j * self.grid_resolution,
                                                               pw=self.anchor_priors[b][0],
                                                               ph=self.anchor_priors[b][1])
                    y_boxs[j, k, b, 5:8] = self.softmax(y_boxs[j, k, b, 5:8])
        y_boxs = np.expand_dims(y_boxs, axis=0)
        picked = y_boxs[inds]
        # picked_gt = gt_boxs[..., 1:5][inds]
        # for i, item in enumerate(np.array(inds).T):
        #     print("index: {}".format(item))
        #     print("gt: {}".format(picked_gt[i]))
        #     print("pr: {}".format(picked[i][1:5]))

        return picked

    @staticmethod
    def softmax(x, epsilon=1e-4):
        """

        :return:
        """
        x_clip = np.clip(x, -10, 8)
        return np.e**x_clip / (np.sum(np.e**x_clip) + epsilon)


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


def loc_encode(box, width=1920, height=1200):
    """

    :param box: [xmin, ymin, xmax, ymax]
    :param width:
    :param height:
    :return: box, [x, y, w, h]
    """
    xmin, ymin, xmax, ymax = box
    x = (xmin + xmax) / 2. / width
    y = (ymin + ymax) / 2. / height
    w = (xmax - xmin) * 1. / width
    h = (ymax - ymin) * 1. / height

    return [x, y, w, h]


def loc_decode(box, width=1920, height=1200):
    """

    :param box: [x, y, w, h]
    :param width:
    :param height:
    :return: box, [xmin, ymin, xmax, ymax]
    """
    x, y, w, h = box
    xmin = max(0, (x - w / 2.) * width)
    xmax = min((x + w / 2.) * width, width)
    ymin = max(0, (y - h / 2.) * height)
    ymax = min((y + h / 2.) * height, height)

    return [xmin, ymin, xmax, ymax]


def iou(box1, box2, encode=True, epsilon=1e-6):
    """

    :param box1: [x1, y1, w1, h1] if encode is True; [xmin1, ymin1, xmax1, ymax1] if encode is False
    :param box2: [x2, y2, w2, h2] if encode is True; [xmin2, ymin2, xmax2, ymax2] if encode is False
    :param encode:
    :param epsilon:
    :return: scalar, iou of box1 and box2
    """

    if encode:
        xmin1, ymin1, xmax1, ymax1 = loc_decode(box1)
        xmin2, ymin2, xmax2, ymax2 = loc_decode(box2)
    else:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

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
        while j < len(lst) and lst[j][column] <= intervals[i+1]:
            temp.append(lst[j])
            j += 1
        res.append(list(np.mean(temp, axis=0)))

    return res


def reverse_sigmoid(y, epsilon=1e-6):
    return np.log((y+epsilon)/(1.-y+epsilon))


def sigmoid(x, epsilon=1e-6):
    x = np.clip(x, -10, 10)
    return 1./(1+np.e**(-x)+epsilon)


def soft_max(x, axis, epsilon=1e-4):
    x = np.clip(x, -10, 10)
    sums = np.sum(np.e**x, axis=axis, keepdims=True)
    return np.e**x/(sums + epsilon)
