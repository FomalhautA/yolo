import numpy as np
import pandas as pd

import csv
import os, cv2
import copy

import random
import collections
from PIL import Image

# from threadsafe_iter import threadsafe_generator
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataUtils:
    def __init__(self, label_file, data_folder, model_dir='./model', batch_size=16):
        """

        :param label_file:
        :param data_folder:
        """
        self.model_dir = model_dir
        self.ORIG_CHANNEL = 1
        self.grid_resolution = 48
        self.image_width = 1920
        self.image_height = 1200
        self.width_box_num = self.image_width // self.grid_resolution
        self.height_box_num = self.image_height // self.grid_resolution
        self.box_num_per_grid = 5
        self.classes = {'Car': 0, 'Truck': 1, 'Pedestrian': 2}
        self.label_file = label_file
        self.data_folder = data_folder
        self.labels = load_data(label_file)
        self.dict_labels = self.labels_to_dict()
        self.train_fnames, self.val_fnames = self.data_partition(self.dict_labels, tv_ratio=0.1, ratio=1,
                                                                 random_seed=1, save_to_file=False)
        self.train_batch_size = batch_size
        self.train_data_gen = self.train_data_generator(self.train_fnames, self.data_folder,
                                                        batch_size=self.train_batch_size, shuffle=True)
        self.val_data_gen = self.eval_data_generator(self.val_fnames, self.data_folder,
                                                     batch_size=self.train_batch_size)

    def labels_to_dict(self):
        re = collections.defaultdict(list)
        for label in self.labels:
            re[label[4]].append(label)

        return re

    @staticmethod
    def mat_to_dict(ids, f_mat):
        """
        Convert id list and feature matrix into dictionary.
        """
        res = dict()
        for id_, f in zip(ids, f_mat):
            res[id_] = f

        return res

    def normalizer(self, mat, norm_avg, norm_std):
        mat = mat.astype(float)
        for i in range(self.ORIG_CHANNEL):
            mat[i] = (mat[i] - norm_avg[i]) / norm_std[i]

        return mat

    def channel_norm_params(self, namelst, folder):

        avg_lst = [0, 0, 0, 0]
        for i, name in enumerate(namelst):
            image_arr = self.get_image_arr([name], folder)[0]
            for j in range(self.ORIG_CHANNEL):
                avg_tmp = np.mean(image_arr[j])

                avg_lst[j] = self.inc_avg(avg_lst[j], i + 1, avg_tmp)
        print([round(item, 2) for item in avg_lst])

        std_lst = [0, 0, 0, 0]
        for i, name in enumerate(namelst):
            image_arr = self.get_image_arr([name], folder)[0]
            for j in range(self.ORIG_CHANNEL):
                std_tmp = np.sqrt(np.mean((image_arr[j] - avg_lst[j]) ** 2))
                std_lst[j] = self.inc_std(std_lst[j], i + 1, std_tmp)
        print([round(item, 2) for item in std_lst])

        return avg_lst, std_lst

    @staticmethod
    def inc_avg(avg, N, x):

        if N >= 1:
            return avg * (N - 1) / N + x / N
        else:
            raise Exception('N must be zero or positive integer.')

    @staticmethod
    def inc_std(std, N, x):

        if N >= 1:
            return (std ** 2 * (N - 1) / N + x ** 2 / N) ** 0.5
        else:
            raise Exception('N must be zero or positive integer.')

    def get_image_arr(self, fname_lst, folder):
        """
        Get full dimension image array in given filename list.
        fname_lst, list of file names.

        Return image array.
        """
        img_arr = []
        for fname in fname_lst:
            img_arr.append(self.get_image(fname, folder))

        return np.array(img_arr)

    def get_image(self, fname, folder, aug=True):

        temp_name = fname
        if not aug:
            temp_name = fname.split('-aug')[0] if fname.__contains__('-aug') else fname

        im = Image.open(os.path.join(folder, temp_name), 'r')

        return np.array(im.convert('RGB'))

    @staticmethod
    def pick_labels(f_dict, fname_lst):
        """
        Pick labels from f_dict.
        The picked feature matrix corresponding to file in fname_lst.

        f_dict, diction with fname as key and label vector as value.
        fname_lst, list of file names.

        Return label vectors.
        """
        res = []
        for fname in fname_lst:
            res.append(f_dict[fname])

        #     print("{} feature vector picked: ".format(len(res)))

        return np.array(res)

    def fetch_data_x(self, fname, folder):
        return self.get_image(fname, folder, aug=False) / 255.

    def fetch_data_y(self, fname):
        return self.label_transform(fname)

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

    @staticmethod
    def get_weights(labels, scale=10, delta=1, threshold=0.004):
        """
        Calculate weights for positive samples and negative samples.
        labels, label vectors
        delta, small number to avoid divided by zero
        threshold, maximum weight threshold

        Return weight for positive, negative; count for positive, negative.
        """
        samples = np.squeeze(labels)

        (N, N_c) = samples.shape  # sample scale, classes

        N_p = np.sum(samples, axis=0)

        N_n = N - N_p

        W_p = N_n / N

        # W_p = np.array([item if item < threshold else threshold for item in W_p])
        #
        # W_n = np.array([item if item < threshold else threshold for item in W_n])
        #
        return W_p, 1 - W_p, N_p, N_n

    @staticmethod
    def showWeights(W_p, W_n, N_p, N_n):
        """
        Show weighted weights W_p, W_n and Count N_p, N_n.
        """
        print([round(item, 4) for item in W_p])
        print([round(item, 4) for item in W_n])
        print([int(item) for item in N_p])
        print([int(item) for item in N_n])

    @staticmethod
    def label_encode(target, cnumber=28):
        """
        target, 2-d list of string
        """
        r = len(target)
        if r == 0:
            return []

        res = np.zeros([r, cnumber])
        for i, str_lst in enumerate(target):
            for item in str_lst:
                col = int(item)
                res[i][col] = 1

        return res.astype("int")

    @staticmethod
    def convert_res_str(hotcode):
        if len(hotcode) == 0:
            return "0"
        res = ""
        for item in hotcode:
            res += str(item)
            res += " "

        return res[:-1]

    @staticmethod
    def lst_substract(lst1, lst2):
        """

        :param lst1:
        :param lst2:
        :return:
        """
        return list(set(lst1) - set(lst2))

    @staticmethod
    def load_test_fname(folder):
        test_f_lst = os.listdir(folder)
        test_f_set = set([item.split("_")[0] for item in test_f_lst])
        test_flst = list(test_f_set)

        return test_flst

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

    @staticmethod
    def convert_y(y_, decode=True):
        temp = []
        if decode:
            for i, item in enumerate(y_):
                if item > 0.5:
                    temp.append(i)
        else:
            for i, item in enumerate(y_):
                if item > 0.5:
                    temp.append(1)
                else:
                    temp.append(0)

        return temp

    def show_labeled_img(self, fname):
        labels = self.dict_labels[fname]
        image = cv2.imread(os.path.join(self.data_folder, fname))
        for label in labels:
            xmin, ymin, xmax, ymax, fname, label, url = label
            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)
            cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0))
        cv2.namedWindow('current_image', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('current_image', image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


def predict_refine(predicts, num_bbox=5, num_classes=3, score_threshold=(0.5, 0.5, 0.5), iou_threshold=0.5):
    """

    :param predicts: grid_y * grid_x * (5 * num_bbox + num_classes)
    :param num_bbox:
    :param num_classes:
    :param score_threshold:
    :param iou_threshold:
    :return: class specified predict box list, each class has a corresponding predict box list,
        each box like (class_specified_confidence, xmin, ymin, xmax, ymax)
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
    print()
    res = []
    for i in range(num_classes):
        class_conf = np.expand_dims(class_confs[:, :, :, i], axis=-1)
        positions = pred_bboxes[:, :, :, 1:]
        bboxes = np.concatenate([class_conf, positions], axis=-1)
        candidate_lst = []
        for j in range(grid_y):
            for k in range(grid_x):
                candidate_lst.extend(bboxes[i][j])

        print(np.array(candidate_lst).shape)
        candidate_lst = list(filter(lambda x: x[0] > score_threshold[i], candidate_lst))
        idxes = list(reversed(np.argsort(candidate_lst[:, 0])))
        candidates = candidate_lst[idxes]

        final_bboxes = non_max_suppression(candidates, iou_threshold)
        res.append(final_bboxes)

    return res


def non_max_suppression(candidate_bboxs, iou_threshold):
    """

    :param candidate_bboxs: each box is like [conf, xmin, ymin, xmax, ymax, conf]
    :param iou_threshold:
    :return: remained bboxs: each box is like [conf, xmin, ymin, xmax, ymax, conf]
    """
    res = []
    while candidate_bboxs:
        picked = candidate_bboxs[0]
        res.append(picked)
        candidate_bboxs = list(filter(lambda x: iou(x[1:], picked[1:]) < iou_threshold, candidate_bboxs))

    return res


def iou(box1, box2, epsilon=1e-6):
    """

    :param box1: [xmin1, ymin1, xmax1, ymax1]
    :param box2: [xmin2, ymin2, xmax2, ymax2]
    :param epsilon:
    :return: scalar, iou of box1 and box2
    """
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


def save_to_file(rows, ckpt=None):
    """

    :param rows:
    :param ckpt:
    :return:
    """
    resfile = 'result_' + ckpt + '.csv' if ckpt else 'result.csv'
    filepath = "./performance/" + resfile

    csv_file = open(filepath, 'w+')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(rows)


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
