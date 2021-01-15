import numpy as np
import tensorflow as tf

import os
import time

from model import create_model, create_model_v2
from util import *
from PIL import Image
from loss import yolo_loss, mean_avg_pr, yolo_loss_v2, mean_avg_pr_v2
from visualize import visualize_evaluation, show_predicted_img


class BrainCore:
    def __init__(self, label_file='../Data/object-detection-crowdai/labels.csv',
                 data_folder='../Data/object-detection-crowdai'):
        """

        :param label_file:
        :param data_folder:
        """
        train_batch_size = 4
        pred_batch_size = 4

        self.label_file = label_file
        self.data_folder = data_folder
        self.data_utils = DataUtils(label_file=self.label_file, data_folder=self.data_folder,
                                    model_dir='./model', train_batch_size=train_batch_size,
                                    pred_batch_size=pred_batch_size)

    def pipline(self, initial_epoch=0, ckpt=None, version='v2'):
        """

        :param initial_epoch: warm start form which epoch
        :param ckpt: checkpoint model file, if ckpt is not None, training will rewarm from ckpt
        :param version:
        :return:
        """
        epochs = 250

        # lr = 1.      # 0.00005
        # lr_decay = 0.95     # 0.99
        # decay_epoch = 1
        save_epochs = 5

        batch_size = self.data_utils.train_batch_size

        scale_train = len(self.data_utils.train_fnames)
        scale_val = len(self.data_utils.val_fnames)

        print('Training samples {}'.format(scale_train))

        steps_per_epoch_train = scale_train // batch_size
        steps_per_epoch_val = scale_val // batch_size
        # decay_steps = decay_epoch * steps_per_epoch_train

        model_dir = self.data_utils.model_dir

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
        #                                                              decay_rate=lr_decay, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=.005)
        if ckpt is not None:
            model = tf.keras.models.load_model(os.path.join(model_dir, ckpt),
                                               custom_objects={'yolo_loss': yolo_loss,
                                                               'yolo_loss_v2': yolo_loss_v2})
        else:
            model = create_model_v2(optimizer, batch_size=batch_size,
                                    num_bbox=self.data_utils.box_num_per_grid,
                                    num_classes=len(self.data_utils.classes))
        tf.keras.callbacks.LambdaCallback()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model_{epoch}.h5'),
                                                         save_weights_only=False, verbose=1, save_best_only=False,
                                                         save_freq=save_epochs * steps_per_epoch_train)

        file_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'logs'))
        file_writer.set_as_default()

        def lr_schedule(epoch):
            if epoch < 35:
                lr = 0.0001
            elif epoch < 45:
                lr = 0.0001
            elif epoch < 55:
                lr = 0.00006
            elif epoch < 65:
                lr = 0.00004
            elif epoch < 80:
                lr = 0.00003
            elif epoch < 100:
                lr = 0.00002
            elif epoch < 120:
                lr = 0.00001
            elif epoch < 135:
                lr = 0.0005
            elif epoch < 145:
                lr = 0.0006
            elif epoch < 165:
                lr = 0.0003
            elif epoch < 185:
                lr = 0.0001
            else:
                lr = 0.00003

            tf.summary.scalar('lr', data=lr, step=epoch)
            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), profile_batch=0,
                                                     write_graph=True, write_images=False, embeddings_freq=0,
                                                     histogram_freq=1, update_freq='epoch')

        model.fit(self.data_utils.train_data_gen, epochs=epochs,
                  callbacks=[cp_callback, tb_callback, lr_callback],
                  validation_data=self.data_utils.val_data_gen,
                  initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch_train,
                  validation_steps=steps_per_epoch_val,
                  validation_batch_size=batch_size,
                  validation_freq=5, max_queue_size=2, workers=1, use_multiprocessing=False)

        model.save(filepath=model_dir, overwrite=True, include_optimizer=True, save_format='tf',
                   signatures=None, options=None)

        model.summary()

    def predict_pipeline(self, model_file, fnames):
        """

        :param model_file:
        :param fnames：
        :return:
        """
        model = tf.keras.models.load_model(os.path.join(self.data_utils.model_dir, model_file),
                                           custom_objects={'yolo_loss': yolo_loss,
                                                           'yolo_loss_v2': yolo_loss_v2})
        predicts = model.predict(self.data_utils.pred_data_generator(fnames,
                                                                     folder=self.data_utils.data_folder,
                                                                     batch_size=self.data_utils.pred_batch_size),
                                 batch_size=self.data_utils.pred_batch_size,
                                 verbose=1)

        rows = [['xmin', 'xmax', 'ymin', 'ymax', 'Frame', 'Label', 'URL']]
        start = time.time()
        for fname, pred in zip(fnames, predicts):
            # print('File name: {}'.format(fname))
            pred_refined = self.data_utils.predict_refine_v2(pred, num_bbox=self.data_utils.box_num_per_grid,
                                                             num_classes=len(self.data_utils.classes),
                                                             score_threshold=(0.5, 0.5, 0.5), iou_threshold=0.5)
            rows.extend(self.label_decode(fname=fname, refined_pred=pred_refined))

        filename = 'result_' + model_file.split('_')[1].split('.')[0] + '.csv'
        save_to_file(rows=rows, filename=filename)
        end = time.time()
        print('Predict Refine time consumed {}s.'.format(int(end - start)))

        return predicts

    def label_decode(self, fname, refined_pred):
        """

        :param fname:
        :param refined_pred: predicted bboxes: [confidence, x, y, w, h]
        :return: list of bboxes: [xmin, ymin, xmax, ymax, fname, Label, url]
        """
        res = []
        for i in range(len(self.data_utils.classes)):
            for item in refined_pred[i]:
                item.append(fname)
                item.append(self.data_utils.reverse_classes[i])
                item.append('URL')
                x, y, w, h = item[1:5]
                item[1] = max(0, int((x - w / 2) * self.data_utils.image_width))
                item[2] = max(0, int((y - h / 2) * self.data_utils.image_height))
                item[3] = min(self.data_utils.image_width, int((x + w / 2) * self.data_utils.image_width))
                item[4] = min(self.data_utils.image_height, int((y + h / 2) * self.data_utils.image_height))
                res.append(item[1:])

        return res

    def label_decode_debug(self, fname, refined_pred):
        """

        :param fname:
        :param refined_pred: predicted bboxes: [confidence, x, y, w, h, p_c1, p_c2, p_c3]
        :return: list of bboxes: [xmin, ymin, xmax, ymax, fname, Label, url, pc]
        """
        res = []
        for item in refined_pred:
            idx = np.argmax(item[5:])
            mx = item[5+idx]
            item = list(item[0:5])
            item.append(fname)
            item.append(self.data_utils.reverse_classes[idx])
            item.append('URL')
            item.append(mx)
            x, y, w, h = item[1:5]
            item[1] = max(0, int((x - w / 2) * self.data_utils.image_width))
            item[2] = max(0, int((y - h / 2) * self.data_utils.image_height))
            item[3] = min(self.data_utils.image_width, int((x + w / 2) * self.data_utils.image_width))
            item[4] = min(self.data_utils.image_height, int((y + h / 2) * self.data_utils.image_height))
            res.append(item[1:])

        return res

    def predict_single_shot_streaming(self, model_file, fnames):
        """

        :param model_file:
        :param fnames:
        :return:
        """
        data_utils = self.data_utils
        model = tf.keras.models.load_model(os.path.join(self.data_utils.model_dir, model_file),
                                           custom_objects={'yolo_loss': yolo_loss,
                                                           'yolo_loss_v2': yolo_loss_v2})

        for fname in fnames:
            print('filename: {}'.format(fname))
            X = data_utils.fetch_data_x(fname=fname, folder=data_utils.data_folder)
            X = np.expand_dims(X, axis=0)
            y = model.predict(X, batch_size=1, verbose=0, steps=None, callbacks=None)
            gt_np = data_utils.fetch_groundtruth_v2([fname])
            gt = tf.convert_to_tensor(gt_np, dtype=tf.float16)
            y_t = tf.convert_to_tensor(y, dtype=tf.float16)
            tf.executing_eagerly()
            y_ = data_utils.dict_labels[fname]
            print('labels: {}'.format(y_))
            # yolo_loss_v2(gt, y_t, batch_size=1)
            y = np.squeeze(y)
            # y = data_utils.predict_refine_v2(y, num_bbox=data_utils.box_num_per_grid,
            #                                  num_classes=len(data_utils.classes),
            #                                  score_threshold=[0.6, 0.7, 0.7],
            #                                  iou_threshold=0.3)
            # y = self.label_decode(fname=fname, refined_pred=y)
            y_pick = data_utils.predict_pick(y, gt_np, num_bbox=5)
            # y_pick = y_pick[..., 0:5]
            y_pick = self.label_decode_debug(fname=fname, refined_pred=y_pick)

            show_predicted_img(fname=fname, labels=y_, predicts=y_pick, data_folder=data_utils.data_folder)

    def evaluation(self, start, end, steps, fnames, groundtruth):
        """

        :param start: int, start epoch number
        :param end: int, end epoch number(not included)
        :param steps: strides
        :param fnames: list of picture file names
        :param groundtruth:
        :return:
        """
        data_utils = self.data_utils
        if len(groundtruth) % data_utils.pred_batch_size:
            counts = len(groundtruth) // data_utils.pred_batch_size * data_utils.pred_batch_size
            groundtruth = groundtruth[0:counts]

        models = []
        pr_rr_curves = []
        mAPs = []
        class_aps = []
        for i in range(start, end, steps):
            models.append(i)
            model_file = 'model_{}.h5'.format(str(i))
            print('Evaluate Model {}'.format(model_file))
            predicts = self.predict_pipeline(model_file=model_file, fnames=fnames)
            start = time.time()
            mAP, class_ap, binned_prs = mean_avg_pr_v2(groundtruth, predicts, fnames,
                                                       num_classes=len(data_utils.classes),
                                                       num_bbox=data_utils.box_num_per_grid, ckpt=i)
            end = time.time()
            print('mAP calculation time consumed {}s.'.format(int(end - start)))
            mAPs.append(mAP)
            class_aps.append(class_ap)
            pr_rr_curves.append(binned_prs)
            print('mAP: {}'.format(round(mAP, 4)))
            for j, c in enumerate(data_utils.classes):
                print('AP for {}: {}'.format(c, round(class_ap[j], 4)))
                save_to_file(binned_prs[j], 'pr_rr_model_{}_class_{}.csv'.format(i, j))

        save_to_file([[item] for item in mAPs], 'mAPs.csv')
        save_to_file(class_aps, 'class_aps.csv')
        visualize_evaluation(models, pr_rr_curves, mAPs, class_aps,
                             data_utils.reverse_classes)


def test():
    a = tf.convert_to_tensor(np.array([[[0, 1], [0, 2], [0, 0]], [[0, 0], [0, 2], [0, 2]]]), dtype=tf.float16)
    tf.print(a[:-1])
    ind = tf.where(a[..., 0] > 0.5)
    if ind:
        tf.print(ind)
    else:
        print('No value!')

    b = tf.convert_to_tensor(np.array([-np.inf, -np.inf, -0.2, 0., 0.2, 0.8, 1.2]))
    c = tf.clip_by_value(b, 0.0001, 0.9999)
    # b = tf.gather_nd(a, indices=ind)
    tf.print(b)
    tf.print(c)


if __name__ == '__main__':
    brain_core = BrainCore(label_file='../Data/object-detection-crowdai/labels.csv',
                           data_folder='../Data/object-detection-crowdai')
    brain_core.pipline(initial_epoch=135, ckpt='model_135.h5', version='v2')
    # groundtruth = brain_core.data_utils.fetch_groundtruth_v2(brain_core.data_utils.val_fnames)
    # brain_core.evaluation(start=80, end=105, steps=5,
    #                       fnames=brain_core.data_utils.val_fnames, groundtruth=groundtruth)

    # brain_core.predict_single_shot_streaming(model_file='model_135.h5',
    #                                          fnames=brain_core.data_utils.val_fnames)
    # fname = '1479502890271041996.jpg'
    # show_predicted_img(fname=fname, labels=brain_core.data_utils.dict_labels[fname], predicts=[],
    #                    data_folder=brain_core.data_folder)
    # print(brain_core.data_utils.dict_labels[fname])
    # x = np.array([[-5.312, 8., -3.514], [0.4988, 15.69, 7.957]], dtype='float16')
    # soft_max(x, axis=-1)
    # test()
