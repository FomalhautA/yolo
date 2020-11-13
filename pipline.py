import numpy as np
import tensorflow as tf

import os
import time

from model import create_model
from util import *
from PIL import Image
from loss import yolo_loss, mean_avg_pr
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

    def pipline(self, initial_epoch=0, ckpt=None):
        """

        :param initial_epoch: warm start form which epoch
        :param ckpt: checkpoint model file, if ckpt is not None, training will rewarm from ckpt
        :return:
        """
        epochs = 100
        lr = 0.0001
        lr_decay = 0.99
        decay_epoch = 1
        save_epochs = 5

        batch_size = self.data_utils.train_batch_size

        scale_train = len(self.data_utils.train_fnames)
        scale_val = len(self.data_utils.val_fnames)

        print('Training samples {}'.format(scale_train))

        steps_per_epoch_train = int(np.floor(scale_train / batch_size))
        steps_per_epoch_val = int(np.floor(scale_val / batch_size))
        decay_steps = decay_epoch * steps_per_epoch_train

        model_dir = self.data_utils.model_dir

        if ckpt is not None:
            model = tf.keras.models.load_model(os.path.join(model_dir, ckpt),
                                               custom_objects={'yolo_loss': yolo_loss})
        else:
            model = create_model(lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, batch_size=batch_size,
                                 num_bbox=self.data_utils.box_num_per_grid,
                                 num_classes=len(self.data_utils.classes))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model_{epoch}.h5'),
                                                         save_weights_only=False, verbose=1, save_best_only=False,
                                                         save_freq=save_epochs * steps_per_epoch_train)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), profile_batch=0,
                                                     write_graph=True, write_images=False, embeddings_freq=0,
                                                     histogram_freq=1, update_freq='epoch')

        model.fit(self.data_utils.train_data_gen, epochs=epochs, callbacks=[cp_callback, tb_callback],
                  validation_data=self.data_utils.val_data_gen,
                  initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch_train,
                  validation_steps=steps_per_epoch_val,
                  validation_batch_size=batch_size,
                  validation_freq=1, max_queue_size=2, workers=1, use_multiprocessing=False)

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
                                           custom_objects={'yolo_loss': yolo_loss})
        predicts = model.predict(self.data_utils.pred_data_generator(fnames,
                                                                     folder=self.data_utils.data_folder,
                                                                     batch_size=self.data_utils.pred_batch_size),
                                 batch_size=self.data_utils.pred_batch_size,
                                 verbose=1)
        rows = [['xmin', 'xmax', 'ymin', 'ymax', 'Frame', 'Label', 'URL']]
        start = time.time()
        for fname, pred in zip(fnames, predicts):
            # print('File name: {}'.format(fname))
            pred_refined = predict_refine(pred, num_bbox=self.data_utils.box_num_per_grid,
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
                item[1] = int((x - w / 2) * self.data_utils.image_width)
                item[2] = int((y - h / 2) * self.data_utils.image_height)
                item[3] = int((x + w / 2) * self.data_utils.image_width)
                item[4] = int((y + h / 2) * self.data_utils.image_height)
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
                                           custom_objects={'yolo_loss': yolo_loss})
        for fname in fnames:
            X = data_utils.fetch_data_x(fname=fname, folder=data_utils.data_folder)
            X = np.expand_dims(X, axis=0)
            y = model.predict(X, batch_size=1, verbose=0, steps=None, callbacks=None)
            y = np.squeeze(y)
            y = predict_refine(y, num_bbox=data_utils.box_num_per_grid, num_classes=len(data_utils.classes),
                               score_threshold=[0.8, 0.8, 0.8], iou_threshold=0.3)
            y = self.label_decode(fname=fname, refined_pred=y)
            y_ = data_utils.dict_labels[fname]
            show_predicted_img(fname=fname, labels=y_, predicts=y, data_folder=data_utils.data_folder)

    def evaluation(self, start, end, steps, fnames):
        """

        :param start:
        :param end:
        :param steps:
        :param fnames:
        :return:
        """
        data_utils = self.data_utils
        groundtruth = data_utils.fetch_groundtruth(fnames)
        if len(groundtruth) % data_utils.train_batch_size:
            counts = len(groundtruth) // data_utils.train_batch_size * data_utils.train_batch_size
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
            mAP, class_ap, binned_prs = mean_avg_pr(groundtruth, predicts, fnames,
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


if __name__ == '__main__':
    brain_core = BrainCore(label_file='../Data/object-detection-crowdai/labels.csv',
                           data_folder='../Data/object-detection-crowdai')
    # brain_core.pipline(initial_epoch=0, ckpt=None)
    # brain_core.evaluation(start=5, end=45, steps=5,
    #                       fnames=brain_core.data_utils.train_fnames)

    brain_core.predict_single_shot_streaming(model_file='model_40.h5',
                                             fnames=brain_core.data_utils.train_fnames)
