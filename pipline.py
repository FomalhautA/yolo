import numpy as np
import tensorflow as tf

import os

from model import create_model
from util import *
from PIL import Image
from loss import yolo_loss, mean_avg_pr


class BrainCore:
    def __init__(self, label_file='../Data/object-detection-crowdai/labels.csv',
                 data_folder='../Data/object-detection-crowdai'):
        """

        :param label_file:
        :param data_folder:
        """
        self.label_file = label_file
        self.data_folder = data_folder
        self.batch_size = 2
        self.data_utils = DataUtils(label_file=self.label_file, data_folder=self.data_folder,
                                    model_dir='./model', batch_size=self.batch_size)

    def pipline(self, ckpt=None):
        """

        :param ckpt: checkpoint model file, if ckpt is not None, training will rewarm from ckpt
        :return:
        """
        epochs = 200
        batch_size = 2
        lr = 0.0005
        lr_decay = 0.99
        decay_epoch = 1
        save_epochs = 20

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
            model = create_model(lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, batch_size=batch_size)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, 'model_{epoch}.h5'),
                                                         save_weights_only=False, verbose=1, save_best_only=False,
                                                         save_freq=save_epochs * steps_per_epoch_train)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), profile_batch=0,
                                                     write_graph=True, write_images=False, embeddings_freq=0,
                                                     histogram_freq=1, update_freq='epoch')

        model.fit(self.data_utils.train_data_gen, epochs=epochs, callbacks=[cp_callback, tb_callback],
                  validation_data=self.data_utils.val_data_gen,
                  initial_epoch=20, steps_per_epoch=steps_per_epoch_train, validation_steps=steps_per_epoch_val,
                  validation_batch_size=batch_size,
                  validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)

        model.save(filepath=model_dir, overwrite=True, include_optimizer=True, save_format='tf',
                   signatures=None, options=None)

        model.summary()

    def predict(self, model_file):
        """

        :param model_file:
        :return:
        """
        model = tf.keras.models.load_model(os.path.join(self.data_utils.model_dir, model_file),
                                           custom_objects={'yolo_loss': yolo_loss})
        predicts = model.predict(self.data_utils.val_data_gen, batch_size=2, verbose=1)
        rows = []
        for fname, pred in zip(self.data_utils.val_fnames, predicts):
            pred_refined = predict_refine(pred)
            for i in range(len(self.data_utils.classes)):
                for item in pred_refined[i]:
                    item.append(fname)
                    item.append(self.data_utils.classes[i])
                    item.append('')
                    rows.append(item)

        save_to_file(rows=rows, ckpt=model_file.split('_')[1].split('.')[0])

        return predicts

    @staticmethod
    def test():
        data_utils = DataUtils(label_file='../Data/object-detection-crowdai/labels.csv',
                               data_folder='../Data/object-detection-crowdai')

        for fname in data_utils.dict_labels.keys():

            # re = data_utils.fetch_data_x(fname, '../Data/object-detection-crowdai')
            # print(re.shape)

            labels_trans = data_utils.label_transform(fname)

            for i in range(len(data_utils.dict_labels[fname])):
                print(data_utils.dict_labels[fname][i])

            print(labels_trans.shape)
            print(labels_trans[7])
            data_utils.show_labeled_img(fname)
            break


if __name__ == '__main__':
    brain_core = BrainCore(label_file='../Data/object-detection-crowdai/labels.csv',
                           data_folder='../Data/object-detection-crowdai')
    # brain_core.pipline(ckpt='model_20.h5')

    predicts = brain_core.predict(model_file='model_160.h5')
    # groundtruth = brain_core.data_utils.test_groundtruth

    # reports = mean_avg_pr(ground_truth, predicts, iou_threshold=0.5)

