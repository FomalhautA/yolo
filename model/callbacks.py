import tensorflow as tf

from loss import mean_avg_pr


class MAPCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super().__init__()

        self.X_val, self.y_val = validation_data

    def on_train_begin(self, logs={}):
        if not ('mAP' in self.params['metrics']):
            self.params['metrics'].append('mAP')

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, batch_size=2, verbose=1)

        y_ = tf.keras.backend.eval(self.y_val)
        m_ap, class_map = mean_avg_pr(y_, y_pred)

        logs['mAP'] = m_ap
        for i, item in enumerate(class_map):
            label = 'mAP_class_{}'.format(i)
            logs[label] = item

