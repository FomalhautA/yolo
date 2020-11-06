import tensorflow as tf

from loss import mean_avg_pr_tf


class MAPCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super().__init__()

        self.X_val, self.y_val = validation_data

    def on_train_begin(self, logs={}):
        if not ('mAP' in self.params['metrics']):
            self.params['metrics'].append('mAP')

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, batch_size=2, verbose=1)

        m_ap = mean_avg_pr_tf(self.y_val, tf.convert_to_tensor(y_pred, dtype=tf.float32))

        logs['mAP'] = m_ap
