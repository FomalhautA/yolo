import tensorflow as tf


class MeanAveragePrecision(tf.keras.metrics.Metric):

    def __init__(self, name='mean_average_presi', **kwargs):
        super(MeanAveragePrecision, self).__init__(name=name, **kwargs)

        self.mAP = self.add_weight(name='mAP', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return self.mAP

    def reset_states(self):
        self.mAP.assign(0.)


