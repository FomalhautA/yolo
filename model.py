####

####

import tensorflow as tf

from loss import yolo_loss


def create_model(lr, lr_decay, decay_steps, batch_size, num_bbox=2, num_classes=3):
    batch_norm_momentum = 0.998
    inputs = tf.keras.layers.Input(shape=(1200, 1920, 3), batch_size=batch_size, dtype=tf.dtypes.float32)

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 4
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 5
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 6
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Layer 7
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 8
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 9
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 10
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 11
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 12
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 13
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 14
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 15
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 16
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same')(x)

    # Layer 170
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 18
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 19 Full Connect
    x = tf.keras.layers.Dense(units=2048, activation=None, use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_norm_momentum,
                                           center=False, scale=True, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 20
    x = tf.keras.layers.Dense(units=num_bbox*5 + num_classes, activation=tf.keras.activations.sigmoid, use_bias=True)(x)

    model = tf.keras.Model(inputs, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
                                                                 decay_rate=lr_decay, staircase=True)

    # lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=yolo_loss,
                  metrics=None,
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model
