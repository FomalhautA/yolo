####

####

import tensorflow as tf


def create_model(lr, lr_decay, decay_steps, batch_size):
    inputs = tf.keras.layers.Input(shape=(608, 608, 3), batch_size=batch_size, dtype=tf.dtypes.float32)

    # Layer 1
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 3
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 4
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 5
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 6
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 7
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 8
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 9
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 10
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 11
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 12
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 13
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Layer 14
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 15
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 16
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 17
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 18
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 19
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 20
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 22
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, center=True, scale=True, training=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # Layer 23
    x = tf.keras.layers.Conv2D(filters=425, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)

    model = tf.keras.Model(inputs, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
                                                                 decay_rate=lr_decay, staircase=True)

    # lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=f1_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy()],
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model