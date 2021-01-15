####

####

import tensorflow as tf

from loss import yolo_loss, yolo_loss_v2


def create_model(lr, lr_decay, decay_steps, batch_size, num_bbox=2, num_classes=3):
    batch_norm_momentum = 0.998
    inputs = tf.keras.layers.Input(shape=(1200, 1920, 3), batch_size=batch_size, dtype=tf.float32)

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

    # Layer 20 v1
    x = tf.keras.layers.Dense(units=num_bbox * 5 + num_classes, activation=tf.keras.activations.sigmoid,
                              use_bias=True)(x)

    model = tf.keras.Model(inputs, x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps,
                                                                 decay_rate=lr_decay, staircase=True)

    # lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=yolo_loss,
                  metrics=None,
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model


def conv_block(x, filters, kernel_size, strides, use_bias, momentum, center, scale):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param use_bias: conv layer whether use bias
    :param momentum: batch norm moving average momentum
    :param center: boolean, whether use parameter beta
    :param scale: boolean, whether use parameter gamma
    :return:
    """
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                               activation=None, use_bias=use_bias)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                           center=center, scale=scale, trainable=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def yolo_block1(x, filters_m, filters_n, use_bias, momentum, center, scale):
    """

    :param x:
    :param filters_m:
    :param filters_n:
    :param use_bias:
    :param momentum:
    :param center:
    :param scale:
    :return:
    """
    x = conv_block(x, filters=filters_m, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_n, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_m, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)

    return x


def yolo_bolck2(x, filters_m, filters_n, use_bias, momentum, center, scale):
    """

    :param x:
    :param filters_m:
    :param filters_n:
    :param use_bias:
    :param momentum:
    :param center:
    :param scale:
    :return:
    """
    x = conv_block(x, filters=filters_m, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_n, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_m, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_n, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)
    x = conv_block(x, filters=filters_m, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=momentum, scale=scale, center=center)

    return x


def create_model_v2(optimizer, batch_size, num_bbox=5, num_classes=3):
    batch_norm_momentum = 0.995
    center = True
    scale = True
    use_bias = False
    tf.keras.backend.set_floatx('float16')
    inputs = tf.keras.layers.Input(shape=(1200, 1920, 3), batch_size=batch_size)

    # Layer 1
    x = conv_block(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=batch_norm_momentum, scale=scale, center=center)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 600 * 960 * 16

    # Layer 2
    x = conv_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=batch_norm_momentum, scale=scale, center=center)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 300 * 480 * 32

    # Layer 3 - 5
    x = yolo_block1(x, filters_m=128, filters_n=32, use_bias=use_bias, momentum=batch_norm_momentum,
                    center=center, scale=scale)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 150 * 240 * 64

    # Layer 6 - 8
    x = yolo_block1(x, filters_m=256, filters_n=64, use_bias=use_bias, momentum=batch_norm_momentum,
                    center=center, scale=scale)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)  # 75 * 120 * 128

    # Layer 9 - 11
    x = yolo_block1(x, filters_m=256, filters_n=128, use_bias=use_bias, momentum=batch_norm_momentum,
                    center=center, scale=scale)
    R = passthrough_layer(x, s1=75, s2=120)  # 25 * 40 * 2304
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same')(x)  # 25 * 40 * 256

    # Layer 12 - 16
    x = yolo_bolck2(x, filters_m=512, filters_n=512, use_bias=use_bias, momentum=batch_norm_momentum,
                    center=center, scale=scale)  # 25 * 40 * 1024

    # Layer 17
    x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=batch_norm_momentum, scale=scale, center=center)

    # Layer 18
    L = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=batch_norm_momentum, scale=scale, center=center)    # 5 * 8 * 1024

    x = tf.concat([L, R], axis=-1)  # 25 * 40 * 2816

    # Layer 19
    x = conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias,
                   momentum=batch_norm_momentum, scale=scale, center=center)

    # Layer 20
    x = tf.keras.layers.Conv2D(filters=num_bbox * (5 + num_classes), kernel_size=(1, 1), strides=(1, 1),
                               activation=None, use_bias=True)(x)

    model = tf.keras.Model(inputs, x)

    model.compile(optimizer=optimizer,
                  loss=yolo_loss_v2,
                  metrics=None,
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None)

    return model


def passthrough_layer(x, s1=15, s2=24):
    """

    :param x: with shape Batch * 15 * 24 * channel
    :param s1:
    :param s2:
    :return: y: with shape Batch * 5 * 8 * (9*channel)
    """
    stride = 3
    slices = tf.split(x, num_or_size_splits=s1//stride, axis=1)     # 5 * Batch * 3 * 24 * channel

    y = tf.concat([tf.expand_dims(item, axis=0) for item in slices], axis=0)    # 5 * Batch * 3 * 24 * channel
    slices = tf.split(y, num_or_size_splits=s2//stride, axis=3)                 # 8 * 5 * Batch * 3 * 3 * channel
    y = tf.concat([tf.expand_dims(item, axis=0) for item in slices], axis=0)    # 8 * 5 * Batch * 3 * 3 * channel

    slices = tf.split(y, num_or_size_splits=3, axis=3)                          # 3 * 8 * 5 * Batch * 1 * 3 * channel
    y = tf.concat([tf.squeeze(item, axis=3) for item in slices], axis=-1)       # 8 * 5 * Batch * 3 * (3*channel)
    slices = tf.split(y, num_or_size_splits=3, axis=3)                          # 3 * 8 * 5 * Batch * 1 * (3*channel)
    y = tf.concat([tf.squeeze(item, axis=3) for item in slices], axis=-1)       # 8 * 5 * Batch * (9*channel)
    y = tf.transpose(y, perm=[2, 1, 0, 3])                                      # Batch * 5 * 8 * (9*channel)

    return y
