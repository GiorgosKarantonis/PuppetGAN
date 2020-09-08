'''
    Adapted from:
    github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
'''
import numpy as np

from tensorflow import split, reshape, random_normal_initializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    concatenate,
    Input,
    Dense,
    Conv2D,
    Conv2DTranspose,
    ZeroPadding2D,
    Dropout,
    BatchNormalization,
    ReLU,
    LeakyReLU,
    Flatten,
    GaussianNoise
)

import tensorflow as tf



def downsample(filters, size, apply_norm=True):
    initializer = random_normal_initializer(0., .02)

    result = Sequential()
    result.add(Conv2D(filters,
                      size,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False))

    if apply_norm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = random_normal_initializer(0., .02)

    result = Sequential()
    result.add(Conv2DTranspose(filters,
                               size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(.5))

    result.add(ReLU())

    return result


def get_bottleneck(dim=128, noise_std=.001):
    assert dim % 2 == 0

    result = Sequential()
    result.add(Flatten('channels_last'))
    result.add(Dense(dim))
    result.add(GaussianNoise(stddev=noise_std))

    return result


def get_encoder(noise_std, bottleneck_dim=128):
    encoder = [
        # downsample(64, 4, apply_norm=False), # (bs, 64, 64, 64)
        downsample(64, 4), # (bs, 64, 64, 64)
        
        downsample(128, 4), # (bs, 32, 32, 128)
        downsample(256, 4), # (bs, 16, 16, 256)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    bottleneck = get_bottleneck(dim=bottleneck_dim, noise_std=noise_std)

    return encoder, bottleneck


def get_decoder():
    return [
        # upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        # upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4), # (bs, 2, 2, 1024)
        upsample(512, 4), # (bs, 4, 4, 1024)

        upsample(512, 4), # (bs, 8, 8, 1024)
        upsample(256, 4), # (bs, 16, 16, 512)
        upsample(128, 4), # (bs, 32, 32, 256)
        upsample(64, 4), # (bs, 64, 64, 128)
        Conv2DTranspose(3,
                        4,
                        strides=2,
                        padding='same',
                        kernel_initializer=random_normal_initializer(0., .02),
                        activation='tanh') # (bs, 128, 128, 3)
    ]


def generator(encoder, decoder):
    inputs = Input(shape=[2*128, 128, 3])
    encoder_, bottleneck_ = encoder
    
    x1, x2 = inputs[:, :128, :, :], inputs[:, 128:, :, :]
    assert x1.get_shape().as_list() == x2.get_shape().as_list()

    for down in encoder_:
        x1 = down(x1)
        x2 = down(x2)

    encoder_output_shape_no_bs = x1.get_shape().as_list()[1:] # don't include the batch size

    x1 = bottleneck_(x1)
    _, rest = split(x1, 2, axis=1)

    x2 = bottleneck_(x2)
    attr, _ = split(x2, 2, axis=1)

    x = concatenate([attr, rest])

    x = Dense(np.prod(encoder_output_shape_no_bs))(x)
    x = tf.keras.layers.Reshape(encoder_output_shape_no_bs)(x)

    for up in decoder:
        x = up(x)

    return Model(inputs=inputs, outputs=x)


def pix2pix_discriminator():
    '''
        PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    '''
    initializer = random_normal_initializer(0., .02)

    inputs = Input(shape=[None, None, 3], name='input_image')
    x = inputs

    # down1 = downsample(64, 4, False)(x) # (bs, 64, 64, 64)
    # REMEMBER TO CHANGE THE NEXT INPUT IF YOU ADD THE REMOVED LAYER!

    down2 = downsample(128, 4)(x) # (bs, 32, 32, 128)
    down3 = downsample(256, 4)(down2) # (bs, 16, 16, 256)

    zero_pad1 = ZeroPadding2D()(down3) # (bs, 18, 18, 256)
    conv = Conv2D(512,
                  4,
                  strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1) # (bs, 15, 15, 512)

    norm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(norm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu) # (bs, 17, 17, 512)

    last = Conv2D(1,
                  4,
                  strides=1,
                  kernel_initializer=initializer)(zero_pad2) # (bs, 14, 14, 1)

    return Model(inputs=inputs, outputs=last)