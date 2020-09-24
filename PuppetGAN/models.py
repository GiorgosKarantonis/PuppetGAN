# MIT License

# Copyright (c) 2020 Georgios (Giorgos) Karantonis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Parts of this scipt are adapted from:
# github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
# These parts are subject to:
# Copyright 2019 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 (the "License").

'''
    A collection of all the sub-models used in the PuppetGAN architecture.
'''

import numpy as np

from tensorflow import split, random_normal_initializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    concatenate,
    Input,
    Reshape,
    Dense,
    Conv2D,
    Conv2DTranspose,
    ZeroPadding2D,
    BatchNormalization,
    ReLU,
    LeakyReLU,
    Flatten,
    GaussianNoise
)



def downsample(filters, size, apply_norm=True, name=None):
    '''
        A downsampling block.

        args:
            filters    : the number of filters in the layer's output
            size       : the size of the kernel
            apply_norm : whether or not to add Batch Normalization
                         at the output of the downsampling layer
    '''
    initializer = random_normal_initializer(0., .02)

    result = Sequential(name=name)
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


def upsample(filters, size, name=None):
    '''
        An upsampling block.

        args:
            filters       : the number of filters in the layer's output
            size          : the size of the kernel
    '''
    initializer = random_normal_initializer(0., .02)

    result = Sequential(name=name)
    result.add(Conv2DTranspose(filters,
                               size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(BatchNormalization())
    result.add(ReLU())

    return result


def get_bottleneck(dim=128, noise_std=0.):
    '''
        Create the bottleneck.

        args:
            dim       : the size of the bottleneck
            noise_std : the standard deviation of the Gaussian Noise
                        added to the output of the bottleneck

        returns:
            the bottleneck model
    '''
    assert dim % 2 == 0

    result = Sequential(name='Bottleneck')
    result.add(Flatten('channels_last'))
    result.add(Dense(dim))
    result.add(GaussianNoise(stddev=noise_std))

    return result


def get_encoder(noise_std=0, bottleneck_dim=128):
    '''
        The shared encoder.
        
        In the case of the faces dataset,
        we start with a shape of (bs, 128, 128, 3)
        and we end up with a shape of (bs, 4, 4, 512).

        In the case of the digits dataset,
        we start with a shape of (bs, 32, 32, 3)
        and we end up with a shape of (bs, 1, 1, 512).

        args:
            noise_std      : the std of the bottleneck noise
            bottleneck_dim : the size of the bottleneck
    '''
    encoder = [
        downsample(64, 4, apply_norm=False, name='Downsampling_1'), # (bs, 64, 64, 64) or (bs, 16, 16, 64)
        downsample(128, 4, name='Downsampling_2'), # (bs, 32, 32, 128) or (bs, 8, 8, 128)
        downsample(256, 4, name='Downsampling_3'), # (bs, 16, 16, 256) or (bs, 4, 4, 512)
        downsample(512, 4, name='Downsampling_4'), # (bs, 8, 8, 512) or (bs, 2, 2, 512)
        downsample(512, 4, name='Downsampling_5'), # (bs, 4, 4, 512) or (bs, 1, 1, 512)
    ]

    bottleneck = get_bottleneck(dim=bottleneck_dim, noise_std=noise_std)

    return encoder, bottleneck


def get_decoder(prefix=None):
    '''
        The decoder architecture.
    '''
    if prefix is not None:
        prefix = f'{prefix}_'

    decoder = [
        upsample(512, 4, name=f'{prefix}Upsampling_1'), # (bs, 8, 8, 512) or (bs, 2, 2, 512)
        upsample(256, 4, name=f'{prefix}Upsampling_2'), # (bs, 16, 16, 256) or (bs, 4, 4, 512)
        upsample(128, 4, name=f'{prefix}Upsampling_3'), # (bs, 32, 32, 128) or (bs, 8, 8, 256)
        upsample(64, 4, name=f'{prefix}Upsampling_4') # (bs, 64, 64, 64) or (bs, 16, 16, 128)
    ]

    return decoder


def generator(encoder, decoder, name=None, img_size=(128, 128)):
    '''
        The generator architecture.

        args:
            encoder : the shared encoder
            decoder : the real or the synthetic decoder
    '''
    inputs = Input(shape=[2*img_size[0], img_size[1], 3])
    x1, x2 = inputs[:, :img_size[0], :, :], inputs[:, img_size[0]:, :, :]

    encoder_, bottleneck_ = encoder

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
    x = Reshape(encoder_output_shape_no_bs)(x)

    for up in decoder:
        x = up(x)

    x = Conv2DTranspose(3,
                        4,
                        strides=2,
                        padding='same',
                        kernel_initializer=random_normal_initializer(0., .02),
                        activation='tanh')(x) # (bs, img_size[0], img_size[1], 3)

    return Model(inputs=inputs, outputs=x, name=name)


def pix2pix_discriminator(name=None, img_size=(128, 128)):
    '''
        PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    '''
    initializer = random_normal_initializer(0., .02)

    inputs = Input(shape=[img_size[0], img_size[1], 3])
    x = inputs

    x = downsample(64, 4, False)(x) # (bs, 64, 64, 64) or (bs, 16, 16, 64)
    x = downsample(128, 4)(x) # (bs, 32, 32, 128) or (bs, 8, 8, 128)
    x = downsample(256, 4)(x) # (bs, 16, 16, 256) or (bs, 4, 4, 256)

    x = ZeroPadding2D()(x) # (bs, 18, 18, 256) or (bs, 6, 6, 256)
    x = Conv2D(512,
               4,
               strides=1,
               kernel_initializer=initializer,
               use_bias=False)(x) # (bs, 15, 15, 512) or (bs, 3, 3, 512)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = ZeroPadding2D()(x) # (bs, 17, 17, 512) or (bs, 5, 5, 512)
    x = Conv2D(1,
               4,
               strides=1,
               kernel_initializer=initializer)(x) # (bs, 14, 14, 1) or (bs, 2, 2, 1)

    return Model(inputs=inputs, outputs=x, name=name)
