'''
    Adapted from:
    github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
'''
import numpy as np

from tensorflow import split, random_normal_initializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    concatenate,
    Input,
    Reshape,
    Dense,
    UpSampling2D,
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



def downsample(filters, size, apply_norm=True):
    '''
        A downsampling layer.

        args:
            filters    : The number of filters in the layer's output.
            size       : The size of the kernel.
            apply_norm : Whether or not to add Batch Normalization
                         at the output of the downsampling layer.
    '''
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
    '''
        An upsampling layer.

        args:
            filters       : The number of filters in the layer's output.
            size          : The size of the kernel.
            apply_dropout : Whether or not to add Dropout
                            at the output of the upsampling layer.
    '''
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


def get_bottleneck(dim=128, noise_std=0.):
    '''
        Create the bottleneck.

        args:
            dim       : The size of the bottleneck.
            noise_std : The standard deviation of the Gaussian Noise
                        added to the output of the bottleneck.
    '''
    assert dim % 2 == 0

    result = Sequential()
    result.add(Flatten('channels_last'))
    result.add(Dense(dim))
    result.add(GaussianNoise(stddev=noise_std))

    return result


def get_encoder(noise_std, bottleneck_dim=128, img_size=(128, 128)):
    '''
        The shared encoder.
        In the case of the faces dataset,
        we start with a shape of (128, 128, 3).

        args:
            noise_std      : The std of the bottleneck noise.
            bottleneck_dim : The size of the bottleneck.
    '''
    encoder = [
        downsample(64, 4, apply_norm=False), # (bs, 64, 64, 64)
        downsample(128, 4), # (bs, 32, 32, 128)
        downsample(256, 4), # (bs, 16, 16, 256)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4) # (bs, 1, 1, 512)
    ]

    bottleneck = get_bottleneck(dim=bottleneck_dim, noise_std=noise_std)

    return encoder, bottleneck


def get_decoder(img_size=(128, 128)):
    '''
        The decoder architecture.
    '''
    if img_size[0] == 128:
        decoder = [
            upsample(512, 4), # (bs, 2, 2, 512)
            upsample(512, 4), # (bs, 4, 4, 512)
            upsample(512, 4), # (bs, 8, 8, 512)
            upsample(256, 4), # (bs, 16, 16, 256)
            upsample(128, 4), # (bs, 32, 32, 128)
            upsample(64, 4) # (bs, 64, 64, 64)
        ]
    elif img_size[0] == 32:
        decoder = [
            upsample(512, 4), # (bs, 2, 2, 512)
            upsample(256, 4), # (bs, 4, 4, 512)
            upsample(128, 4), # (bs, 8, 8, 256)
            upsample(64, 4) # (bs, 16, 16, 128)
        ]
    else:
        raise ValueError('Incompatible image size.')

    return decoder

    # up_to = np.log2(img_size[0]) - 1

    # return full_size_decoder[:up_to]


def generator(encoder, decoder, img_size=(128, 128)):
    '''
        The generator architecture.

        args:
            encoder : The shared encoder.
            decoder : The real or the synthetic decoder.
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

    return Model(inputs=inputs, outputs=x)


def pix2pix_discriminator():
    '''
        PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    '''
    initializer = random_normal_initializer(0., .02)

    inputs = Input(shape=[None, None, 3])
    x = inputs

    x = downsample(64, 4, False)(x) # (bs, 64, 64, 64)
    x = downsample(128, 4)(x) # (bs, 32, 32, 128)
    x = downsample(256, 4)(x) # (bs, 16, 16, 256)

    x = ZeroPadding2D()(x) # (bs, 18, 18, 256)
    x = Conv2D(512,
               4,
               strides=1,
               kernel_initializer=initializer,
               use_bias=False)(x) # (bs, 15, 15, 512)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = ZeroPadding2D()(x) # (bs, 17, 17, 512)
    x = Conv2D(1,
               4,
               strides=1,
               kernel_initializer=initializer)(x) # (bs, 14, 14, 1)

    return Model(inputs=inputs, outputs=x)
