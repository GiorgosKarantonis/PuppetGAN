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



def downsample(filters, size, apply_norm=True):
    initializer = random_normal_initializer(0., 0.02)

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
    initializer = random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2DTranspose(filters, 
                               size, 
                               strides=2, 
                               padding='same', 
                               kernel_initializer=initializer, 
                               use_bias=False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def get_bottleneck(dim=128, noise_std=.01):
    assert dim % 2 == 0

    result = Sequential()
    result.add(Flatten('channels_last'))
    result.add(Dense(dim))
    result.add(GaussianNoise(stddev=noise_std))

    return result


def get_encoder(noise_std=.01):
    encoder = [
        downsample(64, 4, apply_norm=False),  # (bs, 64, 64, 64)
        downsample(128, 4),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    bottleneck = get_bottleneck(dim=128, noise_std=noise_std)
        
    return encoder, bottleneck


def get_decoder():
    return [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4),  # (bs, 8, 8, 1024)
        upsample(256, 4),  # (bs, 16, 16, 512)
        upsample(128, 4),  # (bs, 32, 32, 256)
        upsample(64, 4),  # (bs, 64, 64, 128)
        Conv2DTranspose(3, 
                        4, 
                        strides=2,
                        padding='same', 
                        kernel_initializer=random_normal_initializer(0., 0.02),
                        activation='tanh')  # (bs, 128, 128, 3)
    ]


def generator(encoder, decoder, batch_size):
    try:
        inputs = Input(shape=[128, 128, 3], batch_size=batch_size)
    except:
        inputs = Input(shape=[2*128, 128, 3], batch_size=batch_size)
    
    encoder_, bottleneck_ = encoder

    if inputs.shape[1] == 128:
        x = inputs
        
        for down in encoder_:
            x = down(x)

        encoder_output_shape = x.get_shape().as_list()
        encoder_output_shape_no_bs = encoder_output_shape[1:]  # don't include batch size

        x = bottleneck_(x)
        attr, rest = split(x, 2, axis=1)
    else:
        # combine inputs
        x1, x2 = inputs[:, :128, :, :], inputs[:, 128:, :, :]
        
        assert x1.get_shape().as_list() == x2.get_shape().as_list()

        for down in encoder_:
            x1 = down(x1)
            x2 = down(x2)

        encoder_output_shape = x1.get_shape().as_list()
        encoder_output_shape_no_bs = encoder_output_shape[1:]  # don't include batch size

        x1 = bottleneck_(x1)
        _, rest = split(x1, 2, axis=1)

        x2 = bottleneck_(x2)
        attr, _ = split(x2, 2, axis=1)

    x = concatenate([attr, rest])

    x = Dense(np.prod(encoder_output_shape_no_bs))(x)
    x = reshape(x, encoder_output_shape)

    for up in decoder:
        x = up(x)

    return Model(inputs=inputs, outputs=x)


def pix2pix_discriminator(target=True):
    '''
        PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    '''
    initializer = random_normal_initializer(0., 0.02)

    inp = Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = Input(shape=[None, None, 3], name='target_image')
        x = concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = Conv2D(512, 
                  4, 
                  strides=1, 
                  kernel_initializer=initializer, 
                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(norm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = Conv2D(1, 
                  4, 
                  strides=1, 
                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return Model(inputs=[inp, tar], outputs=last)
    else:
        return Model(inputs=inp, outputs=last)




