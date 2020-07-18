'''
    Adapted from:
    https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
'''

import numpy as np
import tensorflow as tf



class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon


    def build(self, input_shape):
        self.scale = self.add_weight(   name='scale',
                                        shape=input_shape[-1:],
                                        initializer=tf.random_normal_initializer(1., 0.02),
                                        trainable=True)

        self.offset = self.add_weight(  name='offset',
                                        shape=input_shape[-1:], 
                                        initializer='zeros',
                                        trainable=True)


    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        
        return self.scale * normalized + self.offset



def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(  filters, 
                                        size, 
                                        strides=2, 
                                        padding='same', 
                                        kernel_initializer=initializer, 
                                        use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose( filters, 
                                                size, 
                                                strides=2, 
                                                padding='same', 
                                                kernel_initializer=initializer, 
                                                use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def bottleneck(dim=128):
    assert dim % 2 == 0

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Flatten('channels_last'))
    result.add(tf.keras.layers.Dense(dim))
    result.add(tf.keras.layers.GaussianNoise(stddev=1))

    return result


def generator_single(   img_height, 
                        img_width, 
                        encoder, 
                        decoder, 
                        output_channels, 
                        batch_size, 
                        norm_type='batchnorm'):
    
    inputs = tf.keras.layers.Input(shape=[128, 128, output_channels], batch_size=batch_size)

    x = inputs

    encoder_, bottleneck_ = encoder
    
    # Downsampling
    for down in encoder_:
        x = down(x)

    if bottleneck_:
        encoder_output_shape = x.get_shape().as_list()
        encoder_output_shape_no_bs = encoder_output_shape[1:]  # don't include batch size

        x = bottleneck_(x)
        attr, rest = tf.split(x, 2, axis=1)

        # ADAPT THIS TO THE VARIOUS ATTR, REST
        x = tf.keras.layers.concatenate([attr, rest])
        x = tf.keras.layers.Dense(np.prod(encoder_output_shape_no_bs))(x)

        x = tf.reshape(x, encoder_output_shape)

    # Upsampling
    for up in decoder:
        x = up(x)

    x = tf.keras.layers.Conv2DTranspose(    output_channels, 
                                            4, 
                                            strides=2,
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                            activation='tanh')(x)  # (bs, 256, 256, 3)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_combined( img_height, 
                        img_width, 
                        encoder, 
                        decoder, 
                        output_channels, 
                        batch_size, 
                        norm_type='batchnorm'):
    

    inputs = tf.keras.layers.Input(shape=[2*128, 128, output_channels], batch_size=batch_size)

    x1 = inputs[:128, :, :]
    x2 = inputs[128:, :, :]


    assert x1.get_shape().as_list() == x2.get_shape().as_list()

    encoder_, bottleneck_ = encoder
    
    # Downsampling
    for down in encoder_:
        x1 = down(x1)
        x2 = down(x2)

    if bottleneck_:
        encoder_output_shape = x1.get_shape().as_list()
        encoder_output_shape_no_bs = encoder_output_shape[1:]  # don't include batch size

        x1 = bottleneck_(x1)
        _, rest = tf.split(x1, 2, axis=1)

        x2 = bottleneck_(x2)
        attr, _ = tf.split(x2, 2, axis=1)

        x = tf.keras.layers.concatenate([attr, rest])
        x = tf.keras.layers.Dense(np.prod(encoder_output_shape_no_bs))(x)

        x = tf.reshape(x, encoder_output_shape)

    # Upsampling
    for up in decoder:
        x = up(x)

    x = tf.keras.layers.Conv2DTranspose(    output_channels, 
                                            4, 
                                            strides=2,
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                            activation='tanh')(x)  # (bs, 256, 256, 3)

    return tf.keras.Model(inputs=inputs, outputs=x)


def pix2pix_discriminator(norm_type='batchnorm', target=True):
    '''
        PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    '''

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(  512, 
                                    4, 
                                    strides=1, 
                                    kernel_initializer=initializer, 
                                    use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(  1, 
                                    4, 
                                    strides=1, 
                                    kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)



