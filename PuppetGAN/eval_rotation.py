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
# https://www.tensorflow.org/datasets/keras_example
# These parts are subject to:
# Copyright 2020 The TensorFlow Datasets Authors, Licensed under the Apache License, Version 2.0

'''
    Evaluates the performance on the digit rotation task.
'''
import os
import time
import click

import numpy as np
from scipy.stats import linregress

import cv2
from PIL import Image

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()



def normalize_img(image, label):
    '''
        Normalizes images: `uint8` -> `float32`.
    '''
    return tf.cast(image, tf.float32) / 255., label


def get_train_test_data():
    '''
        Fetches and preprocesses the mnist dataset.
    '''
    (ds_train, ds_test), ds_info = tfds.load('mnist',
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)

    ds_train = ds_train.map(normalize_img,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


    ds_test = ds_test.map(normalize_img,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def train_lenet(save_path='./checkpoints/lenet5/ckpt'):
    train_data, test_data = get_train_test_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input([28, 28, 1]),
        tf.keras.layers.Conv2D(6, 3, activation='relu'),
        tf.keras.layers.AveragePooling2D(),

        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.AveragePooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                         save_weights_only=True,
                                                         verbose=1)

    try:
        model.load_weights(save_path).expect_partial()
        print('Restored checkpoint!')
    except:
        print('\nTraining LeNet-5...\n')
        model.fit(train_data,
                  epochs=3,
                  validation_data=test_data,
                  callbacks=[save_checkpoint])

    return model


def get_rotation_size(img_float):
    th3 = cv2.threshold(img_float, 0, 1, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(th3, 0, 2)

    if not cnts:
        return None, None

    contours = np.concatenate(cnts, axis=0)

    if contours.shape[0] < 5:
        return None, None

    hull = cv2.convexHull(contours)
    area = np.sqrt(cv2.contourArea(hull))
    ellipse = cv2.fitEllipse(contours)
    degree = ellipse[2]
    degree_signed = degree if degree < 90 else degree - 180

    return degree_signed, area


def get_scores(path, model, img_size=32):
    print('\nCalculating evaluation scores...')
    n_files = 0
    classification_score = 0
    rotation_score = 0

    for file in os.listdir(path):
        if file.endswith('.png'):
            n_files += 1

            print(f'\tWorking on image: {n_files}\r', end='')
            img = Image.open(os.path.join(path, file)).convert('L')
            img = np.array(img)
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=0)


            # accuracy
            if n_files == 1:
                true_classes = img[:, :img_size, img_size:, :]
                true_classes = np.split(true_classes, (true_classes.shape[2]/img_size), axis=2)

                res = []
                for elem in true_classes:
                    elem = tf.image.resize(elem, (28, 28))
                    y = model.predict(elem)[0]

                    y = np.argmax(y)
                    res.append(y)

                n_rows = int(img.shape[1]/img_size)
                n_columns = int(img.shape[2]/img_size)

            correct, total = 0, 0
            for j in range(1, n_columns):
                for i in range(3, n_rows - 2):
                    cur_img = img[:, i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size, :]
                    cur_img = tf.image.resize(cur_img, (28, 28))
                    y = model.predict(cur_img)
                    try:
                        y = np.argwhere(y==1)[0][1]

                        if y == res[j-1]:
                            correct += 1
                        total += 1
                    except:
                        continue
            
            classification_score += correct / total


            # rotation score
            rots = []
            for j in range(n_columns):
                col_rots = []

                for i in range(3, n_rows - 2):
                    cur_img = img[:, i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size, :].squeeze()
                    cur_rot, _ = get_rotation_size(cur_img)

                    col_rots.append(cur_rot)

                rots.append(col_rots)

            coefs = []
            for j in range(1, len(rots)):
                corr_coef = linregress(rots[0], rots[j])[2]
                coefs.append(corr_coef)

            rotation_score += np.mean(coefs)

    accuracy = classification_score / n_files
    rotation_score = rotation_score / n_files
    
    return accuracy, rotation_score


def get_v_rest(path, img_size=32):
    print('\nCalculating V_rest...')

    images = []
    for file in os.listdir(path):
        if file.endswith('.png'):
            img = Image.open(os.path.join(path, file)).convert('L')
            img = np.array(img)/255

            cur_img = img[3*img_size:-2*img_size, img_size:]
            images.append(cur_img)
    
    images = np.array(images)
    
    std = images.std(0).mean()
    var = images.var(0).mean()

    return std, var


@click.command()
@click.option('--path',
              '-p',
              help='The path to the folder that contains the evaluation images.')
@click.option('--target-path',
              '-t',
              default=None,
              help='Where to save the evaluation report.')
def main(path, target_path):
    start = time.time()

    if not target_path:
        target_path = path

    model = train_lenet()
    acc, rot = get_scores(path, model)
    v_rest_std, v_rest_var = get_v_rest(path)

    with open(os.path.join(target_path, 'evaluation_scores.txt'), 'w') as f:
        f.write(f'Acc : {acc}\n')
        f.write(f'Rot : {rot}\n')
        f.write(f'V_rest (std) : {v_rest_std}\n')
        f.write(f'V_rest (var) : {v_rest_var}\n')

    print('\nResults')
    print(f'Accuracy: {acc}')
    print(f'Rotation: {rot}')
    print(f'V_rest (std): {v_rest_std}')
    print(f'V_rest (var): {v_rest_var}')
    print(f'\nTime elapsed: {time.time() - start}sec.\n')


if __name__ == '__main__':
    main()
