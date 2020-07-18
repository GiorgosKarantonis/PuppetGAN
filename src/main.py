import time

import numpy as np

from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

import puppetGAN as puppet

from itertools import islice



FACES_REAL_PATH = 'data/dummy/real_'
FACES_SYNTH_PATH = 'data/dummy/synth_'

IMG_SIZE = (128, 128)
IMG_SIZE_SYNTH = (3*128, 3*128)
BATCH_SIZE = 50

EPOCHS = 40



def now():
    return time.time()


def normalize(img):
    '''
        Normalizing the images to [-1, 1]. 
    '''
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    
    return img


def split_to_attributes(img):
    window = int(img.shape[1] / 3)

    rest = img[:, :window, :window, :]
    attr = img[:, window:2*window, :window, :]
    both = img[:, 2*window:, :window, :]

    return attr, rest, both


data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

train_real = data_gen.flow_from_directory(	FACES_REAL_PATH, 
                                            target_size=IMG_SIZE, 
                                            batch_size=BATCH_SIZE, 
                                            class_mode=None)

train_synth = data_gen.flow_from_directory(	FACES_SYNTH_PATH, 
                                            target_size=IMG_SIZE_SYNTH, 
                                            batch_size=BATCH_SIZE, 
                                            class_mode=None)


puppet_GAN = puppet.PuppetGAN(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
puppet_GAN.restore_checkpoint()


n_batches_real = len(train_real) if len(train_real) % BATCH_SIZE == 0 else len(train_real) - 1

train_real = list(islice(train_real, n_batches_real))
train_synth = list(islice(train_synth, n_batches_real))


# train
for epoch in range(EPOCHS):
    reconstruction_loss, disentanglement_loss, cycle_loss, attr_cycle_loss = 0, 0, 0, 0

    print(f'\nEpoch: {epoch+1} / {EPOCHS}')
    start = now()

    batch_count = 1    
    for a, b in zip(train_real, train_synth):
        print(f'\tBatch: {batch_count} / {n_batches_real}\r', end='')

        a = normalize(a)
        b = normalize(b)

        b1, b2, b3 = split_to_attributes(b)

        rec, dis, cycle, attr_cycle = puppet_GAN.train_step(a, b1, b2, b3)
        
        reconstruction_loss += rec
        disentanglement_loss += dis
        cycle_loss += cycle
        attr_cycle_loss += attr_cycle

        batch_count += 1

    # if epoch % 5 == 0:
    #   ckpt_path = puppet_GAN.ckpt_manager.save()
    #   print(f'\tSaving checkpoint for epoch {epoch+1} at {ckpt_path}\n')


    print(f'\tReconstruction Loss:\t{reconstruction_loss / batch_count}')
    print(f'\tDisentanglement Loss:\t{disentanglement_loss / batch_count}')
    print(f'\tCycle Loss:\t\t{cycle_loss / batch_count}')
    print(f'\tAttribute Cycle Loss:\t{attr_cycle_loss / batch_count}')

    print(f'\n\tTime taken for epoch {epoch+1}: {now()-start} sec. ')








