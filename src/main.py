import time

import numpy as np

from PIL import Image

import tensorflow as tf
import puppetGAN as puppet



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


# train
reconstruction_loss, disentanglement_loss, cycle_loss, attr_cycle_loss = 0, 0, 0, 0

for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1} / {EPOCHS}')
    start = now()

    batch_count = 1
    n_batches_real = len(train_real)
    n_batches_synth = len(train_synth)
    
    for a, b in zip(train_real, train_synth):
        print(f'Batch: {batch_count} / {n_batches_real} - {n_batches_synth}\n')
        
        # don't go over the last batch
        # because it would break due to different dimensions 
        # compared to all the other batches
        if batch_count % n_batches_real != 0 and batch_count % n_batches_synth != 0:
            a = normalize(a)
            b = normalize(b)

            b1, b2, b3 = split_to_attributes(b)

            losses = puppet_GAN.train_step(a, b1, b2, b3)
            
            reconstruction_loss += losses[0]
            disentanglement_loss += losses[1]
            cycle_loss += losses[2]
            attr_cycle_loss += losses[3]

        batch_count += 1

    # if epoch % 5 == 0:
    #   ckpt_path = puppet_GAN.ckpt_manager.save()
    #   print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_path}')


    print(f'Reconstruction Loss:\t{reconstruction_loss}\n   \
            Disentanglement Loss:\t{disentanglement_loss}\n \
            Cycle Loss:\t{cycle_loss}\n                     \
            Attribute Cycle Loss:\t{attr_cycle_loss}\n')

    print(f'Time taken for epoch {epoch+1}: {now()-start} sec. \n')








