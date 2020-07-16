import time

import numpy as np

from PIL import Image

import tensorflow as tf
import puppetGAN as puppet



FACES_REAL_PATH = 'data/dummy/real_'
FACES_SYNTH_PATH = 'data/dummy/synth_'

IMG_SIZE = (128, 128)
IMG_SIZE_SYNTH = (3*128, 3*128)
BATCH_SIZE = 1

EPOCHS = 40



def normalize(img):
    '''
        Normalizing the images to [-1, 1]. 
    '''
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    
    return img


def split_to_attributes(img):
    # img_np = np.array(Image.open(img))

    # h_window = int(img_np.shape[0] / 3)
    # w_window = int(img_np.shape[1] / 3)

    # rest = Image.fromarray(img_np[:h_window, :w_window, :]).convert('RGB')
    # attr = Image.fromarray(img_np[h_window:2*h_window, :w_window, :]).convert('RGB')
    # both = Image.fromarray(img_np[2*h_window:, :w_window:, :]).convert('RGB')

    img = tf.squeeze(img)
    window = int(img.shape[0] / 3)
    # img = img.numpy()

    rest = img[:window, :window, :]
    attr = img[window:2*window, :window, :]
    both = img[2*window:, :window, :]

    return tf.expand_dims(attr, 0), tf.expand_dims(rest, 0), tf.expand_dims(both, 0)


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
for epoch in range(EPOCHS):
    print(f'{epoch+1} / {EPOCHS}')
    start = time.time()

    
    for a, b in zip(train_real, train_synth):

        a = normalize(a)
        b = normalize(b)

        b1, b2, b3 = split_to_attributes(b)

        print(b1.shape)
        print(b2.shape)
        print(b3.shape)
        print()

        reconstruction_loss, disentanglement_loss, cycle_loss, attr_cycle_loss = puppet_GAN.train_step(a, b1, b2, b3)

    # if epoch % 5 == 0:
    #   ckpt_path = puppet_GAN.ckpt_manager.save()
    #   print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_path}')

    print(f'{reconstruction_loss}\n{disentanglement_loss}\n{cycle_loss}\n{attr_cycle_loss}\n')
    print(f'Time taken for epoch {epoch+1}: {time.time()-start} sec. \n')








