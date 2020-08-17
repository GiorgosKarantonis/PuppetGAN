import os
from itertools import islice

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf



def normalize(img):
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    
    return img


def denormalize(img):
    return ((img + 1) * 127.5) / 255


def split_to_attributes(img):
    window = int(img.shape[1] / 3)

    rest = img[:, :window, :window, :]
    attr = img[:, window:2*window, :window, :]
    both = img[:, 2*window:, :window, :]

    return attr, rest, both


def get_batch_flow(path, target_size, batch_size):
    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    return generator.flow_from_directory(path, 
                                         target_size=target_size, 
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         class_mode=None)


def make_noisy(img, mean=0., stddev=.01):
    noise = tf.random.normal(img.shape, mean=mean, stddev=stddev)

    return img + noise


def print_losses(losses):
    print(f'\tReconstruction Loss:\t{losses[0]}')
    print(f'\tDisentanglement Loss:\t{losses[1]}')
    print(f'\tCycle Loss:\t\t{losses[2]}')
    print(f'\tAttribute Cycle Loss:\t{losses[3]}')
    print()
    print(f'\tReal Generator Loss:\t{losses[4]}')
    print(f'\tReal Discriminator Loss:\t{losses[6]}')
    print()
    print(f'\tSynthetic Generator Loss:\t{losses[5]}')
    print(f'\tSynthetic Discriminator Loss:\t{losses[7]}')


def plot_losses(losses, save_path='./results/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs = [i for i in range(losses.shape[0])]

    plt.figure()
    plt.title('Supervised Losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, losses[:, 0], color='blue', label='Reconstruction')
    plt.plot(epochs, losses[:, 1], color='green', label='Disentanglement')
    plt.plot(epochs, losses[:, 2], color='orange', label='Cycle')
    plt.plot(epochs, losses[:, 3], color='red', label='Attirbute Cycle')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, 'supervised'))

    plt.figure()
    plt.title('Adversarial Losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, losses[:, 4], color='green', label='Real Generator')
    plt.plot(epochs, losses[:, 5], color='blue', label='Synthetic Generator')
    plt.plot(epochs, losses[:, 6], color='orange', label='Real Discriminator')
    plt.plot(epochs, losses[:, 7], color='red', label='Synthetic Discriminator')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, 'adversarial'))


def save(a, b1, b2, b3, gen_imgs, batch, epoch, base_path='./results/', remove_existing=False):
    save_path = os.path.join(base_path, f'epoch_{epoch}')

    if remove_existing and os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, (a_, 
            b1_, 
            b2_, 
            b3_, 
            rec_a, 
            rec_b1, 
            rec_b2, 
            rec_b3, 
            dis_b3, 
            cyc_b_tilde, 
            cyc_a, 
            cyc_a1_tilde, 
            cyc_b1, 
            cyc_a2_tilde, 
            cyc_b2, 
            cyc_a3_tilde, 
            cyc_b3, 
            attr_cyc_a_tilde, 
            attr_cyc_b3, 
            attr_cyc_b_tilde, 
            attr_cyc_a) in enumerate(zip(a, 
                                         b1, 
                                         b2, 
                                         b3, 
                                         gen_imgs['reconstructed a'], 
                                         gen_imgs['reconstructed b1'], 
                                         gen_imgs['reconstructed b2'], 
                                         gen_imgs['reconstructed b3'], 
                                         gen_imgs['disentangled b3'], 
                                         gen_imgs['cycle b tilde'], 
                                         gen_imgs['cycled a'], 
                                         gen_imgs['cycle a1 tilde'], 
                                         gen_imgs['cycled b1'], 
                                         gen_imgs['cycle a2 tilde'], 
                                         gen_imgs['cycled b2'], 
                                         gen_imgs['cycle a3 tilde'], 
                                         gen_imgs['cycled b3'], 
                                         gen_imgs['attr cycle a tilde'], 
                                         gen_imgs['attr cycled b3'], 
                                         gen_imgs['attr cycle b tilde'], 
                                         gen_imgs['attr cycled a'], )):

        a_results = np.concatenate((a_,  # original
                                    rec_a,  # reconstruct
                                    np.zeros(a_.shape),  # disentangle
                                    cyc_b_tilde,  # intermediate cycle
                                    cyc_a,  # cycle
                                    attr_cyc_b_tilde,  # intermediate attribute cycle
                                    attr_cyc_a  # attribute cycle
                                    ), axis=0)
        
        b1_results = np.concatenate((b1_,  # original
                                     rec_b1,  # reconstruct
                                     np.zeros(b1_.shape),  # disentangle
                                     cyc_a1_tilde,  # intermediate cycle
                                     cyc_b1,  # cycle
                                     np.zeros(b1_.shape),  # intermediate attribute cycle
                                     np.zeros(b1_.shape)  # attribute cycle
                                    ), axis=0)
        
        b2_results = np.concatenate((b2_,  # original
                                     rec_b2,  # reconstruct
                                     np.zeros(b2_.shape),  # disentangle
                                     cyc_a2_tilde,  # intermediate cycle
                                     cyc_b2,  # cycle
                                     np.zeros(b2_.shape),  # intermediate attribute cycle
                                     np.zeros(b2_.shape)  # attribute cycle
                                    ), axis=0)
        
        b3_results = np.concatenate((b3_,  # original
                                     rec_b3,  # reconstruct
                                     dis_b3,  # disentangle
                                     cyc_a3_tilde,  # intermediate cycle
                                     cyc_b3,  # cycle
                                     attr_cyc_a_tilde,  # intermediate attribute cycle
                                     attr_cyc_b3  # attribute cycle
                                    ), axis=0)

        img = np.concatenate((a_results, b1_results, b2_results, b3_results), axis=1)
        img = denormalize(img)

        plt.imsave(f'{save_path}/{batch}_{i}.png', img)
