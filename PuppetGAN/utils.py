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

'''
    A collection of various helper functions.
'''

import os

import imageio
import PIL.Image
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf



def normalize(img):
    '''
        Normalizes in [-1, 1], images of shape [batch_size, height, width, filters].

        args:
            img : the images to normalize

        returns:
            the normalized images
    '''
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    
    return img


def denormalize(img):
    '''
        Projects to [0, 1], images of shape [batch_size, height, width, filters].

        args:
            img : the images to de-normalize

        returns:
            the de-normalized images
    '''
    return ((img + 1) * 127.5) / 255


def split_to_attributes(img):
    '''
        Split b to b1, b2 and b3.
        
        args:
            img : the images, of shape [batch_size, height, width, filters], to split

        returns:
            attr : the image that contains the AoI
            rest : the image that contains all the attributes other than the AoI
            both : the image that contains all the attributes
    '''
    window = int(img.shape[1] / 3)

    rest = img[:, :window, :window, :]
    attr = img[:, window:2*window, :window, :]
    both = img[:, 2*window:, :window, :]

    return attr, rest, both


def get_batch_flow(path, target_size, batch_size):
    '''
        Create a flow of minibatches.
        Ideally tf.keras.preprocessing.image_dataset_from_directory would be used,
        but my cloud's tensorflow version didn't support it.

        args:
            path        : the path where the dataset is stored
            target_size : the size of the images
            batch_size  : the size of the mini-batch

        returns:
            the flow of batches
    '''
    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    return generator.flow_from_directory(path, 
                                         target_size=target_size, 
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         class_mode=None)


def load_test_data(path, img_size=None):
    '''
        Locates and loads the rows used for evaluation.

        args:
            path : the path where the rows are saved at

        returns:
            a tensor containing the evaluation images
    '''
    data = []

    for file in os.listdir(path):
        if file.endswith('.png'):
            img = PIL.Image.open(os.path.join(path, file)).convert('RGB')
            img = np.array(img)
            
            img = tf.convert_to_tensor(img)
            if img_size is not None:
                img = tf.image.resize(img, img_size)
            
            img = normalize(img)

            data.append(img)
    data = tf.convert_to_tensor(data)

    return data


def make_noisy(img, mean=0., stddev=.01):
    '''
        Add noise to an image.

        args:
            img    : the images to add the noise to
            mean   : the mean of the noise distribution
            stddev : the standard deviation of the noise distribution

        returns:
            the noisy images
    '''
    noise = tf.random.normal(img.shape, mean=mean, stddev=stddev)

    return img + noise


def rows_to_gif(img,
                img_size=128,
                target_path='gif',
                gif_name='row_gif',
                header=True,
                start_row=3,
                end_row=2,
                duration=.2):
    '''
        Converts an image of evaluation rows to gif.

        args:
            img         : the image of the evaluation rows
            img_size    : the size of each of the images in the rows
            target_path : where to save the gif
            gif_name    : the name of the output gif
            header      : whether or not to always show to real images
            start_row   : how many rows, from the beggining, to skip
            end_row     : how many rows, from the end, to skip
            duration    : for how many msec to show each row
    '''
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_seq = []

    if header:
        if start_row == 0:
            start_row = 1

        header_img = []
        for j in range(int(img.shape[1]/img_size)):
            cur_img = img[:img_size, j*img_size:(j+1)*img_size, :]
            header_img.append(cur_img)

        header_img = np.concatenate([cur_img for cur_img in header_img], axis=1)

    for i in range(start_row, int(img.shape[0]/img_size) - end_row):
        col_seq = []
        
        for j in range(int(img.shape[1]/img_size)):
            cur_img = img[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size, :]
            col_seq.append(cur_img)

        col_seq = np.concatenate([cur_img for cur_img in col_seq], axis=1)

        if header:
            col_seq = np.concatenate((header_img, col_seq), axis=0)

        image_seq.append((col_seq * 255).astype('uint8'))

    imageio.mimsave(os.path.join(target_path, f'{gif_name}.gif'),
                    image_seq,
                    duration=duration)


def print_losses(losses):
    '''
        Print the training losses.
    '''
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
    '''
        Plot the training losses.
    '''
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
    plt.close()

    plt.figure()
    plt.title('Adversarial Losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, losses[:, 4], color='green', label='Real Generator')
    plt.plot(epochs, losses[:, 6], color='orange', label='Real Discriminator')
    plt.plot(epochs, losses[:, 5], color='blue', label='Synthetic Generator')
    plt.plot(epochs, losses[:, 7], color='red', label='Synthetic Discriminator')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_path, 'adversarial'))
    plt.close()


def save(a, b1, b2, b3, gen_imgs, batch, epoch, base_path='./results/train/', remove_existing=False):
    '''
        Save the generated images.
    '''
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


def crop_from_paper(path, target_path='.'):
    '''
        Gets as input a screenshot from the original paper
        and removes any unncecessary space around the image.
        Additionally, if it detects multiple sub-images in the screenshot
        it splits them and saves them seperately.

        args:
            path        : the path to the folder
                          that contains the screenshot
            target_path : the path where all the detected images
                          will be saved at

        returns:
            sub_images : all the detected images
    '''
    sub_images = []
    for file in os.listdir(path):
        if file.endswith('.png'):
            img = PIL.Image.open(os.path.join(path, file)).convert('RGB')

            img = np.array(img)

            if len(img.shape) < 3:
                img = np.expand_dims(img, axis=2)

            white = 1 if img.max() <= 1 else 255

            columns_to_drop = [i for i in range(img.shape[1]) if img[:, i, :].mean() == white]
            img = np.delete(img, columns_to_drop, axis=1)

            pairs = []
            start, end = None, None
            prev, cur = None, None
            for i, row in enumerate(img):
                cur = row.mean()
                
                if cur == white:
                    if prev != white and prev is not None:
                        end = i-1 if i < len(img) else i
                        pairs.append((start, end))
                else:
                    if prev == white:
                        start = i

                prev = cur

            for i, (start, end) in enumerate(pairs):
                try:
                    if len(sub_images) >= 1:
                        assert cur_img.shape == sub_images[-1].shape

                    cur_img = img[start:end, :, :].squeeze()
                    sub_images.append(cur_img)
                except:
                    continue

    return sub_images
