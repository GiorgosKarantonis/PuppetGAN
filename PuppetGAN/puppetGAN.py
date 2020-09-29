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
    The PuppeGAN model.
'''

import os
from time import time
from datetime import datetime
from itertools import islice
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

import utils
import models as m



class PuppetGAN:
    def __init__(self,
                 img_size=(128, 128),
                 noise_std=.2,
                 bottleneck_noise=0.,
                 real_gen_lr=2e-4,
                 real_disc_lr=5e-5,
                 synth_gen_lr=2e-4,
                 synth_disc_lr=5e-5):
        # the desired size of the images
        # it doesn't have to be equal to the input size
        # every input will be resized to this value
        self.img_size = img_size

        # the standard deviation of the noise
        # that will be added to the translations
        self.noise_std = noise_std

        # the standard deviation of the noise
        # that will be added at the bottleneck
        self.bottleneck_noise = bottleneck_noise

        # make the learning rates global
        self.real_gen_lr = real_gen_lr
        self.real_disc_lr = real_disc_lr
        self.synth_gen_lr = synth_gen_lr
        self.synth_disc_lr = synth_disc_lr

        # the supervised objective
        self.sup_loss = tf.keras.losses.MeanAbsoluteError()
        
        # the adversarial objective
        self.gan_loss = tf.keras.losses.MeanSquaredError()

        # create the shared encoder
        self.encoder = m.get_encoder(self.bottleneck_noise)
        
        # create the decoders
        self.decoder_real = m.get_decoder(prefix='Real')
        self.decoder_synth = m.get_decoder(prefix='Synthetic')

        # initialize the generators
        self.gen_real, self.gen_real_opt, self.gen_real_grads = self.init_generator(self.encoder,
                                                                                    self.decoder_real,
                                                                                    name='Real_Generator',
                                                                                    lr=self.real_gen_lr)
        self.gen_synth, self.gen_synth_opt, self.gen_synth_grads = self.init_generator(self.encoder,
                                                                                       self.decoder_synth,
                                                                                       name='Synthetic_Generator',
                                                                                       lr=self.synth_gen_lr)

        # initialize the discriminators
        self.disc_real, self.disc_real_opt, self.disc_real_grads = self.init_discriminator(name='Real_Discriminator',
                                                                                           lr=self.real_disc_lr)
        self.disc_synth, self.disc_synth_opt, self.disc_synth_grads = self.init_discriminator(name='Synthetic_Discriminator',
                                                                                              lr=self.synth_disc_lr)

        self.ckpt, self.ckpt_manager = self.define_checkpoints()


    def log_config(self, target_path='results', file_name='config', **kwargs):
        '''
            Creates and saves a report 
            of the model's architecture and its parameters.

            args:
                target_path : where to save the report
                file_name   : the name of the report file.
                              the extension is set automatically to .txt
        '''
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        with open(os.path.join(target_path, f'{file_name}.txt'), 'w') as config_f:
            # write the name of the parent directory
            config_f.write(f'{os.path.basename(os.getcwd())}\n\n')

            # write the current date and time
            config_f.write(f'{datetime.now()}\n')
            config_f.write(f"{'-' * 65}\n\n")

            # write the hyperparameters
            config_f.write('HYPERPARAMS\n')
            
            config_f.write(f'Real Generator LR: {self.real_gen_lr}\n')
            config_f.write(f'Real Discriminator LR: {self.real_disc_lr}\n')
            config_f.write(f'Synthetic Generator LR: {self.synth_gen_lr}\n')
            config_f.write(f'Synthetic Discriminator LR: {self.synth_disc_lr}\n')
            
            config_f.write(f'Image Size: {self.img_size}\n')
            config_f.write(f'Noise std: {self.noise_std}\n')
            config_f.write(f'Bottleneck Noise: {self.bottleneck_noise}\n')
            
            try:
                config_f.write(f"Batch Size: {kwargs['batch_size']}\n")
            except:
                pass
            try:
                config_f.write(f"On Roids: {kwargs['roids']}\n")
            except:
                pass
            try:
                config_f.write(f"Reconstruction Loss Weight: {kwargs['rec_weight']}\n")
            except:
                pass
            try:
                config_f.write(f"Disentanglement Loss Weight: {kwargs['dis_weight']}\n")
            except:
                pass
            try:
                config_f.write(f"Cycle Loss Weight: {kwargs['cycle_weight']}\n")
            except:
                pass
            try:
                config_f.write(f"Attribute Cycle b3 Loss Weight: {kwargs['attr_cycle_b3_weight']}\n")
            except:
                pass
            try:
                config_f.write(f"Attribute Cycle a Loss Weight: {kwargs['attr_cycle_a_weight']}\n")
            except:
                pass

            config_f.write(f"{'-' * 65}\n")
            config_f.write(f'\n\n')

            # write the PuppetGAN architecture
            with redirect_stdout(config_f):
                for d in self.encoder[0]:
                    d.summary()
                    config_f.write('\n\n')
                
                self.encoder[1].summary()
                config_f.write('\n\n')
                
                for u in self.decoder_real:
                    u.summary()
                    config_f.write('\n\n')

                for u in self.decoder_synth:
                    u.summary()
                    config_f.write('\n\n')
                
                self.gen_real.summary()
                config_f.write('\n\n')
                self.gen_synth.summary()
                config_f.write('\n\n')

                self.disc_real.summary()
                config_f.write('\n\n')
                self.disc_synth.summary()
                config_f.write('\n\n')


    def define_checkpoints(self, path='./checkpoints/puppetGAN'):
        '''
            The structure of the checkpoints.
        '''
        ckpt = tf.train.Checkpoint(generator_real=self.gen_real,
                                   generator_synth=self.gen_real,
                                   generator_real_optimizer=self.gen_real_opt,
                                   generator_synth_optimizer=self.gen_synth_opt,
                                   discriminator_real=self.disc_real,
                                   discriminator_synth=self.disc_synth,
                                   discriminator_real_optimizer=self.disc_real_opt,
                                   discriminator_synth_optimizer=self.disc_synth_opt)

        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)

        return ckpt, ckpt_manager


    def restore_checkpoint(self, path='./checkpoints/puppetGAN', ckpt=-1, partial=False):
        '''
            Restores a checkpoint of the model.

            args:
                path    : the folder where the model's checkpoints are stored
                ckpt    : which checkpoints to restore
                          the default value , -1, restores the latest one
                partial : whether or not to restore all the weights (False)
                          or just the ones needed for evaluation (True)
        '''
        if ckpt == -1:
            if self.ckpt_manager.latest_checkpoint:
                if partial:
                    self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
                else:
                    self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored!')
        else:
            if partial:
                self.ckpt.restore(os.path.join(path, ckpt)).expect_partial()
            else:
                self.ckpt.restore(os.path.join(path, ckpt))
            print(f'Restored checkpoint {ckpt}!')


    def init_generator(self, encoder, decoder, name=None, lr=2e-4, beta_1=.5):
        '''
            Creates a generator.
        '''
        generator = m.generator(encoder, decoder, img_size=self.img_size, name=name)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        gradients = None

        return generator, optimizer, gradients


    def init_discriminator(self, lr=5e-5, beta_1=.5, name=None):
        '''
            Creates a discriminator.
        '''
        discriminator = m.pix2pix_discriminator(name=name, img_size=self.img_size)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        gradients = None

        return discriminator, optimizer, gradients


    def generator_loss(self, generated, weight=1):
        '''
            Calculates the generator loss, defined in the __init__ function.
            By default the Mean Squared Error is used. 

            args:
                real      : the real image
                generated : the generated image
                weight    : the weight of the discriminator loss
        '''
        return weight * self.gan_loss(tf.ones_like(generated), generated)


    def discriminator_loss(self, real, generated, weight=1):
        '''
            Calculates the discriminator loss, defined in the __init__ function.
            By default the Mean Squared Error is used. 

            args:
                real      : the real image
                generated : the generated image
                weight    : the weight of the discriminator loss
        '''
        loss_real = self.gan_loss(tf.ones_like(real), real)
        loss_generated = self.gan_loss(tf.zeros_like(generated), generated)

        return weight * (loss_real + loss_generated)


    def supervised_loss(self, real, generated, weight=1):
        '''
            Calculates the supervised loss, defined in the __init__ function.
            By default the Mean Absolute Error is used. 

            args:
                real      : the real image
                generated : the generated image
                weight    : the weight of the supervised loss
        '''
        return weight * self.sup_loss(real, generated)


    @tf.function # enable eager execution for faster run times
    def train_step(self,
                   a,
                   b1,
                   b2,
                   b3,
                   use_roids=False,
                   rec_weight=10,
                   dis_weight=10,
                   cycle_weight=10,
                   attr_cycle_b3_weight=5,
                   attr_cycle_a_weight=3):
        '''
            Performs one training step.

            args:
                a                    : the images from the real domain
                b1                   : the images from the synthetic domain
                                       where only the AoI is present
                b2                   : the images from the synthetic domain
                                       where all the attributes, except the AoI, are present
                b3                   : the images from the synthetic domain
                                       where all the attributes are present
                use_roids            : a boolean variable indicating whether or not
                                       to use the improved PuppetGAN
                rec_weight           : the weight of the reconstruction loss
                dis_weight           : the weight of the disentanglement loss
                cycle_weight         : the weight of the cycle loss
                attr_cycle_b3_weight : the weight of the attribute cycle loss for b3
                attr_cycle_a_weight  : the weight of the attribute cycle loss for a

            returns:
                a dictionary containing all the losses.
                a dictionary containing all the generated images.
        '''
        losses, generated_images = {}, {}
        with tf.GradientTape(persistent=True) as tape:
            # define the weights for each loss
            reconstruction_weight = rec_weight
            dissentaglement_weight = dis_weight
            cycle_weight = cycle_weight
            attr_cycle_weight_b_star = attr_cycle_b3_weight
            attr_cycle_weight_a_star = attr_cycle_a_weight

            # initialize the losses
            reconstruction_loss = 0
            disentanglement_loss = 0
            cycle_loss = 0
            attr_cycle_loss_b_star = 0
            attr_cycle_loss_a_star = 0
            gen_real_loss = 0
            gen_synth_loss = 0
            disc_real_loss = 0
            disc_synth_loss = 0


            # Reconstruction
            # a -> a
            a_hat = self.gen_real(tf.concat([a, a], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(a, a_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_hat))

            # b1 -> b1
            b1_hat = self.gen_synth(tf.concat([b1, b1], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b1, b1_hat)
            gen_synth_loss += self.generator_loss(self.disc_synth(b1_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b1_hat))

            # b2 -> b2
            b2_hat = self.gen_synth(tf.concat([b2, b2], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b2, b2_hat)
            gen_synth_loss += self.generator_loss(self.disc_synth(b2_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b2), self.disc_synth(b2_hat))

            # b3 -> b3
            b3_hat_rec = self.gen_synth(tf.concat([b3, b3], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b3, b3_hat_rec)
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_rec))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_rec))

            # weight reconstruction loss
            reconstruction_loss *= reconstruction_weight


            # Dissentaglement
            # (b2, b1) -> b3
            b3_hat_dis = self.gen_synth(tf.concat([b2, b1], axis=1), training=True)

            disentanglement_loss += self.supervised_loss(b3, b3_hat_dis)
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_dis))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_dis))

            # weight disentanglement loss
            disentanglement_loss *= dissentaglement_weight


            # CycleGAN
            # a -> b -> a
            b_cycled_tilde = self.gen_synth(tf.concat([a, a], axis=1), training=True)
            b_cycled_tilde_noisy = utils.make_noisy(b_cycled_tilde, stddev=self.noise_std)
            a_cycled_hat = self.gen_real(tf.concat([b_cycled_tilde_noisy, b_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(a, a_cycled_hat)
            gen_synth_loss += self.generator_loss(self.disc_synth(b_cycled_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b_cycled_tilde))
            gen_real_loss += self.generator_loss(self.disc_real(a_cycled_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_cycled_hat))

            # b1 -> a -> b1
            a1_cycled_tilde = self.gen_real(tf.concat([b1, b1], axis=1), training=True)
            a1_cycled_tilde_noisy = utils.make_noisy(a1_cycled_tilde, stddev=self.noise_std)
            b1_cycled_hat = self.gen_synth(tf.concat([a1_cycled_tilde_noisy, a1_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b1, b1_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a1_cycled_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a1_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b1_cycled_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b1_cycled_hat))

            # b2 -> a -> b2
            a2_cycled_tilde = self.gen_real(tf.concat([b2, b2], axis=1), training=True)
            a2_cycled_tilde_noisy = utils.make_noisy(a2_cycled_tilde, stddev=self.noise_std)
            b2_cycled_hat = self.gen_synth(tf.concat([a2_cycled_tilde_noisy, a2_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b2, b2_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a2_cycled_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a2_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b2_cycled_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b2), self.disc_synth(b2_cycled_hat))

            # b3 -> a -> b3
            a3_cycled_tilde = self.gen_real(tf.concat([b3, b3], axis=1), training=True)
            a3_cycled_tilde_noisy = utils.make_noisy(a3_cycled_tilde, stddev=self.noise_std)
            b3_cycled_hat = self.gen_synth(tf.concat([a3_cycled_tilde_noisy, a3_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b3, b3_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a3_cycled_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a3_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_cycled_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_cycled_hat))

            # weight cycle loss
            cycle_loss *= cycle_weight


            # Attribute CycleGAN
            # (a, b1) -> a -> b3
            a_tilde = self.gen_real(tf.concat([a, b1], axis=1), training=True)
            a_tilde_noisy = utils.make_noisy(a_tilde, stddev=self.noise_std)
            b3_hat_star = self.gen_synth(tf.concat([b2, a_tilde_noisy], axis=1), training=True)

            attr_cycle_loss_b_star += self.supervised_loss(b3, b3_hat_star)
            gen_real_loss += self.generator_loss(self.disc_real(a_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_star))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_star))

            # (b1, a) -> b -> a
            b_tilde = self.gen_synth(tf.concat([b1, a], axis=1), training=True)
            b_tilde_noisy = utils.make_noisy(b_tilde, stddev=self.noise_std)
            a_hat_star = self.gen_real(tf.concat([a, b_tilde_noisy], axis=1), training=True)

            attr_cycle_loss_a_star += self.supervised_loss(a, a_hat_star)
            gen_synth_loss += self.generator_loss(self.disc_synth(b_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b_tilde))
            gen_real_loss += self.generator_loss(self.disc_real(a_hat_star))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_hat_star))


            # some extra constraints on the Attribute CycleGAN
            if use_roids:
                # # disentanglement roid
                # a3_dis_cycled_tilde = self.gen_real(tf.concat([b2, b1], axis=1), training=True)
                
                # disentanglement_loss += dissentaglement_weight * self.supervised_loss(a3_cycled_tilde, a3_dis_cycled_tilde)
                # gen_real_loss += self.generator_loss(self.disc_real(a3_dis_cycled_tilde))
                # disc_real_loss += self.discriminator_loss(self.disc_real(a3_cycled_tilde), self.disc_real(a3_dis_cycled_tilde))
                
                # attribute cycle roid
                # add new a loss
                a_tilde_star = self.gen_real(tf.concat([a_tilde_noisy, a], axis=1), training=True)
                
                attr_cycle_loss_a_star += self.supervised_loss(a, a_tilde_star)
                gen_real_loss += self.generator_loss(self.disc_real(a_tilde_star))

                # add new b loss
                b_tilde_star = self.gen_real(tf.concat([b_tilde_noisy, b1], axis=1), training=True)
                
                attr_cycle_loss_b_star += self.supervised_loss(b1, b_tilde_star)
                gen_real_loss += self.generator_loss(self.disc_real(b_tilde_star))


            # weight L(f(a, b1)=b3)
            attr_cycle_loss_b_star *= attr_cycle_weight_b_star

            # weight L(g(b1, a)=a)
            attr_cycle_loss_a_star *= attr_cycle_weight_a_star


            # normalize the losses
            losses['reconstruction'] = reconstruction_loss / reconstruction_weight
            losses['disentanglement'] = disentanglement_loss / dissentaglement_weight
            losses['cycle'] = cycle_loss / cycle_weight
            losses['attribute cycle'] = (attr_cycle_loss_b_star / attr_cycle_weight_b_star) + \
                                        (attr_cycle_loss_a_star / attr_cycle_weight_a_star)
            losses['generator real'] = gen_real_loss
            losses['generator synth'] = gen_synth_loss
            losses['discriminator real'] = disc_real_loss
            losses['discriminator synth'] = disc_synth_loss

            # add the supervised losses to the generators
            gen_real_loss += reconstruction_loss +    \
                             cycle_loss +             \
                             attr_cycle_loss_b_star + \
                             attr_cycle_loss_a_star
            gen_synth_loss += reconstruction_loss +    \
                              disentanglement_loss +   \
                              cycle_loss +             \
                              attr_cycle_loss_b_star + \
                              attr_cycle_loss_a_star

        # calculate the gradients
        self.gen_real_grads = tape.gradient(gen_real_loss, self.gen_real.trainable_variables)
        self.disc_real_grads = tape.gradient(disc_real_loss, self.disc_real.trainable_variables)

        self.gen_synth_grads = tape.gradient(gen_synth_loss, self.gen_synth.trainable_variables)
        self.disc_synth_grads = tape.gradient(disc_synth_loss, self.disc_synth.trainable_variables)

        # apply the gradients to the optimizers
        self.gen_real_opt.apply_gradients(zip(self.gen_real_grads, self.gen_real.trainable_variables))
        self.disc_real_opt.apply_gradients(zip(self.disc_real_grads, self.disc_real.trainable_variables))

        self.gen_synth_opt.apply_gradients(zip(self.gen_synth_grads, self.gen_synth.trainable_variables))
        self.disc_synth_opt.apply_gradients(zip(self.disc_synth_grads, self.disc_synth.trainable_variables))

        return losses, {
                        'reconstructed a' : a_hat,
                        'reconstructed b1' : b1_hat,
                        'reconstructed b2' : b2_hat,
                        'reconstructed b3' : b3_hat_rec,
                        'disentangled b3' : b3_hat_dis,
                        'cycled a' : a_cycled_hat,
                        'cycle b tilde' : b_cycled_tilde,
                        'cycled b1' : b1_cycled_hat,
                        'cycle a1 tilde' : a1_cycled_tilde,
                        'cycled b2' : b2_cycled_hat,
                        'cycle a2 tilde' : a2_cycled_tilde,
                        'cycled b3' : b3_cycled_hat,
                        'cycle a3 tilde' : a3_cycled_tilde,
                        'attr cycle a tilde' : a_tilde,
                        'attr cycled b3' : b3_hat_star,
                        'attr cycle b tilde' : b_tilde,
                        'attr cycled a' : a_hat_star
                    }
                        


    def fit(self,
            path_real,
            path_synth,
            path_eval=None,
            batch_size=30,
            epochs=500,
            save_model_every=10,
            save_images_every=10, 
            use_roids=False,
            rec_weight=10,
            dis_weight=10,
            cycle_weight=10,
            attr_cycle_b3_weight=5,
            attr_cycle_a_weight=3,
            save_summary=True):
        '''
            The training function.

            args:
                path_real            : the path containing the images from the real domain are stored
                path_synth           : the path where the images from the real domain are stored
                img_size             : the size of the images. It's used to deal with different datasets
                batch_size           : the size of the mini-batch
                epochs               : the number of epochs
                save_model_every     : every how many epochs to create a new checkpoint
                save_images_every    : every how many epochs to save the outputs of the model
                use_roids            : whether or not to use extra conditions, other than the ones of the paper
                rec_weight           : the weight of the reconstruction loss
                dis_weight           : the weight of the disentanglement loss
                cycle_weight         : the weight of the cycle loss
                attr_cycle_b3_weight : the weight of the attribute cycle loss for b3
                attr_cycle_a_weight  : the weight of the attribute cycle loss for a
                save_summary         : whether or not to create a summary report for
                                       the architecture and the hyperparameters
        '''
        if save_summary:
            self.log_config(batch_size=batch_size,
                            roids=use_roids,
                            rec_weight=rec_weight,
                            dis_weight=dis_weight,
                            cycle_weight=cycle_weight,
                            attr_cycle_b3_weight=attr_cycle_b3_weight,
                            attr_cycle_a_weight=attr_cycle_a_weight)

        losses = np.empty((0, 8), float)

        for epoch in range(epochs):
            start = time()
            print(f'\nEpoch: {epoch} / {epochs}')

            epoch_losses = np.zeros([8])

            # load the datasets
            data_real = utils.get_batch_flow(path_real, self.img_size, batch_size)
            data_synth = utils.get_batch_flow(path_synth, tuple(3*dim for dim in self.img_size), batch_size)

            n_batches_real = len(data_real) if len(data_real) % batch_size == 0 else len(data_real) - 1

            data_real = list(islice(data_real, n_batches_real))
            data_synth = list(islice(data_synth, n_batches_real))

            for i, (a, b) in enumerate(zip(data_real, data_synth)):
                print(f'\tBatch: {i+1} / {n_batches_real}\r', end='')

                # normalize the input images
                a = utils.normalize(a)
                b = utils.normalize(b)

                # split b to b1, b2 and b3
                b1, b2, b3 = utils.split_to_attributes(b)

                batch_losses, generated_images = self.train_step(a=a,
                                                                 b1=b1,
                                                                 b2=b2,
                                                                 b3=b3,
                                                                 use_roids=use_roids,
                                                                 rec_weight=rec_weight,
                                                                 dis_weight=dis_weight,
                                                                 cycle_weight=cycle_weight,
                                                                 attr_cycle_b3_weight=attr_cycle_b3_weight,
                                                                 attr_cycle_a_weight=attr_cycle_a_weight)
                batch_losses = [
                    batch_losses['reconstruction'],
                    batch_losses['disentanglement'],
                    batch_losses['cycle'],
                    batch_losses['attribute cycle'],
                    batch_losses['generator real'],
                    batch_losses['generator synth'],
                    batch_losses['discriminator real'],
                    batch_losses['discriminator synth']
                ]

                epoch_losses = np.add(epoch_losses, batch_losses)

            # calculate the losses for the whole epoch
            epoch_losses = epoch_losses / n_batches_real
            losses = np.append(losses, [epoch_losses], axis=0)

            # save only the images from the last batch to save space
            if save_images_every:
                if epoch % save_images_every == 0 or epoch + 1 == epochs:
                    if path_eval:
                        self.eval(base_path=path_eval, target_folder=epoch)
                    print(f'\tSaved evaluation rows and gifs for epoch {epoch}!')
                    
                    utils.plot_losses(losses)
                    utils.save(a, b1, b2, b3, generated_images, i, epoch, remove_existing=False)
                    print(f'\tSaved losses and images for epoch {epoch}!')

            if save_model_every:
                if epoch % save_model_every == 0 or epoch + 1 == epochs:
                    ckpt_path = self.ckpt_manager.save()
                    print(f'\tSaved checkpoint for epoch {epoch} at {ckpt_path}!\n')

            utils.print_losses(epoch_losses)
            print(f'\n\tTime taken for epoch {epoch}: {time()-start}sec.')


    def eval(self,
             base_path,
             target_path='results/test',
             target_folder=None,
             sample=6):
        '''
            Creates rows of images like the ones from the paper for evaluation.

            args:
                base_path     : where to find the images based on which
                                the function will generate the rows
                target_path   : where to save the output images and gifs
                target_folder : a specific folder inside 'target_path'
                                where the outputs will be saved
                sample        : how many evaluation images to create;
                                if set to 'None' it will generate all of them
        '''
        print('\n\tCreating evaluation rows.')

        if target_folder is not None:
            target_path = os.path.join(target_path, str(target_folder))

        if not os.path.exists(os.path.join(target_path, 'images')):
            os.makedirs(os.path.join(target_path, 'images'))

        if not os.path.exists(os.path.join(target_path, 'gifs')):
            os.makedirs(os.path.join(target_path, 'gifs'))

        alphas_path = os.path.join(base_path, 'real')
        betas_path = os.path.join(base_path, 'synth')

        # load the rows
        alphas = utils.load_test_data(alphas_path, self.img_size)
        betas = utils.load_test_data(betas_path)

        i = 0
        for b1_file in betas:
            i += 1
            if sample is not None and i > sample:
                return

            print(f'\t\tCreating evaluation for image: {i}\r', end='')

            result = np.concatenate([a for a in alphas], axis=1)
            result = np.concatenate((np.zeros(self.img_size + (3,)), result), axis=1)

            b1_file = tf.convert_to_tensor(b1_file)

            b1_file = tf.split(b1_file, 10)
            b1_file = tf.convert_to_tensor(b1_file)[:, :, :self.img_size[1], :]

            for b1 in b1_file:
                new_result_row = np.array(b1)
                b1 = tf.expand_dims(b1, axis=0)

                for a in alphas:
                    a = tf.expand_dims(a, axis=0) # add batch size of 1
                    a_tilde = self.gen_real(tf.concat([a, b1], axis=1), training=False)

                    for img in a_tilde:
                        new_result_row = np.concatenate((new_result_row, img), axis=1)

                result = np.concatenate((result, new_result_row), axis=0)

            result = utils.denormalize(result)
            plt.imsave(os.path.join(target_path, 'images', f'{i}.png'), result)

            start_row = 0 if self.img_size[0] == 128 else 3
            utils.rows_to_gif(result,
                              img_size=self.img_size[0],
                              target_path=os.path.join(target_path, 'gifs'),
                              gif_name=i,
                              start_row=start_row)
