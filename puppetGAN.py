import os
from time import time
from itertools import islice

import numpy as np
import tensorflow as tf

import PIL.Image
from matplotlib import pyplot as plt

import utils
import models as m



class PuppetGAN:
    def __init__(self, batch_size, noise_std=.001):
        self.batch_size = batch_size
        self.noise_std = noise_std

        self.sup_loss = tf.keras.losses.MeanAbsoluteError()
        self.gan_loss = tf.keras.losses.MeanSquaredError()

        self.encoder = m.get_encoder()
        self.decoder_real, self.decoder_synth = m.get_decoder(), m.get_decoder()

        self.gen_real, self.gen_real_opt, self.gen_real_grads = None, None, None
        self.gen_synth, self.gen_synth_opt, self.gen_synth_grads = None, None, None

        self.disc_real, self.disc_real_opt, self.disc_real_grads = None, None, None
        self.disc_synth, self.disc_synth_opt, self.disc_synth_grads = None, None, None

        self.setup_model()

        self.ckpt, self.ckpt_manager = self.define_checkpoints()


    def setup_model(self):
        if not self.gen_real:
            self.gen_real, self.gen_real_opt, self.gen_real_grads = self.init_generator(self.encoder, self.decoder_real)

        if not self.gen_synth:
            self.gen_synth, self.gen_synth_opt, self.gen_synth_grads = self.init_generator(self.encoder, self.decoder_synth)

        if not self.disc_real:
            self.disc_real, self.disc_real_opt, self.disc_real_grads = self.init_discriminator()

        if not self.disc_synth:
            self.disc_synth, self.disc_synth_opt, self.disc_synth_grads = self.init_discriminator()


    def define_checkpoints(self, path='./checkpoints/train'):
        ckpt = tf.train.Checkpoint(generator_real=self.gen_real,
                                   generator_synth=self.gen_real,
                                   generator_real_optimizer=self.gen_real_opt,
                                   generator_synth_optimizer=self.gen_synth_opt,
                                   discriminator_real=self.disc_real,
                                   discriminator_synth=self.disc_synth,
                                   discriminator_real_optimizer=self.disc_real_opt,
                                   discriminator_synth_optimizer=self.disc_synth_opt)

        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=2)

        return ckpt, ckpt_manager


    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!')


    def init_generator(self, encoder, decoder, lr=2e-4, beta_1=.5):        
        generator = m.generator(encoder, decoder)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        gradients = None

        return generator, optimizer, gradients


    def init_discriminator(self, lr=5e-5, beta_1=.5, target=False):        
        discriminator = m.pix2pix_discriminator(target=target)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        gradients = None

        return discriminator, optimizer, gradients


    def generator_loss(self, generated):
        return self.gan_loss(tf.ones_like(generated), generated)


    def discriminator_loss(self, real, generated, weight=1):
        loss_real = self.gan_loss(tf.ones_like(real), real)
        loss_generated = self.gan_loss(tf.zeros_like(generated), generated)

        return weight * (loss_real + loss_generated)


    def supervised_loss(self, real, generated, weight=1):
        return weight * self.sup_loss(real, generated)


    @tf.function
    def train_step(self, a, b1, b2, b3):
        losses, generated_images = {}, {}

        with tf.GradientTape(persistent=True) as tape:
            reconstruction_loss, reconstruction_weight = 0, 10
            disentanglement_loss, dissentaglement_weight = 0, 10
            cycle_loss, cycle_weight = 0, 10
            attr_cycle_loss_b_star, attr_cycle_weight_b_star = 0, 5
            attr_cycle_loss_a_star, attr_cycle_weight_a_star = 0, 3

            gen_real_loss = 0
            gen_synth_loss = 0
            disc_real_loss = 0
            disc_synth_loss = 0


            # Reconstruction Loss
            a_hat = self.gen_real(tf.concat([a, a], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(a, a_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_hat))

            b1_hat = self.gen_synth(tf.concat([b1, b1], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b1, b1_hat)
            gen_synth_loss += self.generator_loss(self.disc_synth(b1_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b1_hat))

            b2_hat = self.gen_synth(tf.concat([b2, b2], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b2, b2_hat)
            gen_synth_loss += self.generator_loss(self.disc_synth(b2_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b2), self.disc_synth(b2_hat))

            b3_hat_rec = self.gen_synth(tf.concat([b3, b3], axis=1), training=True)

            reconstruction_loss += self.supervised_loss(b3, b3_hat_rec)
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_rec))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_rec))

            reconstruction_loss *= reconstruction_weight


            # Dissentaglement Loss
            b3_hat_dis = self.gen_synth(tf.concat([b2, b1], axis=1), training=True)

            disentanglement_loss += self.supervised_loss(b3, b3_hat_dis)
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_dis))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_dis))

            disentanglement_loss *= dissentaglement_weight


            # Cycle Loss
            b_cycled_tilde = self.gen_synth(tf.concat([a, a], axis=1), training=True)
            b_cycled_tilde_noisy = utils.make_noisy(b_cycled_tilde, stddev=self.noise_std)
            a_cycled_hat = self.gen_real(tf.concat([b_cycled_tilde_noisy, b_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(a, a_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a_cycled_hat))
            gen_synth_loss += self.generator_loss(self.disc_synth(b_cycled_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_cycled_hat))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b_cycled_tilde))

            a1_cycled_tilde = self.gen_real(tf.concat([b1, b1], axis=1), training=True)
            a1_cycled_tilde_noisy = utils.make_noisy(a1_cycled_tilde, stddev=self.noise_std)
            b1_cycled_hat = self.gen_synth(tf.concat([a1_cycled_tilde_noisy, a1_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b1, b1_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a1_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b1_cycled_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a1_cycled_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b1_cycled_hat))

            a2_cycled_tilde = self.gen_real(tf.concat([b2, b2], axis=1), training=True)
            a2_cycled_tilde_noisy = utils.make_noisy(a2_cycled_tilde, stddev=self.noise_std)
            b2_cycled_hat = self.gen_synth(tf.concat([a2_cycled_tilde_noisy, a2_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b2, b2_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a2_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b2_cycled_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a2_cycled_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b2), self.disc_synth(b2_cycled_hat))

            a3_cycled_tilde = self.gen_real(tf.concat([b3, b3], axis=1), training=True)
            a3_cycled_tilde_noisy = utils.make_noisy(a3_cycled_tilde, stddev=self.noise_std)
            b3_cycled_hat = self.gen_synth(tf.concat([a3_cycled_tilde_noisy, a3_cycled_tilde_noisy], axis=1), training=True)

            cycle_loss += self.supervised_loss(b3, b3_cycled_hat)
            gen_real_loss += self.generator_loss(self.disc_real(a3_cycled_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_cycled_hat))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a3_cycled_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_cycled_hat))

            cycle_loss *= cycle_weight


            # Attribute Cycle Loss
            a_tilde = self.gen_real(tf.concat([a, b1], axis=1), training=True)
            a_tilde_noisy = utils.make_noisy(a_tilde, stddev=self.noise_std)
            b3_hat_star = self.gen_synth(tf.concat([b2, a_tilde_noisy], axis=1), training=True)

            attr_cycle_loss_b_star += self.supervised_loss(b3, b3_hat_star)
            gen_real_loss += self.generator_loss(self.disc_real(a_tilde))
            gen_synth_loss += self.generator_loss(self.disc_synth(b3_hat_star))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_tilde))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b3), self.disc_synth(b3_hat_star))

            attr_cycle_loss_b_star *= attr_cycle_weight_b_star

            b_tilde = self.gen_synth(tf.concat([b1, a], axis=1), training=True)
            b_tilde_noisy = utils.make_noisy(b_tilde, stddev=self.noise_std)
            a_hat_star = self.gen_real(tf.concat([a, b_tilde_noisy], axis=1), training=True)

            attr_cycle_loss_a_star += self.supervised_loss(a, a_hat_star)
            gen_real_loss += self.generator_loss(self.disc_real(a_hat_star))
            gen_synth_loss += self.generator_loss(self.disc_synth(b_tilde))
            disc_real_loss += self.discriminator_loss(self.disc_real(a), self.disc_real(a_hat_star))
            disc_synth_loss += self.discriminator_loss(self.disc_synth(b1), self.disc_synth(b_tilde))

            attr_cycle_loss_a_star *= attr_cycle_weight_a_star


            losses['reconstruction'] = reconstruction_loss / reconstruction_weight
            losses['disentanglement'] = disentanglement_loss / dissentaglement_weight
            losses['cycle'] = cycle_loss / cycle_weight
            losses['attribute cycle'] = (attr_cycle_loss_b_star / attr_cycle_weight_b_star) + (attr_cycle_loss_a_star / attr_cycle_weight_a_star)

            losses['generator real'] = gen_real_loss
            losses['generator synth'] = gen_synth_loss
            losses['discriminator real'] = disc_real_loss
            losses['discriminator synth'] = disc_synth_loss

            # Add the supervised losses to the generators
            gen_real_loss += reconstruction_loss + cycle_loss + attr_cycle_loss_b_star + attr_cycle_loss_a_star
            gen_synth_loss += reconstruction_loss + disentanglement_loss + cycle_loss + attr_cycle_loss_b_star + attr_cycle_loss_a_star


        # Calculate the gradients
        self.gen_real_grads = tape.gradient(gen_real_loss, self.gen_real.trainable_variables)
        self.gen_synth_grads = tape.gradient(gen_synth_loss, self.gen_synth.trainable_variables)

        self.disc_real_grads = tape.gradient(disc_real_loss, self.disc_real.trainable_variables)
        self.disc_synth_grads = tape.gradient(disc_synth_loss, self.disc_synth.trainable_variables)

        # Apply the gradients to the optimizers
        self.gen_real_opt.apply_gradients(zip(self.gen_real_grads, self.gen_real.trainable_variables))
        self.gen_synth_opt.apply_gradients(zip(self.gen_synth_grads, self.gen_synth.trainable_variables))

        self.disc_real_opt.apply_gradients(zip(self.disc_real_grads, self.disc_real.trainable_variables))
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
            img_size=(128, 128),
            epochs=500,
            save_model_every=5,
            save_images_every=1):

        losses = np.empty((0, 8), float)

        for epoch in range(epochs):
            start = time()
            print(f'\nEpoch: {epoch} / {epochs}')

            epoch_losses = np.zeros([8])

            data_real = utils.get_batch_flow(path_real, img_size, self.batch_size)
            data_synth = utils.get_batch_flow(path_synth, tuple(3*dim for dim in img_size), self.batch_size)

            n_batches_real = len(data_real) if len(data_real) % self.batch_size == 0 else len(data_real) - 1

            data_real = list(islice(data_real, n_batches_real))
            data_synth = list(islice(data_synth, n_batches_real))

            for i, (a, b) in enumerate(zip(data_real, data_synth)):
                print(f'\tBatch: {i+1} / {n_batches_real}\r', end='')

                a = utils.normalize(a)
                b = utils.normalize(b)

                b1, b2, b3 = utils.split_to_attributes(b)

                batch_losses, generated_images = self.train_step(a, b1, b2, b3)
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

            epoch_losses = epoch_losses / n_batches_real
            losses = np.append(losses, [epoch_losses], axis=0)

            # save only the images from the last batch to save space
            if save_images_every:
                if epoch % save_images_every == 0 or epoch + 1 == epochs:
                    utils.plot_losses(losses)
                    utils.save(a, b1, b2, b3, generated_images, i, epoch, remove_existing=False)
                    print(f'\n\tSaved losses and images for epoch {epoch}!\n')

            if save_model_every:
                if epoch % save_model_every == 0 or epoch + 1 == epochs:
                    ckpt_path = self.ckpt_manager.save()
                    print(f'\tSaving checkpoint for epoch {epoch} at {ckpt_path}\n')

            utils.print_losses(epoch_losses)
            print(f'\n\tTime taken for epoch {epoch}: {time()-start}sec.')


    def get_face_rows(self, base_path='data/faces/rows_', img_size=(128, 128), target_path='face_rows'):
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        alphas, betas = [], []

        alphas_path = os.path.join(base_path, 'real')
        betas_path = os.path.join(base_path, 'synth')

        for file in os.listdir(alphas_path):
            if file.endswith('.png'):
                img = PIL.Image.open(os.path.join(alphas_path, file)).convert('RGB')
                img = np.array(img)
                img = tf.convert_to_tensor(img)
                img = utils.normalize(img)

                alphas.append(img)
        alphas = tf.convert_to_tensor(alphas)

        for file in os.listdir(betas_path):
            if file.endswith('.png'):
                img = PIL.Image.open(os.path.join(betas_path, file)).convert('RGB')
                img = np.array(img)
                img = tf.convert_to_tensor(img)
                img = utils.normalize(img)

                betas.append(img)
        betas = tf.convert_to_tensor(betas)

        i = 0
        for b1_file in betas:
            i += 1
            print(i)

            result = np.concatenate([a for a in alphas], axis=1)
            result = np.concatenate((np.zeros((128, 128, 3)), result), axis=1)

            b1_file = tf.convert_to_tensor(b1_file)

            b1_file = tf.split(b1_file, 10)
            b1_file = tf.convert_to_tensor(b1_file)[:, :, :img_size[1], :]

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
            plt.imsave(os.path.join(target_path, f'{i}.png'), result)












