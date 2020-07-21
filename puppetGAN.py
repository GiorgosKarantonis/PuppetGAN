from time import time
from itertools import islice

import numpy as np
import tensorflow as tf

import models


import os
from matplotlib import pyplot as plt



class PuppetGAN:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

        self.gan_loss = tf.keras.losses.MSE

        self.encoder = self.init_encoder()
        self.decoder_real = self.init_decoder()
        self.decoder_synth = self.init_decoder()

        self.gen_dec_r, self.gen_dec_r_opt, self.gen_dec_r_grads = self.init_generator(encoder=self.encoder, decoder=self.decoder_real)
        self.gen_dec_s, self.gen_dec_s_opt, self.gen_dec_s_grads = self.init_generator(encoder=self.encoder, decoder=self.decoder_synth)
        
        self.gen_comb_dec_r, self.gen_comb_dec_r_opt, self.gen_comb_dec_r_grads = self.init_generator(  encoder=self.encoder, 
                                                                                                        decoder=self.decoder_real, 
                                                                                                        combine_inputs=True)
        self.gen_comb_dec_s, self.gen_comb_dec_s_opt, self.gen_comb_dec_s_grads = self.init_generator(  encoder=self.encoder, 
                                                                                                        decoder=self.decoder_synth, 
                                                                                                        combine_inputs=True)

        self.disc_real, self.disc_real_opt, self.disc_real_grads = self.init_discriminator()
        self.disc_synth, self.disc_synth_opt, self.disc_synth_grads = self.init_discriminator()

        self.checkpoint_path = "./checkpoints/train"
        self.ckpt, self.ckpt_manager = self.define_checkpoints()

    
    def define_checkpoints(self):
        ckpt = tf.train.Checkpoint( generator_decoder_real=self.gen_dec_r, 
                                    generator_decoder_synth=self.gen_dec_r, 
                                    generator_combined_decoder_real=self.gen_comb_dec_r, 
                                    generator_combined_decoder_synth=self.gen_comb_dec_s, 
                                    disc_real=self.disc_real,
                                    disc_synth=self.disc_synth,
                                    generator_decoder_real_optimizer=self.gen_dec_r_opt, 
                                    generator_decoder_synth_optimizer=self.gen_dec_s_opt, 
                                    generator_combined_decoder_real_optimizer=self.gen_comb_dec_r_opt, 
                                    generator_combined_decoder_synth_optimizer=self.gen_comb_dec_s_opt)

        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)

        return ckpt, ckpt_manager


    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!')        


    def init_generator(self, encoder, decoder, combine_inputs=False, lr=2e-4, beta_1=.5):
        generator = self.create_generator(encoder, decoder)
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        gradients = None

        return generator, optimizer, gradients


    def init_discriminator(self):
        discriminator = self.create_discriminator()
        optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        gradients = None

        return discriminator, optimizer, gradients


    def init_encoder(self, norm_type='batchnorm', use_bottleneck=True):
        encoder = [
            models.downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
            models.downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
            models.downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
            models.downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
            models.downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
            models.downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
            models.downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
            # models.downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
        ]

        bottleneck = None
        if use_bottleneck:
            bottleneck = models.bottleneck()
            
        return encoder, bottleneck


    def init_decoder(self, norm_type='batchnorm'):
        return [
            # models.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
            models.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
            models.upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
            models.upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
            models.upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
            models.upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
            models.upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
        ]


    def create_generator(self, encoder, decoder, combine_inputs=False, **kwargs):
        if 'norm_type' not in kwargs:
            kwargs['norm_type'] = 'batchnorm'  # 'instancenorm'

        if not combine_inputs:
            generator = models.generator_single(    encoder=encoder, 
                                                    decoder=decoder, 
                                                    batch_size=self.batch_size, 
                                                    norm_type=kwargs['norm_type'])
        else:
            generator = models.generator_combined(  encoder=encoder, 
                                                    decoder=decoder, 
                                                    batch_size=self.batch_size, 
                                                    norm_type=kwargs['norm_type'])


        return generator


    def create_discriminator(self, discriminator_type='pix2pix', **kwargs):
        if discriminator_type == 'pix2pix':
            if 'norm_type' not in kwargs:
                kwargs['norm_type'] = 'batchnorm'

            if 'target' not in kwargs:
                kwargs['target'] = False

            discriminator = models.pix2pix_discriminator(   norm_type=kwargs['norm_type'], 
                                                            target=kwargs['target'])
        else:
            raise NotImplementedError


        return discriminator


    def discriminator_loss(self, real, generated, weight=.5):
        real_loss = self.gan_loss(tf.ones_like(real), real)
        generated_loss = self.gan_loss(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return weight * total_disc_loss


    def l_p_loss(self, real, generated, p=2, weight=1):
        return weight * tf.norm((real - generated), ord=p)


    def make_noisy(self, img, mean=0., stddev=1., dtype=tf.dtypes.float32, seed=None, name=None):
        noise = tf.random.normal(   img.shape, 
                                    mean=mean, 
                                    stddev=stddev, 
                                    dtype=dtype, 
                                    seed=seed, 
                                    name=name)

        return img + noise


    def normalize(self, img):
        '''
            Normalizing the images to [-1, 1]. 
        '''
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        
        return img


    def denormalize(self, img):
        return ((img + 1) * 127.5) / 255


    def split_to_attributes(self, img):
        window = int(img.shape[1] / 3)

        rest = img[:, :window, :window, :]
        attr = img[:, window:2*window, :window, :]
        both = img[:, 2*window:, :window, :]

        return attr, rest, both


    def get_batch_flow(self, generator, path, target_size):
        return generator.flow_from_directory(   path, 
                                                target_size=target_size, 
                                                batch_size=self.batch_size, 
                                                shuffle=True, 
                                                class_mode=None)


    def save(self, b1, b2, b3, b3_hat_dis, batch, epoch):
        for i, (b1_, b2_, b3_, b3_hat_dis_) in enumerate(zip(b1, b2, b3, b3_hat_dis)):
            save_path = f'./results/disentangled/epoch_{epoch}/'

            top = np.concatenate((b1_.numpy(), b2_.numpy()), axis=1)
            bottom = np.concatenate((b3_.numpy(), b3_hat_dis_.numpy()), axis=1)

            img = np.concatenate((top, bottom), axis=0)
            img = self.denormalize(img)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.imsave(f'{save_path}{batch}_{i}.png', img)


    @tf.function
    def train_step(self, a, b1, b2, b3):
        losses, generated_images = {}, {}

        with tf.GradientTape(persistent=True) as tape:
            # persistent=True because the tape is used more than once to calculate the gradients

            total_discriminator_loss_real, total_discriminator_loss_synth = 0, 0

            # Reconstruction Loss            
            a_hat = self.gen_dec_r(a, training=True)
            generated_images['reconstructed a'] = a_hat

            reconstruction_loss_a = self.l_p_loss(a, a_hat, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a_hat))


            b1_hat = self.gen_dec_s(b1, training=True)
            generated_images['reconstructed b1'] = b1_hat

            reconstruction_loss_b1 = self.l_p_loss(b1, b1_hat, p=1)
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b1), generated=self.disc_synth(b1_hat))


            b2_hat = self.gen_dec_s(b2, training=True)
            generated_images['reconstructed b2'] = b2_hat

            reconstruction_loss_b2 = self.l_p_loss(b2, b2_hat, p=1)
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b2), generated=self.disc_synth(b2_hat))


            b3_hat_rec = self.gen_dec_s(b3, training=True)
            generated_images['reconstructed b3'] = b3_hat_rec

            reconstruction_loss_b3 = self.l_p_loss(b3, b3_hat_rec, p=1)
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b3), generated=self.disc_synth(b3_hat_rec))


            total_reconstruction_loss =     reconstruction_loss_a   \
                                        +   reconstruction_loss_b1  \
                                        +   reconstruction_loss_b2  \
                                        +   reconstruction_loss_b3
            losses['recostruction'] = total_reconstruction_loss


            # Dissentaglement Loss
            b3_hat_dis = self.gen_comb_dec_s([b2, b1], training=True)
            generated_images['disentangled b3'] = b3_hat_dis

            disentanglement_loss_b3 = self.l_p_loss(b3, b3_hat_dis, p=1)
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b3), generated=self.disc_synth(b3_hat_dis))
            

            total_disentanglement_loss = disentanglement_loss_b3
            losses['disentanglement'] = total_disentanglement_loss


            # Cycle Loss
            b_cycled_tilde = self.gen_dec_s(a, training=True)
            b_cycled_tilde = self.make_noisy(b_cycled_tilde)
            a_cycled_hat = self.gen_dec_r(b_cycled_tilde, training=True)
            generated_images['cycled b tilde'] = b_cycled_tilde
            generated_images['cycled a hat'] = a_cycled_hat

            cycle_loss_a = self.l_p_loss(a, a_cycled_hat, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a_cycled_hat))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b1), generated=self.disc_synth(b_cycled_tilde))


            a1_cycled_tilde = self.gen_dec_r(b1, training=True)
            a1_cycled_tilde = self.make_noisy(a1_cycled_tilde)
            b1_cycled_hat = self.gen_dec_s(a1_cycled_tilde, training=True)

            cycle_loss_b1 = self.l_p_loss(b1, b1_cycled_hat, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a1_cycled_tilde))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b1), generated=self.disc_synth(b1_cycled_hat))


            a2_cycled_tilde = self.gen_dec_r(b2, training=True)
            a2_cycled_tilde = self.make_noisy(a2_cycled_tilde)
            b2_cycled_hat = self.gen_dec_s(a2_cycled_tilde, training=True)

            cycle_loss_b2 = self.l_p_loss(b2, b2_cycled_hat, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a2_cycled_tilde))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b2), generated=self.disc_synth(b2_cycled_hat))


            a3_cycled_tilde = self.gen_dec_r(b3, training=True)
            a3_cycled_tilde = self.make_noisy(a3_cycled_tilde)
            b3_cycled_hat = self.gen_dec_s(a3_cycled_tilde, training=True)

            cycle_loss_b3 = self.l_p_loss(b3, b3_cycled_hat, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a3_cycled_tilde))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b3), generated=self.disc_synth(b3_cycled_hat))

            
            total_cycle_loss = cycle_loss_a + cycle_loss_b1 + cycle_loss_b2 + cycle_loss_b3
            losses['cycle'] = total_cycle_loss


            # Attribute Cycle Loss
            a_tilde = self.gen_comb_dec_r([a, b1], training=True)
            a_tilde = self.make_noisy(a_tilde)
            b3_hat_star = self.gen_comb_dec_s([b2, a_tilde], training=True)
            generated_images['a tilde'] = a_tilde
            generated_images['b3 hat star'] = b3_hat_star

            attr_cycle_loss_b3 = self.l_p_loss(b3, b3_hat_star, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a_tilde))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b3), generated=self.disc_synth(b3_hat_star))


            b_tilde = self.gen_comb_dec_s([b1, a], training=True)
            b_tilde = self.make_noisy(b_tilde)
            a_hat_star = self.gen_comb_dec_r([a, b_tilde], training=True)
            generated_images['b tilde'] = b_tilde
            generated_images['a hat star'] = a_hat_star

            attr_cycle_loss_a = self.l_p_loss(a, a_hat_star, p=1)
            total_discriminator_loss_real += self.discriminator_loss(real=self.disc_real(a), generated=self.disc_real(a_hat_star))
            total_discriminator_loss_synth += self.discriminator_loss(real=self.disc_synth(b1), generated=self.disc_synth(b_tilde))


            total_attr_cycle_loss = attr_cycle_loss_b3 + attr_cycle_loss_a
            losses['attribute cycle'] = total_attr_cycle_loss


            # Supervised Losses per Generator
            total_gen_dec_r_loss = reconstruction_loss_a + total_cycle_loss
            total_gen_dec_s_loss = reconstruction_loss_b1 + reconstruction_loss_b2 + reconstruction_loss_b3 + total_cycle_loss
            total_gen_comb_dec_r_loss = total_attr_cycle_loss
            total_gen_comb_dec_s_loss = total_disentanglement_loss + total_attr_cycle_loss

            losses['generator with real encoder'] = total_gen_dec_r_loss
            losses['generator with synthetic encoder'] = total_gen_dec_s_loss
            losses['combined generator with real encoder'] = total_gen_comb_dec_r_loss
            losses['combined generator with synthetic encoder'] = total_gen_comb_dec_s_loss

            losses['real discriminator'] = total_discriminator_loss_real
            losses['synthetic discriminator'] = total_discriminator_loss_synth


        # Calculate the Gradients
        self.gen_dec_r_grads = tape.gradient(total_gen_dec_r_loss, self.gen_dec_r.trainable_variables)
        self.gen_dec_s_grads = tape.gradient(total_gen_dec_s_loss, self.gen_dec_s.trainable_variables)
        
        self.gen_comb_dec_r_grads = tape.gradient(total_gen_comb_dec_r_loss, self.gen_comb_dec_r.trainable_variables)
        self.gen_comb_dec_s_grads = tape.gradient(total_gen_comb_dec_s_loss, self.gen_comb_dec_s.trainable_variables)

        self.disc_real_grads = tape.gradient(total_discriminator_loss_real, self.disc_real.trainable_variables)
        self.disc_synth_grads = tape.gradient(total_discriminator_loss_synth, self.disc_synth.trainable_variables)


        # Apply the Gradients to the Optimizers
        self.gen_dec_r_opt.apply_gradients(zip(self.gen_dec_r_grads, self.gen_dec_r.trainable_variables))
        self.gen_dec_s_opt.apply_gradients(zip(self.gen_dec_s_grads, self.gen_dec_s.trainable_variables))
        
        self.gen_comb_dec_r_opt.apply_gradients(zip(self.gen_comb_dec_r_grads, self.gen_comb_dec_r.trainable_variables))
        self.gen_comb_dec_s_opt.apply_gradients(zip(self.gen_comb_dec_s_grads, self.gen_comb_dec_s.trainable_variables))

        self.disc_real_opt.apply_gradients(zip(self.disc_real_grads, self.disc_real.trainable_variables))
        self.disc_synth_opt.apply_gradients(zip(self.disc_synth_grads, self.disc_synth.trainable_variables))


        return losses, generated_images


    def train(  self, 
                path_real, 
                path_synth, 
                img_size=(128, 128), 
                epochs=50, 
                save_model_every=5, 
                save_images_every=5):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator()

        # train
        for epoch in range(epochs):
            print(f'\nEpoch: {epoch+1} / {epochs}')
            
            start = time()
            losses = np.zeros([4])

            data_real = self.get_batch_flow(data_generator, path_real, img_size)
            data_synth = self.get_batch_flow(data_generator, path_synth, tuple(3*dim for dim in img_size))

            n_batches_real = len(data_real) if len(data_real) % self.batch_size == 0 else len(data_real) - 1

            data_real = list(islice(data_real, n_batches_real))
            data_synth = list(islice(data_synth, n_batches_real))

            for i, (a, b) in enumerate(zip(data_real, data_synth)):
                print(f'\tBatch: {i+1} / {n_batches_real}\r', end='')

                a = self.normalize(a)
                b = self.normalize(b)

                b1, b2, b3 = self.split_to_attributes(b)

                batch_losses, generated_images = self.train_step(a, b1, b2, b3)
                batch_losses = [
                    batch_losses['recostruction'], 
                    batch_losses['disentanglement'], 
                    batch_losses['cycle'], 
                    batch_losses['attribute cycle']
                ]

                losses = np.add(losses, batch_losses)

                if save_images_every and epoch % save_images_every == 0:
                    self.save(b1, b2, b3, generated_images['disentangled b3'], i, epoch)


            losses = losses / n_batches_real

            if save_model_every and epoch % save_model_every == 0:
                ckpt_path = self.ckpt_manager.save()
                print(f'\tSaving checkpoint for epoch {epoch+1} at {ckpt_path}\n')

            print(f'\tReconstruction Loss:\t{losses[0]}')
            print(f'\tDisentanglement Loss:\t{losses[1]}')
            print(f'\tCycle Loss:\t\t{losses[2]}')
            print(f'\tAttribute Cycle Loss:\t{losses[3]}')

            print(f'\n\tTime taken for epoch {epoch+1}: {time()-start}sec.')




