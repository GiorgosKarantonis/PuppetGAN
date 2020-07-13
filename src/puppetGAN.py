'''
	Partially apapted from: 
	https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb
'''

import numpy as np
import tensorflow as tf

import models



class PuppetGAN:
	def __init__(	self, 
					img_size=(256, 256), 
					batch_size=1, 
					output_channels=3, 
					lamda=10):
		
		self.img_height = img_size[0]
		self.img_width = img_size[1]
		self.batch_size = batch_size

		self.output_channels = output_channels
		self.lamda = lamda

		# self.gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		self.gan_loss = tf.keras.losses.MSE()

		self.encoder = self.init_encoder()
		self.decoder_real = self.init_decoder()
		self.decoder_synth = self.init_decoder()

		self.gen_dec_r, self.gen_dec_r_opt, self.gen_dec_r_grads = self.init_generator(encoder=self.encoder, decoder=self.decoder_real)
		self.gen_dec_s, self.gen_dec_s_opt, self.gen_dec_s_grads = self.init_generator(encoder=self.encoder, decoder=self.decoder_synth)
		self.gen_comb_dec_r, self.gen_comb_dec_r_opt, self.gen_comb_dec_r_grads = self.init_generator(	encoder=self.encoder, 
																										decoder=self.decoder_real, 
																										combine_inputs=True)
		self.gen_comb_dec_s, self.gen_comb_dec_s_opt, self.gen_comb_dec_s_grads = self.init_generator(	encoder=self.encoder, 
																										decoder=self.decoder_synth, 
																										combine_inputs=True)

		# JUST FOR TESTING, DELETE IT WHEN DONE
		self.discriminator_x, self.discriminator_x_optimizer, self.discriminator_x_gradients = self.init_discriminator()
		self.discriminator_y, self.discriminator_y_optimizer, self.discriminator_y_gradients = self.init_discriminator()

		self.checkpoint_path = "./checkpoints/train"
		self.ckpt, self.ckpt_manager = self.define_checkpoints()

	
	def define_checkpoints(self):
		ckpt = tf.train.Checkpoint(	generator_decoder_real=self.gen_dec_r, 
									generator_decoder_synth=self.gen_dec_r, 
									generator_combined_decoder_real=self.gen_comb_dec_r, 
									generator_combined_decoder_synth=self.gen_comb_dec_s, 
									discriminator_x=self.discriminator_x,
									discriminator_y=self.discriminator_y,
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


	def init_encoder(self, norm_type='instancenorm', use_bottleneck=True):
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


	def init_decoder(self, norm_type='instancenorm'):
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
			kwargs['norm_type'] = 'instancenorm'

		generator = models.generator(	self.img_height, 
										self.img_width, 
										encoder=encoder, 
										decoder=decoder, 
										combine_inputs=combine_inputs, 
										output_channels=self.output_channels, 
										batch_size=self.batch_size, 
										norm_type=kwargs['norm_type'])


		return generator


	def create_discriminator(self, discriminator_type='pix2pix', **kwargs):
		if discriminator_type == 'pix2pix':
			if 'norm_type' not in kwargs:
				kwargs['norm_type'] = 'instancenorm'

			if 'target' not in kwargs:
				kwargs['target'] = False

			discriminator = models.pix2pix_discriminator(	norm_type=kwargs['norm_type'], 
															target=kwargs['target'])
		else:
			raise NotImplementedError


		return discriminator


	def discriminator_loss(self, real, generated):
		real_loss = self.gan_loss(tf.ones_like(real), real)
		generated_loss = self.gan_loss(tf.zeros_like(generated), generated)

		total_disc_loss = real_loss + generated_loss

		return total_disc_loss * 0.5


	def generator_loss(self, generated):
		return self.gan_loss(tf.ones_like(generated), generated)


	def cycle_loss(self, real, cycled):
		loss = tf.reduce_mean(tf.abs(real - cycled))
		
		return self.lamda * loss


	def identity_loss(self, real, same_image):
		loss = tf.reduce_mean(tf.abs(real - same_image))
		
		return self.lamda * 0.5 * loss


	def l_p_loss(self, real, generated, p=2, weight=1):
		return weight * tf.norm((real - generated), ord=p)


	@tf.function
	def train_step_test(self, a, b1, b2, b3):
		b = tf.keras.layers.concatenate([b1, b2, b3])

		with tf.GradientTape(persistent=True) as tape:
			# persistent=True because the tape is used more than once to calculate the gradients

			# Reconstruction Loss
			a_hat = self.gen_dec_r(a, training=True)
			b_hat = self.gen_dec_s(b, training=True)

			reconstruction_loss_a = l_p_loss(a, a_hat, p=1)
			reconstruction_loss_b = l_p_loss(b, b_hat, p=1)
			
			total_reconstruction_loss = reconstruction_loss_a + reconstruction_loss_b


			# Dissentaglement Loss
			b3_hat = self.gen_comb_dec_s(b2, b1, training=True)

			disentanglement_loss_b3 = l_p_loss(b3, b3_hat, p=1)
			
			total_disentanglement_loss = disentanglement_loss_b3


			# Cycle Loss
			b_cycled_tilde = self.gen_dec_s(a, training=True)
			a_cycled_hat = self.gen_dec_r(b_cycled_tilde, training=True)

			a_cycled_tilde = self.gen_dec_r(b, training=True)
			b_cycled_hat = self.gen_dec_s(a_cycled_tilde, training=True)

			cycle_loss_a = self.l_p_loss(a, a_cycled_hat, p=1)
			cycle_loss_b = self.l_p_loss(b, b_cycled_hat, p=1)
			
			total_cycle_loss = cycle_loss_a + cycle_loss_b


			# Attribute Cycle Loss
			a_tilde = self.gen_comb_dec_r(a, b1, training=True)
			b3_hat_star = self.gen_comb_dec_s(b2, a_tilde, training=True)

			b_tilde = self.gen_comb_dec_s(b1, a, training=True)
			a_hat_star = self.gen_comb_dec_r(a, b_tilde, training=True)

			attr_cycle_loss_b3 = l_p_loss(b3, b3_hat_star, p=1)
			attr_cycle_loss_a = l_p_loss(a, a_hat_star, p=1)
			
			total_attr_cycle_loss = attr_cycle_loss_b3 + attr_cycle_loss_a


			# Supervised Losses per Generator
			total_gen_dec_r_loss = reconstruction_loss_a + total_cycle_loss
			total_gen_dec_s_loss = reconstruction_loss_b + total_cycle_loss
			total_gen_comb_dec_r_loss = total_attr_cycle_loss
			total_gen_comb_dec_s_loss = total_disentanglement_loss + total_attr_cycle_loss


		# Calculate the Gradients
		self.gen_dec_r_grads = tape.gradient(total_gen_dec_r_loss, self.gen_dec_r.trainable_variables)
		self.gen_dec_s_grads = tape.gradient(total_gen_dec_s_loss, self.gen_dec_s.trainable_variables)
		self.gen_comb_dec_r_grads = tape.gradient(total_gen_comb_dec_r_loss, self.gen_comb_dec_r.trainable_variables)
		self.gen_comb_dec_s_grads = tape.gradient(total_gen_comb_dec_s_loss, self.gen_comb_dec_s.trainable_variables)


		# Apply the Gradients to the Optimizers
		self.gen_dec_r_opt.apply_gradients(zip(self.gen_dec_r_grads, self.gen_dec_r.trainable_variables))
		self.gen_dec_s_opt.apply_gradients(zip(self.gen_dec_s_grads, self.gen_dec_s.trainable_variables))
		self.gen_comb_dec_r_opt.apply_gradients(zip(self.gen_comb_dec_r_grads, self.gen_comb_dec_r.trainable_variables))
		self.gen_comb_dec_s_opt.apply_gradients(zip(self.gen_comb_dec_s_grads, self.gen_comb_dec_s.trainable_variables))


		return total_reconstruction_loss, total_disentanglement_loss, total_cycle_loss, total_attr_cycle_loss


	@tf.function
	def train_step_old(self, real_x, real_y):
		# persistent is set to True because the tape is used more than
		# once to calculate the gradients.
		with tf.GradientTape(persistent=True) as tape:
			# Generator G translates X -> Y
			# Generator F translates Y -> X.
			
			fake_y = self.generator_g(real_x, training=True)
			cycled_x = self.generator_f(fake_y, training=True)

			fake_x = self.generator_f(real_y, training=True)
			cycled_y = self.generator_g(fake_x, training=True)

			# same_x and same_y are used for identity loss.
			same_x = self.generator_f(real_x, training=True)
			same_y = self.generator_g(real_y, training=True)

			disc_real_x = self.discriminator_x(real_x, training=True)
			disc_real_y = self.discriminator_y(real_y, training=True)

			disc_fake_x = self.discriminator_x(fake_x, training=True)
			disc_fake_y = self.discriminator_y(fake_y, training=True)

			# calculate the loss
			gen_g_loss = self.generator_loss(disc_fake_y)
			gen_f_loss = self.generator_loss(disc_fake_x)
			
			total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)
			
			# Total generator loss = adversarial loss + cycle loss
			total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
			total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

			disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
			disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
		

		# Calculate the gradients for generator and discriminator
		self.generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
		self.generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
		
		self.discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
		self.discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)
		

		# Apply the gradients to the optimizers
		self.generator_g_optimizer.apply_gradients(zip(self.generator_g_gradients, self.generator_g.trainable_variables))
		self.generator_f_optimizer.apply_gradients(zip(self.generator_f_gradients, self.generator_f.trainable_variables))
		
		self.discriminator_x_optimizer.apply_gradients(zip(self.discriminator_x_gradients, self.discriminator_x.trainable_variables))
		self.discriminator_y_optimizer.apply_gradients(zip(self.discriminator_y_gradients, self.discriminator_y.trainable_variables))


		return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


	def train(self, epochs=40):
		raise NotImplementedError









