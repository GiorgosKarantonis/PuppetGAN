'''
	Partially apapted from: 
	https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb
'''

import numpy as np
import tensorflow as tf

import models



class PuppetGAN:
	def __init__(	self, 
					batch_size=1, 
					img_width=256, 
					img_height=256, 
					output_channels=3, 
					lamda=10):
		
		self.batch_size = batch_size
		self.img_width = img_width
		self.img_height = img_height

		self.output_channels = output_channels
		self.lamda = lamda

		self.generator_g = self.create_generator()
		self.generator_f = self.create_generator()
		
		self.discriminator_x = self.create_discriminator()
		self.discriminator_y = self.create_discriminator()

		self.gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
		self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

		self.generator_g_gradients = None
		self.generator_f_gradients = None
		self.discriminator_x_gradients = None
		self.discriminator_y_gradients = None

		self.checkpoint_path = "./checkpoints/train"

		self.ckpt = tf.train.Checkpoint(	generator_g=self.generator_g,
											generator_f=self.generator_f,
											discriminator_x=self.discriminator_x,
											discriminator_y=self.discriminator_y,
											generator_g_optimizer=self.generator_g_optimizer,
											generator_f_optimizer=self.generator_f_optimizer,
											discriminator_x_optimizer=self.discriminator_x_optimizer,
											discriminator_y_optimizer=self.discriminator_y_optimizer)

		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)


	def restore_checkpoint(self):
		if self.ckpt_manager.latest_checkpoint:
			self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
			print ('Latest checkpoint restored!')


	def create_generator(self, generator_type='pix2pix', **kwargs):
		if generator_type == 'pix2pix':
			if 'norm_type' not in kwargs:
				kwargs['norm_type'] = 'instancenorm'

			if 'drop_skips' not in kwargs:
				kwargs['drop_skips'] = True

			if 'bottleneck' not in kwargs:
				kwargs['bottleneck'] = 128

			generator = models.pix2pix_generator(	img_width=self.img_width, 
													img_height=self.img_height, 
													output_channels=self.output_channels, 
													batch_size=self.batch_size, 
													norm_type=kwargs['norm_type'], 
													drop_skips=kwargs['drop_skips'], 
													bottleneck=kwargs['bottleneck'])
		else:
			raise NotImplementedError()


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


	def cycle_loss(self, real_image, cycled_image):
		loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
		
		return self.lamda * loss


	def identity_loss(self, real_image, same_image):
		loss = tf.reduce_mean(tf.abs(real_image - same_image))
		
		return self.lamda * 0.5 * loss


	def reconstruction_loss(self, real_image, generated, norm_kind=2):
		return tf.norm((generated - real_image), ord=norm_kind)


	@tf.function
	def train_step(self, real_x, real_y):
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
		
		# Apply the gradients to the optimizer
		self.generator_g_optimizer.apply_gradients(zip(self.generator_g_gradients, self.generator_g.trainable_variables))

		self.generator_f_optimizer.apply_gradients(zip(self.generator_f_gradients, self.generator_f.trainable_variables))
		
		self.discriminator_x_optimizer.apply_gradients(zip(self.discriminator_x_gradients, self.discriminator_x.trainable_variables))
		
		self.discriminator_y_optimizer.apply_gradients(zip(self.discriminator_y_gradients, self.discriminator_y.trainable_variables))


		return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss









