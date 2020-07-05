'''
	Apapted from: 
	https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb
'''

import time

import tensorflow as tf
import tensorflow_datasets as tfds

import puppetGAN as ppt



tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']

BUFFER_SIZE = 1000



def random_crop(image):
	cropped_image = tf.image.random_crop(image, size=[256, 256, 3])

	return cropped_image


def normalize(image):
	# normalizing the images to [-1, 1]
	image = tf.cast(image, tf.float32)
	image = (image / 127.5) - 1
	return image


def random_jitter(image):
	# resizing to 286 x 286 x 3
	image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	# randomly cropping to 256 x 256 x 3
	image = random_crop(image)

	# random mirroring
	image = tf.image.random_flip_left_right(image)

	return image


def preprocess_image_train(image, label):
	image = random_jitter(image)
	image = normalize(image)
	
	return image


def preprocess_image_test(image, label):
	return normalize(image)



# prepare datasets
train_horses = train_horses.map(	preprocess_image_train, 
									num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
train_zebras = train_zebras.map(	preprocess_image_train, 
									num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)

test_horses = test_horses.map(	preprocess_image_test, 
								num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_zebras = test_zebras.map(	preprocess_image_test, 
								num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)



CHECKPOINT_PATH = "./checkpoints/train"

puppet_GAN = ppt.PuppetGAN()

# setup checkpoints' structure
ckpt = tf.train.Checkpoint(	generator_g=puppet_GAN.generator_g,
							generator_f=puppet_GAN.generator_f,
							discriminator_x=puppet_GAN.discriminator_x,
							discriminator_y=puppet_GAN.discriminator_y,
							generator_g_optimizer=puppet_GAN.generator_g_optimizer,
							generator_f_optimizer=puppet_GAN.generator_f_optimizer,
							discriminator_x_optimizer=puppet_GAN.discriminator_x_optimizer,
							discriminator_y_optimizer=puppet_GAN.discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
	ckpt.restore(ckpt_manager.latest_checkpoint)
	print ('Latest checkpoint restored!')


EPOCHS = 40

# train
for epoch in range(EPOCHS):
	print(f'{epoch+1} / {EPOCHS}')
	start = time.time()

	for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
		gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = puppet_GAN.train_step(image_x, image_y)

	if epoch % 5 == 0:
		ckpt_save_path = ckpt_manager.save()
		print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')


	print(f'Generator G Loss: {gen_g_loss}\nGenerator F Loss: {gen_f_loss}')
	print(f'Discriminator X Loss: {disc_x_loss}\nDiscriminator Y Loss: {disc_y_loss}')
	print (f'Time taken for epoch {epoch+1} is {time.time()-start} sec\n')







