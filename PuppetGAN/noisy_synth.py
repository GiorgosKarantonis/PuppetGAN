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
	Adds noise to the background of the synthetic images.
'''

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



NOISE_STD = .1
TO_CONVERT = 'test'



def normalize(img):
    return (img / 127.5) - 1


def denormalize(img):
	return ((img + 1) * 127.5) / 255



if __name__ == '__main__':
	faces_path = '../data/mouth'

	if TO_CONVERT == 'train':
		synth_faces_path = os.path.join(faces_path, 'synth_/synth')
		noisy_synth_path = os.path.join(faces_path, 'noisy_/noisy2')
		scale_factor = 3
	elif TO_CONVERT == 'test':
		synth_faces_path = os.path.join(faces_path, 'rows_/synth')
		noisy_synth_path = os.path.join(faces_path, 'rows_/noisy')
		scale_factor = 10

	if not os.path.exists(noisy_synth_path):
		os.makedirs(noisy_synth_path)

		faces_counter = 0
		for f in os.listdir(synth_faces_path):
			if f.endswith('.png'):
				img = Image.open(os.path.join(synth_faces_path, f))
				img = np.array(img)
				img = normalize(img)

				noise = np.random.normal(scale=NOISE_STD, size=(scale_factor*128, scale_factor*128, 3))
				noise = np.where(noise < -1, -1, noise)
				noise = np.where(noise > 1, 1, noise)

				img = np.where(img == -1, noise, img)
				img = denormalize(img)

				img_name = f'{faces_counter}.png'
				plt.imsave(f'{os.path.join(noisy_synth_path, img_name)}', img)
				
				faces_counter += 1
	else:
		print('Noisy faces already exist.')
