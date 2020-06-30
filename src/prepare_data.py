import os
import shutil
import numpy as np

from PIL import Image



DIGITS_REAL_PATH = 'data/real-digits'
DIGITS_SYNTH_PATH = 'data/syn-rot'

FACES_REAL_PATH = 'data/real-face'
FACES_SYNTH_PATH = 'data/syn-face'



def create_required_dirs():
	os.makedirs('data/digits/real/')
	os.rename(DIGITS_REAL_PATH, 'data/digits/real/')

	os.makedirs('data/digits/attr/')
	os.makedirs('data/digits/rest/')
	os.makedirs('data/digits/attr_rest/')

	os.makedirs('data/faces/real/')
	os.rename(FACES_REAL_PATH, 'data/faces/real/')

	os.makedirs('data/faces/attr/')
	os.makedirs('data/faces/rest/')
	os.makedirs('data/faces/attr_rest/')


def remove_leftover_dirs(*paths):
	for path in paths:
		shutil.rmtree(path)


def fix_syn_digits(path=DIGITS_SYNTH_PATH):
	for file in os.listdir(path):
		if file.endswith('.png'):
			img_np = np.array(Image.open(f'{path}/{file}'))

			h_window = int(img_np.shape[0] / 3)
			w_window = int(img_np.shape[1] / 3)

			img_rest = Image.fromarray(img_np[:h_window, :w_window, :]).convert('RGB')
			img_attr = Image.fromarray(img_np[h_window:2*h_window, :w_window, :]).convert('RGB')
			img_attr_rest = Image.fromarray(img_np[2*h_window:, :w_window:, :]).convert('RGB')

			img_rest.save(f'data/digits/rest/{file}')
			img_attr.save(f'data/digits/attr/{file}')
			img_attr_rest.save(f'data/digits/attr_rest/{file}')


def fix_syn_faces(path=FACES_SYNTH_PATH):
	pass



create_required_dirs()
# fix_syn_digits()
fix_syn_faces()
# remove_leftover_dirs('data/syn-rot', 'data/syn-face')


