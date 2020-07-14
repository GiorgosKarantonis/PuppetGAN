import os
import shutil
import numpy as np

from PIL import Image



DIGITS_REAL_PATH = 'data/real-digits'
DIGITS_SYNTH_PATH = 'data/syn-rot'
DIGITS_PATH_TARGET = 'data/digits/'

FACES_REAL_PATH = 'data/real-face'
FACES_SYNTH_PATH = 'data/syn-face'
FACES_PATH_TARGET = 'data/faces/'



def create_required_dirs(home, home_target):
    for feature in ['real', 'attr', 'rest', 'full']:
        new_path = os.path.join(home_target, feature)
        os.makedirs(new_path)

    os.rename(home, os.path.join(home_target, 'real'))


def split_synthetic_data(path, target_path):
    for file in os.listdir(path):
        if file.endswith('.png'):
            img_np = np.array(Image.open(f'{path}/{file}'))

            h_window = int(img_np.shape[0] / 3)
            w_window = int(img_np.shape[1] / 3)

            img_rest = Image.fromarray(img_np[:h_window, :w_window, :]).convert('RGB')
            img_attr = Image.fromarray(img_np[h_window:2*h_window, :w_window, :]).convert('RGB')
            img_full = Image.fromarray(img_np[2*h_window:, :w_window:, :]).convert('RGB')

            img_rest.save(os.path.join(target_path, 'rest', file))
            img_attr.save(os.path.join(target_path, 'attr', file))
            img_full.save(os.path.join(target_path, 'full', file))


def remove_leftover_dirs(*paths):
    for path in paths:
        shutil.rmtree(path)


def prepare_dataset(real_path, synth_path, new_base_dir):
    create_required_dirs(real_path, new_base_dir)
    split_synthetic_data(synth_path, new_base_dir)

    remove_leftover_dirs(synth_path)



if __name__ == '__main__':
    prepare_dataset(DIGITS_REAL_PATH, DIGITS_SYNTH_PATH, DIGITS_PATH_TARGET)
    prepare_dataset(FACES_REAL_PATH, FACES_SYNTH_PATH, FACES_PATH_TARGET)













