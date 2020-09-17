import click
import puppetGAN as puppet



MODE = 'train' # use 'train' or 'test'
DATASET = 'digits' # use 'digits', 'mouth' or 'light'

EPOCHS = 500
BATCH_SIZE, IMG_SIZE = (200, (32, 32)) if DATASET == 'digits' else (30, (128, 128))
SAVE_IMG_EVERY = 20
SAVE_MODEL_EVERY = 100
ROIDS = False



if __name__ == '__main__':
    real_path = f'../data/{DATASET}/real_'
    synth_path = f'../data/{DATASET}/synth_'

    puppet_GAN = puppet.PuppetGAN(IMG_SIZE)
    puppet_GAN.restore_checkpoint()

    if MODE.lower() == 'test':
        puppet_GAN.eval(f'../data/{DATASET}/rows_')
    elif MODE.lower() == 'train':
        puppet_GAN.fit(path_real=real_path,
                       path_synth=synth_path,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       save_images_every=SAVE_IMG_EVERY,
                       save_model_every=SAVE_MODEL_EVERY,
                       use_roids=ROIDS)
    else:
        raise ValueError('Wrong mode...')
