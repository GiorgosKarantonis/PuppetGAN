import puppetGAN as puppet



IMG_SIZE = (128, 128)

DATASET = 'mouth' # use 'digits', 'mouth' or 'light'
EPOCHS = 1000
BATCH_SIZE, SAVE_IMG_EVERY, SAVE_MODEL_EVERY = (30, 20, 10) if DATASET == 'mouth' else (200, 20, 20)

MODE = 'train'



if __name__ == '__main__':
    real_path = f'../data/{DATASET}/real_'
    synth_path = f'../data/{DATASET}/synth_'

    puppet_GAN = puppet.PuppetGAN()
    puppet_GAN.restore_checkpoint()

    if MODE.lower() == 'test':
        puppet_GAN.get_face_rows()
    elif MODE.lower() == 'train':
        puppet_GAN.fit(path_real=real_path,
                       path_synth=synth_path,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       save_images_every=SAVE_IMG_EVERY,
                       save_model_every=SAVE_MODEL_EVERY)
    else:
        raise ValueError('Wrong mode...')
