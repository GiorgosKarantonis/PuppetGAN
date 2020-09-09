import puppetGAN as puppet



DATASET = 'faces'
EPOCHS = 1000
BATCH_SIZE, IMG_SIZE, SAVE_IMG_EVERY, SAVE_MODEL_EVERY = (30, (128, 128), 20, 5) if DATASET == 'faces' else (200, (32, 32), 20, 20)



if __name__ == '__main__':
    real_path = f'../data/{DATASET}/real_'
    synth_path = f'../data/{DATASET}/synth_'

    puppet_GAN = puppet.PuppetGAN()
    puppet_GAN.restore_checkpoint()

    # puppet_GAN.get_face_rows(img_size=IMG_SIZE)

    puppet_GAN.fit(path_real=real_path,
                   path_synth=synth_path,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   img_size=IMG_SIZE,
                   save_images_every=SAVE_IMG_EVERY,
                   save_model_every=SAVE_MODEL_EVERY)




