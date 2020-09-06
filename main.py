import puppetGAN as puppet



DATASET = 'faces'
EPOCHS = 1000



if __name__ == '__main__':
    dataset = DATASET.lower()
    assert dataset in ['faces', 'digits']
    
    batch_size, img_size, save_images_every, save_model_every = (30, (128, 128), 20, 5) if dataset == 'faces' else (200, (32, 32), 20, 20)

    faces_real_path = f'data/{dataset}/real_'
    faces_synth_path = f'data/{dataset}/synth_'

    puppet_GAN = puppet.PuppetGAN()
    puppet_GAN.restore_checkpoint()

    # puppet_GAN.get_face_rows(img_size=img_size)

    puppet_GAN.fit(path_real=faces_real_path,
                   path_synth=faces_synth_path,
                   batch_size=batch_size,
                   epochs=EPOCHS,
                   img_size=img_size,
                   save_images_every=save_images_every,
                   save_model_every=save_model_every)




