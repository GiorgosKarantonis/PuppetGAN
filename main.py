import puppetGAN as puppet



FACES_REAL_PATH = 'data/dummy/real_'
FACES_SYNTH_PATH = 'data/dummy/synth_'

IMG_SIZE = (128, 128)
IMG_SIZE_SYNTH = (3*128, 3*128)
BATCH_SIZE = 50

EPOCHS = 40



if __name__ == '__main__':
    puppet_GAN = puppet.PuppetGAN(BATCH_SIZE)
    puppet_GAN.restore_checkpoint()
    puppet_GAN.train(   path_real=FACES_REAL_PATH, 
                        path_synth=FACES_SYNTH_PATH, 
                        epochs=EPOCHS)




