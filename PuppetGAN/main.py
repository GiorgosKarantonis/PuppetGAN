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
    The entrypoint to PuppetGAN.
'''

import json
import click
import puppetGAN as puppet



@click.command()
@click.option('--test',
              '-t',
              is_flag=True,
              default=False,
              help='Whether or not to run on test mode.')
@click.option('--ckpt',
              '-c',
              default=-1,
              type=int,
              help='Which checkpoint to restore. Leave to -1 for the latest one.')
def main(test, ckpt):
    dataset_mapping = {
        'mnist' : 'digits',
        'usps' : 'digits',
        'mouth' : 'faces',
        'light' : 'faces'
    }


    # load the configuration file
    with open('config.json') as config:
        hyperparams = json.load(config)

    # universal hyperparameters
    DATASET = hyperparams['dataset']
    EPOCHS = hyperparams['epochs']
    NOISE_STD = hyperparams['noise std']
    BOTTLENECK_NOISE = hyperparams['bottleneck noise']
    ROIDS = hyperparams['on roids']

    # get the learning rates for the
    # different components of PuppetGAN
    REAL_GEN_LR = hyperparams['learning rates']['real generator']
    REAL_DISC_LR = hyperparams['learning rates']['real discriminator']
    SYNTH_GEN_LR = hyperparams['learning rates']['synthetic generator']
    SYNTH_DISC_LR = hyperparams['learning rates']['synthetic discriminator']

    # dataset-specific hyperparameters
    BATCH_SIZE = hyperparams[dataset_mapping[DATASET]]['batch size']
    IMG_SIZE = tuple(hyperparams[dataset_mapping[DATASET]]['image size'])
    SAVE_IMG_EVERY = hyperparams[dataset_mapping[DATASET]]['save images every']
    SAVE_MODEL_EVERY = hyperparams[dataset_mapping[DATASET]]['save model every']

    # get the weights for the different losses
    # they are used only during training
    RECONSTRUCTION_WEIGHT = hyperparams['losses weights']['reconstruction']
    DISENTANGLEMENT_WEIGHT = hyperparams['losses weights']['disentanglement']
    CYCLE_WEIGHT = hyperparams['losses weights']['cycle']
    ATTR_CYCLE_B3_WEIGHT = hyperparams['losses weights']['attribute cycle b3']
    ATTR_CYCLE_A_WEIGHT = hyperparams['losses weights']['attribute cycle a']


    # define the required directories
    real_path = f'../data/{DATASET}/real_'
    synth_path = f'../data/{DATASET}/synth_'
    eval_path = f'../data/{DATASET}/rows_'

    # load the model
    puppet_GAN = puppet.PuppetGAN(img_size=IMG_SIZE,
                                  noise_std=NOISE_STD,
                                  bottleneck_noise=BOTTLENECK_NOISE,
                                  real_gen_lr=REAL_GEN_LR,
                                  real_disc_lr=REAL_DISC_LR,
                                  synth_gen_lr=SYNTH_GEN_LR,
                                  synth_disc_lr=SYNTH_DISC_LR)
    
    if ckpt != -1:
        ckpt = f'ckpt-{ckpt}'

    if test:
        # restore only the weights
        # that are needed for evaluation
        puppet_GAN.restore_checkpoint(ckpt=ckpt, partial=True)
        # evaluate PuppetGAN
        puppet_GAN.eval(f'../data/{DATASET}/rows_',sample=None, target_folder=ckpt)
    else:
        # restore all the weights
        puppet_GAN.restore_checkpoint(ckpt=ckpt)
        # train PuppetGAN
        puppet_GAN.fit(path_real=real_path,
                       path_synth=synth_path,
                       path_eval=eval_path,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       save_images_every=SAVE_IMG_EVERY,
                       save_model_every=SAVE_MODEL_EVERY,
                       use_roids=ROIDS,
                       rec_weight=RECONSTRUCTION_WEIGHT,
                       dis_weight=DISENTANGLEMENT_WEIGHT,
                       cycle_weight=CYCLE_WEIGHT,
                       attr_cycle_b3_weight=ATTR_CYCLE_B3_WEIGHT,
                       attr_cycle_a_weight=ATTR_CYCLE_A_WEIGHT)



if __name__ == '__main__':
    main()
