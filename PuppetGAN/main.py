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
              help='Which checkpoint to restore. Leave to -1 for the latest.')
def main(test, ckpt):
    dataset_mapping = {
        'mnist' : 'digits',
        'usps' : 'digits',
        'mouth' : 'faces',
        'light' : 'faces'
    }

    if ckpt != -1:
        ckpt = f'ckpt-{ckpt}'

    # load the configuration file
    with open('config.json') as config:
        hyperparams = json.load(config)

    # universal hyperparameters
    DATASET = hyperparams['dataset']
    EPOCHS = hyperparams['epochs']
    NOISE_STD = hyperparams['noise std']
    BOTTLENECK_NOISE = hyperparams['bottleneck noise']
    ROIDS = hyperparams['on roids']

    # dataset-specific hyperparameters
    BATCH_SIZE = hyperparams[dataset_mapping[DATASET]]['batch size']
    IMG_SIZE = tuple(hyperparams[dataset_mapping[DATASET]]['image size'])
    SAVE_IMG_EVERY = hyperparams[dataset_mapping[DATASET]]['save images every']
    SAVE_MODEL_EVERY = hyperparams[dataset_mapping[DATASET]]['save model every']

    # define the required directories
    real_path = f'../data/{DATASET}/real_'
    synth_path = f'../data/{DATASET}/synth_'
    eval_path = f'../data/{DATASET}/rows_'

    # load the model
    puppet_GAN = puppet.PuppetGAN(img_size=IMG_SIZE,
                                  noise_std=NOISE_STD,
                                  bottleneck_noise=BOTTLENECK_NOISE)

    puppet_GAN.restore_checkpoint(ckpt=ckpt)

    if test:
        puppet_GAN.eval(f'../data/{DATASET}/rows_',sample=None)
    else:
        puppet_GAN.fit(path_real=real_path,
                       path_synth=synth_path,
                       path_eval=eval_path,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       save_images_every=SAVE_IMG_EVERY,
                       save_model_every=SAVE_MODEL_EVERY,
                       use_roids=ROIDS)



if __name__ == '__main__':
    main()
