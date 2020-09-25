# Cross-Domain Feature Disentanglement and Manipulation with PuppetGAN

<p align="center">
  <a href="#">
    <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.gif" width="100%">
    <em>Cool, right?</em>
  </a>
</p>

## Introduction

This repo contains a tensorflow implementation of [PuppetGAN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf) as well as **an improved version of it, capable of manipulating features up to 100% better and up to 300% faster!** 😎

**PuppetGAN is** model that builds on top of the CycleGAN idea and is **capable of extracting and manipulating a features from a domain using examples from a different domain**. On top of that, one amazing aspect of PuppetGAN is that it **does not require a great amount of data**; the biggest dataset I used contained 5000 sets of examples while the smallest one just **slightly over 1000 sets of examples**!

## The Model(s)



## Performance

**Both my baseline implementation and my proposed architecture(s) significantly outperform the original PuppetGAN!**

<p align="center">
  <a href="#">
    <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_baseline.gif" width="100%">
    <em>Rotation of MNIST digits</em>
  </a>
</p>

### By the Numbers

Just like in the original paper, all the reported score are for the MNIST dataset. Due to the fact that I didn't have access to the *size* dataset, I was able to measure the performance of my models only in the *rotation* dataset.

| PuppetGAN                                | Accuracy         | r_attr     | V_rest |  Epoch  |
|:----------------------------------------:|:----------------:|:----------:|:------:|:-------:|
| *Original (paper)*                       |      *0.97*      |   *0.40*   | *0.01* |   *-*   |
| My baseline                              |       0.96       |    0.59    |  0.01  |   300   |
| **Roids in Attribute Cycle Component**   |       0.96       |  **0.84**  |  0.02  | **100** |
| **Roids in Attribute Cycle Component**   |     **0.98**     |    0.77    |  0.02  |   150   |
| Roids in Disentanglement Component       |       0.91       |    0.73    |  0.01  |   250   |
| **Roids in Both Components**             |       0.97       |  **0.79**  |  0.01  |   300   |

#### Accuracy
The accuracy measures how well the original class is preserved, using a LeNet-5 network. In other words, this metric is indicative of how well the model manages to disentangle the Attribute of Interest without affecting the rest of the attributes. As we'll see later it is possible though to get very high accuracy while having suboptimal disentanglement performance...

The closer to 1 the better.

#### r_attr
This score is the correlation coefficient between the Attribute of Interest in the known and the generated images and it captures how well a model manipulates the attribute of interest.

The closer to 1 the better.

#### V_rest
This score captures how similar are the results between images that have identical the Attribute of Interest and different the rest of the attributes. For this metric I report the standard deviation instead of the variance that it is mentioned in the paper, due to the fact that the variance of my models was magnitudes smaller than the one reported on the paper. This makes me believe that the standard deviation was used in the paper as well.

The closer to 0 the better.

### Discussion about the Results

The most well balanced model seems to be one that uses both kinds of *roids*, since it achieves the same accuracy and V_rest score as the original model while **increasing the manipulation score by** more than 30% compared to my baseline implementation and almost **100% compared to the original paper**. Nevertheless, although it is intuitive that a combination of both *roids* would yield better results, I believe that more experiments are required to determine if its benefits are sufficient to outweigh the great speed up of the model that uses *roids* only in the Attribute Cycle component. 

For now, I would personally favor the model that uses only the *roid* in the Attribute Cycle component due to the fact that it manages to outperform every other model in the attribute manipulation score **at the 1/3 of the time**, while having seemingly insignificant differences in the values of the other metrics.

<p align="center">
  <a href="#">
    <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_roids.gif" width="100%">
    <em>After adding Roids on the Attribute Cycle component</em>
  </a>
</p>


A significant drawback of the original model is that it looks like it memorizes images instead of editing the given ones. This can be observed in the rotation results reported in the [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf) where the representation of a real digit may change during the rotation or different representations of a real digit may have the same rotated representations. This doesn't stop it though from having a very high accuracy, which shows why this metric is not necessarily ideal for calculating the quality of the disentanglement.

## Running the Code

*I used `cuda 7.5.18` in all my experiments.*

You can manage all the dependencies with Pipenv using the provided [Pipfile](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/Pipfile). This allows for easier reproducibility of the code due to the fact that Pipenv creates a virtual environment containing all the necessary libraries. **Just run `pipenv shell` in the base directory of the project and you're ready to go!**

On the other hand, if for any reason you don't want to use Pipenv, you can install all the required libraries using the provided `requirements.txt` file. 

In order to get the datasets, you can use the `fetch_data.sh` script which downloads them and extracts them in the correct directory, running:

```bash
. fetch_data.sh
```

Unfortunately, I am not allowed to publish any dataset other than `MNIST`, but feel free to ask the authors of the original PuppetGAN for them, following the instructions on [their website](http://ai.bu.edu/puppetgan/) 🙂.

### Training a Model
To start a new training, simply run:

```bash
python3 main.py
```

This will automatically look first for any existing checkpoints and will restore the latest one. If you want to continue the training from a specific checkpoint just run:

```bash
python3 main.py -c [checkpoint number]
```
or
```bash
python3 main.py --ckpt=[checkpoint number]
```

To help you keep better track of your work, every time you start a new training, a configuration report is created in [`./PuppetGAN/results/config.txt`](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/PuppetGAN/results/config.txt) which stores a detailed report of your current configuration. This report contains all your hyper-parameters and their respective values as well as the whole architecture of the model you are using, including every single layer, its parameters and how it is connected to the rest of the model.

Also, to help you keep better track of your process, during a certain number of epochs (which is specified by you), my model creates in `./PuppetGAN/results` a sample of [evaluation rows of generated images](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.png) along with [`gif` animations for these rows](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.gif) to visualize better how well accurate is the disentanglement and the manipulation of the attribute of interest. 

On top of all that, in `./PuppetGAN/results` are also stored plots of both the supervised and the adversarial losses as well as the images that are produced during the training. This allows you to have in a single folder everything you need to evaluate an experiment, keep track of its progress and reproduce its results!

Unless you want to experiment with different architectures, [`PuppetGAN/config.json`](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/PuppetGAN/config.json) is the only file you'll need. This file allows you to control all the hyper-parameters of the model without having to look at any of code! More specifically, the parameters you can control are: 

* `dataset` : The dataset to use. You can choose between *"mnist"*, *"mouth"* and *"light"*.

* `epochs` : The number of epochs that the model will be trained for.

* `noise std` : The standard deviation of the noise that will be applied to the translated images. The mean of the noise is 0.

* `bottleneck noise` : The standard deviation of the noise that will be applied to the bottleneck. The mean of the noise is 0.

* `on roids` : Whether or not to use the proposed roids.

* `learning rates`-`real generator` : The learning rate of the real generator.

* `learning rates`-`real discriminator` : The learning rate of the real discriminator

* `learning rates`-`synthetic generator` : The learning rate of the synthetic generator.

* `learning rate`-`synthetic discriminator` : The learning rate of the synthetic discriminator.

* `losses weights`-`reconstruction` : The weight of the reconstruction loss.

* `losses weights`-`disentanglement` : The weight of the disentanglement loss.

* `losses weights`-`cycle` : The weight of the cycle loss.

* `losses weights`-`attribute cycle b3` : The weight of part of the attribute cycle loss that is a function of the synthetic image that has both the Attribute of Interest and all the rest of the attributes.

* `losses weights`-`attribute cycle a` : The weight of part of the attribute cycle loss that is a function of the real image.

* `batch size` : The batch size. Depending on the kind of the dataset different values can be given.

* `image size` : At what size to resize the images of the dataset.

* `save images every` : Every how many epochs to save the training images and the sample of the evaluation images.

* `save model every` : Every how many epochs to create a checkpoint. Keep in mind that the 5 latest checkpoints are always kept during training.

### Evaluation of a Model
You can start an evaluation just by running:

```bash
python3 main.py -t
```
or
```bash
python3 main.py --test
```

Just like with training, this will look for the latest checkpoint; if you want to evaluate the performance of a different checkpoint you can simply use the `-c` and `--ckpt` options the same way as before.

During the evaluation process, the model creates all the [rows of the generated images](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.png), where each cell corresponds to the generated image for the respective synthetic and a real input. Additionally, for each of the evaluation images, [their corresponding `gif` file](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.gif) is also created to help you get a better idea of your results!

If you want to calculate the scores of your model in the `MNIST` dataset you can use my [`./PuppetGAN/eval_rotation.py`](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/PuppetGAN/eval_rotation.py) script, by running:

```bash
python3 eval_rotation.py -p [path to the directory of your evaluation images]
```
or
```bash
python3 eval_rotation.py -path=[path to the directory of your evaluation images]
```

You can also specify a path to save the evaluation report file using the option `-t` or `--target-path`. For example, let's say you have just trained and produced the evaluation images for a model and you want to get the evaluation scores for epoch 100 and save the report in the folder of this epoch. Then you should just run:

```bash
# make sure you are in ./PuppetGAN
python3 eval_rotation.py -p results/test/100/images -t results/test/100
```

For a fair comparison I am also providing the checkpoint of my `LeNet-5` network in [`./PuppetGAN/checkpoints/lenet5`](https://github.com/GiorgosKarantonis/PuppetGAN/tree/master/PuppetGAN/checkpoints/lenet5). If the `eval_rotation.py` script doesn't detect the checkpoint it will train one from scratch and in this case there may be a small difference in the accuracy of your model. 
