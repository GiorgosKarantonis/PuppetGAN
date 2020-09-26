# Better Cross-Domain Feature Disentanglement and Manipulation with Improved PuppetGAN

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_roids_190_.gif" width="100%">
  <em>Quite cool... Right?</em>
</p>

## Introduction

This repo contains a tensorflow implementation of [PuppetGAN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf) as well as **an improved version of it, capable of manipulating features up to 100% better and up to 300% faster!** 😎

**PuppetGAN is** model that extends the CycleGAN idea and is **capable of extracting and manipulating features from a domain using examples from a different domain**. On top of that, one amazing aspect of PuppetGAN is that it **does not require a great amount of data**; the biggest dataset I used contained 5000 sets of examples while the smallest one **just slightly over 1000 sets of examples**!

## The Model(s)

### Overview

PuppetGAN consists of 4 different components; one that is responsible for learning to reconstruct the input images, one that is responsible for learning to disentangle the the *Attribute of Interest*, a CycleGAN component and an Attribute CycleGAN. The Attribute CycleGAN acts in a similar manner to CycleGAN with the exception that it deals with cross-domain inputs. 

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/PuppetGAN.png" width="80%">
  <br></br>
  <em>The full architecture of the baseline PuppetGAN (the image is copied from the original paper)</em>
</p>

**With this repo I add a few more components, which I call Roids, that greatly improve the performance of the baseline PuppetGAN**. One *Roid* is applied in the *disentanglement* part and the rest in the *attribute cycle* part while **the objective of all of them is pretty much the same; to guarantee better disentanglement!**

* The original architecture performs the disentanglement only in the synthetic domain and this ability is passed to the real domain through implicitly. *The disentanglement *Roid* takes advantage of the CycleGAN model and performs the disentanglement in the translations of the synthetic images passing the ability explicitly to the real domain.*

* The attribute cycle *Roids* act in a similar way, but they instead force the attributes, other that the *Attribute of Interest*, of the cross-domain translations to be as precise as possible.

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/roid_dis.png" width="25%">
  <br></br>
  <em>The Disentanglement Roid</em>
  <br></br>
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/roids_attr.png" width="60%">
  <br></br>
  <em>The Attribute Cycle Roids</em>
</p>

### Implementation

The only difference between my baseline and the model from the paper is that my generators and discriminators are a modified version of the ones used in tensorflow's [CycleGAN tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan). The fact that the creators of PuppetGAN used ResNet blocks may be partially responsible for the memorization effect that seems to be present in some of the results of the paper due to the fact that the skip connections allow for information to be passed unchanged between different layers.

**Other than that, all my implementations use exactly the same parameters as the ones in the original model. Also, neither my architectures nor the parameters have been modified at all between different datasets.**

## Performance

**Both my baseline implementation and my proposed architecture(s) significantly outperform the original PuppetGAN!**

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_baseline.gif" width="100%">
  <em>Rotation of MNIST digits</em>
</p>

### By the Numbers

Just like in the original paper, all the reported score are for the MNIST dataset. Due to the fact that I didn't have access to the *size* dataset, I was able to measure the performance of my models only in the *rotation* dataset.

|PuppetGAN|Accuracy|<img src="https://render.githubusercontent.com/render/math?math=\bf{r_{attr}}">|<img src="https://render.githubusercontent.com/render/math?math=\bf{V_{rest}}">|Epoch|
|:----------------------------------------:|:----------------:|:----------:|:------:|:-------:|
| *Original (paper)*                       |      *0.97*      |   *0.40*   | *0.01* |   *-*   |
| My baseline                              |       0.96       |    0.59    |  0.01  |   300   |
| **Roids in Attribute Cycle Component**   |       0.96       |  **0.84**  |  0.02  | **100** |
| **Roids in Attribute Cycle Component**   |     **0.98**     |    0.77    |  0.02  |   150   |
| Roids in Disentanglement Component       |       0.91       |    0.73    |  0.01  |   250   |
| **Roids in Both Components**             |       0.97       |  **0.79**  |  0.01  |   300   |

* **Accuracy** *(The closer to 1 the better)*

The accuracy measures, using a LeNet-5 network, how well the original class is preserved. In other words, this metric is indicative of how well the model manages to disentangle without affecting the *rest* of the attributes. As we'll see later it is possible though to get very high accuracy while having suboptimal disentanglement performance...

* **<img src="https://render.githubusercontent.com/render/math?math=\bf{r_{attr}}">** *(The closer to 1 the better)*

This score is the correlation coefficient between the *Attribute of Interest* between the known and the generated images and it captures how well the model manipulates the *Attribute of Interest*.

* **<img src="https://render.githubusercontent.com/render/math?math=\bf{V_{rest}}">** *(The closer to 0 the better)*

This score captures how similar are the results between images that have identical the *Attribute of Interest* and different the *rest* of the attributes. For this metric I report the standard deviation instead of the variance, that it is mentioned in the paper, due to the fact that the variance of my models was magnitudes smaller than the one reported on the paper. This makes me believe that the standard deviation was used in the paper as well.

### Discussion about the Results

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.gif" width="100%">
  <em>Mouth manipulation after 440 epochs, using the Baseline.</em>
  <br></br>
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_roids_190_.gif" width="100%">
  <em>Mouth manipulation after 190 epochs with Roids in the Attribute Cycle component. The model learns to both open and close the mouth more accurately, disentangle in a better way, produce more clear images and all that way faster!</em>
</p>

The most well balanced model seems to be one that uses both kinds of *Roids*, since it achieves the same *accuracy* and 
<img src="https://render.githubusercontent.com/render/math?math=\textit{V_{rest}}"> 
score as the original model while **increasing the manipulation score by** more than 30% compared to my baseline implementation and almost **100% compared to the original paper**. Nevertheless, although it is intuitive that a combination of all the *Roids* would yield better results, I believe that more experiments are required to determine if its benefits are sufficient to outweigh the great speed up of the model that uses *Roids* only in the Attribute Cycle component.

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_roids.gif" width="100%">
  <em>MNIST rotation after adding Roids on the Attribute Cycle component</em>
</p>

For now, I would personally favor the model that uses only the *Roids* of the Attribute Cycle component due to the fact that it manages to outperform every other model in the *AoI* manipulation score **at the 1/3 of the time**, while having insignificant differences in the values of the other metrics.

A significant drawback of the original model is that seems to memorizes seen images instead of editing the given ones. This can be observed in the rotation results reported in the [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Usman_PuppetGAN_Cross-Domain_Image_Manipulation_by_Demonstration_ICCV_2019_paper.pdf) where the representation of a real digit may change during the rotation or different representations of a real digit may have the same rotated representations. This doesn't stop it though from having a very high accuracy, which highlights why this metric is not necessarily ideal for calculating the quality of the disentanglement.

<p align="center">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_paper.png" width="100%">
  <img src="https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mnist_paper_.gif" width="100%">
  <em>The rotation results of the paper</em>
</p>

Another issue with both the model of the paper and my models can be observed in the mouth dataset, where PuppetGAN confuses the microphone with the opening of the mouth; when the synthetic image dictates a wider opening, PuppetGAN moves the microphone closer to the mouth. This effect is slightly bigger in my baseline but I believe that it is due to the fact that I haven't done any hyper-parameter tuning; some experimentation with the magnitude of the noise or with the weights of the different components could eliminate it. Also, the model with *Roids* in the *Attribute of Interest* seems to deal with issue better than the baseline.

## Running the Code

You can manage all the dependencies with Pipenv using the provided [Pipfile](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/Pipfile). This allows for easier reproducibility of the code due to the fact that Pipenv creates a virtual environment containing all the necessary libraries. **Just run `pipenv shell` in the base directory of the project and you're ready to go!**

On the other hand, if for any reason you don't want to use Pipenv you can install all the required libraries using the provided `requirements.txt` file. Neither this file nor Pipenv take care of cuda though; in all my experiments I used `cuda 7.5.18`.

In order to download the datasets, you can use the `fetch_data.sh` script which downloads and extracts them in the correct directory, running:

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

Also, to help you keep better track of your process, every a certain number of epochs my model creates in `./PuppetGAN/results` a sample of [evaluation rows of generated images](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.png) along with [`gif` animations for these rows](https://github.com/GiorgosKarantonis/images/blob/master/PuppetGAN/mouth_baseline.gif) to visualize better the performance of your model. 

On top of that, in `./PuppetGAN/results` are also stored plots of both the supervised and the adversarial losses as well as the images that are produced during the training. This allows you to have in a single folder everything you need to evaluate an experiment, keep track of its progress and reproduce its results!

Unless you want to experiment with different architectures, [`PuppetGAN/config.json`](https://github.com/GiorgosKarantonis/PuppetGAN/blob/master/PuppetGAN/config.json) is the only file you'll need. This file allows you to control all the hyper-parameters of the model without having to look at any of code! More specifically, the parameters you can control are: 

* `dataset` : The dataset to use. You can choose between *"mnist"*, *"mouth"* and *"light"*.

* `epochs` : The number of epochs that the model will be trained for.

* `noise std` : The standard deviation of the noise that will be applied to the translated images. The mean of the noise is 0.

* `bottleneck noise` : The standard deviation of the noise that will be applied to the bottleneck. The mean of the noise is 0.

* `on roids` : Whether or not to use the proposed Roids.

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
