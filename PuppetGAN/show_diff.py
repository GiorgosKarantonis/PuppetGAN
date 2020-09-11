import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def normalize(x):
    return x / 255


def show(x, title=''):
    plt.imshow(x, cmap='hot', interpolation='nearest')
    plt.title(title)
    # plt.imshow(x)
    plt.show()


full_img = '../results/epoch_700/58_1.png'

img = normalize(np.array(Image.open(full_img)))

original = img[:128, :128]
reconstructed = img[128:2*128, :128]
cycled = img[4*128:5*128, :128]
attr_intermediate = img[5*128:6*128, -128:]
attr_cycled = img[-128:, :128]

show(np.abs(original - attr_intermediate)[:, :, 1], title='|Original - Intermediate|')
show(np.abs(reconstructed - attr_intermediate)[:, :, 1], title='|Reconstructed - Intermediate|')
show(np.abs(cycled - attr_intermediate)[:, :, 1], title='|Cycled - Intermediate|')
show(np.abs(attr_cycled - attr_intermediate)[:, :, 1], title='|Attribute Cycled - Intermediate|')
