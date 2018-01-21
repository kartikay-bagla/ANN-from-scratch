from mnist_loader import *
import numpy as np #using numpy for arrays and optimized matrix multiplication
np.random.seed(1) #seeding so that repeated results are the same and we can observe
#changes from editing.
np.seterr(over = 'ignore')
np.set_printoptions(precision=0, suppress=True)
import pickle
import os.path
from matplotlib import pyplot as plt

def print_img(img):
    first_image = img
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

