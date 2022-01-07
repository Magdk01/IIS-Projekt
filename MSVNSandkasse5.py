import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    transform = transforms.Compose(
        [
         transforms.Resize((64))

        ])

    grsctrans = transforms.Compose(
        [transforms.Grayscale()]
    )

    batch_size = 1



    filename = ('landscape.jpg')
    filename2 = ('CarWorld-16.jpg')

    im = imread(filename)
    im1 = rgb2lab(im)
    im1[..., 0] = im1[..., 0] = 0
    im1=lab2rgb(im1)

    # plt.figure(figsize=(20, 10))
    # plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('Original image', size=20)
    plt.imshow(im1), plt.axis('off')
    plt.show()

