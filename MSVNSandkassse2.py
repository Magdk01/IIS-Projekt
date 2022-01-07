import numpy as np
from PIL import Image
import torch
from PIL.Image import fromarray
from torch import from_numpy
from torchvision import transforms
import os
from skimage.color import rgb2lab, lab2rgb


filename = "600x600Sharingan.png"
device = torch.device('cuda')

class ImageProcessing:

    def __init__(self, filename):
        image = Image.open(filename).convert("RGB")
        image = transforms.Resize((64,64))(image)
        imagearray = np.array(image).astype("int")
        imagetensor = from_numpy(imagearray)
        self.imagetensor = imagetensor
        self.imagearray = imagearray
        self.image = image
        self.LABimg = rgb2lab(imagearray).astype("float32")
        self.grsc = image.convert("L").convert("RGB")
        self.grscarray = np.array(image.convert("L").convert("RGB")).astype("int")


    def printimage(self):

        print(self.image)
        self.grsc.show()

    def comparrison(self):
        total_loss = []
        loss_matrix = np.zeros([np.size(self.imagearray[0,::,0]),np.size(self.imagearray[::,0,0])])
        for columns in range(np.size(self.imagearray[0,::,0])):
            for rows in range(np.size(self.imagearray[::,0,0])):

                loss = sum(self.LABimg[columns,rows,::]-self.grscarray[columns,rows,::])
                print(loss)
                loss_matrix[columns,rows]=loss
                total_loss.append(loss)

        print(sum(total_loss))
        self.loss_matrix = loss_matrix






pic = ImageProcessing(filename)
rows = pic.comparrison()


print('test')
