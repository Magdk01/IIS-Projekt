import torchvision
from PIL import Image
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb
import torch
import os
import numpy as np
from skimage import io
from skimage import img_as_ubyte

# load data
if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    batch_size = 1

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    dataiter = (iter(trainloader))
    images, labels = dataiter.next()


    # download data to new folder
    for loader_data in trainloader:
        images, labels = loader_data
        datas = torch.squeeze(images)
        datas = datas.numpy()
        datas = np.transpose(datas,(1,2,0))
        datas = datas*255
        datas = datas.astype(np.uint8)
        lab = rgb2lab(datas)
        img = io.imread(lab)
        io.imsave('C:\\Users\\Jacob pc\\lab\\{}')
