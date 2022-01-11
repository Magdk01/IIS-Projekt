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

    Size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((Size))])

    transform2 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((Size)),
         transforms.Grayscale(3)])

    grsctrans = transforms.Compose(
        [transforms.Grayscale()]
    )




    filename = ('landscape2.jpg')
    filename2 = ('CarWorld-16.jpg')

    image1 = imread(filename)


    image1 = rgb2lab(image1).astype('float32')
    image1_grayscale = np.copy(image1)

    # image1_grayscale[...,1]=image1_grayscale[...,2]=0
    image1_grayscale = image1_grayscale[...,0]
    image1_grayscale_for_display = np.copy(image1_grayscale)
    image1_grayscale_for_display = transform(image1_grayscale_for_display)
    image1_grayscale_for_display = image1_grayscale_for_display.numpy()


    image1 = transform(image1)
    image1_grayscale = transform(image1_grayscale)


    image1 = torch.unsqueeze(image1, 0)
    image1_grayscale = torch.unsqueeze(image1_grayscale, 0)

    # image1_grayscale = torch.squeeze(image1_grayscale)
    # image1_grayscale = image1_grayscale.numpy()
    # image1_grayscale = np.transpose(image1_grayscale,(1,2,0))
    # plt.imshow(lab2rgb(image1_grayscale))
    # plt.show()





    image3 = imread(filename2)
    image3 = rgb2lab(image3).astype('float32')
    image3 = image3 = image3[...,0]
    image3 = transform(image3)
    image3 = torch.unsqueeze(image3, 0)




    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 255, 1, padding='same'),
                nn.ReLU(),
                # nn.BatchNorm2d(255),
                nn.Conv2d(255, 50, 1, padding='same'),

                nn.Conv2d(50, 25, 1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(25, 3, 1, padding='same'))

        def forward(self, input):
            return self.model(input)


    gen = Generator().to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(gen.parameters(), lr=0.0001, momentum=0.5)
    optimizer = torch.optim.Adam(gen.parameters(),lr = 0.0001)
    epochs = 30000

    for epoch in range(epochs):

        optimizer.zero_grad()

        data = image1_grayscale.to(device)
        output = gen(data)

        loss = criterion(output, image1.to(device))

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch = {epoch},  Loss: {loss.item()}')


    new = gen.forward(image1_grayscale.to(device))

    newdiff = gen.forward(image3.to(device))



    img = image1_grayscale_for_display
    img = np.transpose(img,(1,2,0))

    img2 = torch.squeeze(new)
    img2 = img2.cpu()
    img2 = img2.detach().numpy()
    img2 = np.transpose(img2,(1,2,0))
    img2 = lab2rgb(img2)


    img3 = torch.squeeze(newdiff)
    img3 = img3.cpu()
    img3 = img3.detach().numpy()
    img3 = np.transpose(img3,(1,2,0))
    img3 = lab2rgb(img3)


    rows = 1
    columns = 3
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img,cmap='gray')
    plt.title('Original')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img2)
    plt.title('Generated')
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img3)
    plt.title('new')


    plt.show()
