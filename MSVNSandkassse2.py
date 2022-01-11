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



    file_list = ['Vibrant.jpg', 'landscape2.jpg','landscape.jpg','CarWorld-16.jpg','rollercoaster.jpg']

    filename = file_list[1]

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










    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 512, 1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 128, 1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 3, 1, padding='same')
                )

        def forward(self, input):
            return self.model(input)


    gen = Generator().to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(gen.parameters(), lr=0.0001, momentum=0.5)
    optimizer = torch.optim.Adam(gen.parameters(),lr = 0.0001)

    reruns = 10000

    for epoch in range(reruns):

        optimizer.zero_grad()

        data = image1_grayscale.to(device)
        output = gen(data)

        loss = criterion(output, image1.to(device))

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch = {epoch},  Loss: {loss.item()}')


    new = gen.forward(image1_grayscale.to(device))

    img = image1_grayscale_for_display
    img = np.transpose(img, (1, 2, 0))

    img2 = torch.squeeze(new)
    img2 = img2.cpu()
    img2 = img2.detach().numpy()
    img2 = np.transpose(img2, (1, 2, 0))
    img2 = lab2rgb(img2)


    rows = 1
    columns = 2
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img2)
    plt.title('Generated')
    plt.show()


    genfig = plt.figure(figsize=(10,7))
    for index, filename2 in enumerate(file_list):
        generated_image = imread(filename2)
        generated_image = rgb2lab(generated_image).astype('float32')
        generated_image = generated_image = generated_image[..., 0]
        generated_image = transform(generated_image)
        generated_image = torch.unsqueeze(generated_image, 0)
        gened_img3 = gen.forward(generated_image.to(device))

        GenImg_p2 = torch.squeeze(gened_img3)
        GenImg_p2 = GenImg_p2.cpu()
        GenImg_p2 = GenImg_p2.detach().numpy()
        GenImg_p2 = np.transpose(GenImg_p2, (1, 2, 0))
        GenImg_p2 = lab2rgb(GenImg_p2)

        genfig.add_subplot(1,len(file_list),index+1)
        plt.imshow(GenImg_p2)
    plt.show()











