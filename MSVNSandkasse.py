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
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
from torch.nn import BCEWithLogitsLoss

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    Size = 32

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((Size)),
        ])

    grsctrans = transforms.Compose(
        [transforms.Grayscale()]
    )

    batch_size = 8

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=2)
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    #
    #
    #

    dataiter = (iter(trainloader))
    images, labels = dataiter.next()

    input_images = transforms.Grayscale(3)(images)

    fig = plt.figure(figsize=(10, 7))
    for index in range(batch_size):
        disp_images = transforms.ToPILImage()(images[index])
        fig.add_subplot(3, batch_size, index + 1)
        # plt.title(f'Original{index + 1}')
        plt.axis('off')
        plt.imshow(disp_images)

    for index in range(batch_size):
        disp_images = transforms.ToPILImage()(input_images[index])
        fig.add_subplot(3, batch_size, batch_size + index + 1)
        # plt.title(f'GRSC{index+1}')
        plt.axis('off')
        plt.imshow(disp_images)

    filename = ('landscape2.jpg')
    filename2 = ('CarWorld-16.jpg')

    image2 = Image.open(filename).convert('RGB')
    image2grsc = image2.convert('L')
    image2grsc = image2grsc.convert('RGB')

    image2 = transform(image2)
    image2grsc = transform(image2grsc)

    image2 = torch.unsqueeze(image2, 0)
    image2grsc = torch.unsqueeze(image2grsc, 0)


    # print(images)

    # filename = ('landscape2.jpg')
    # filename2 = ('CarWorld-16.jpg')

    # image1 = Image.open(filename).convert('RGB')
    # image2grsc = image1.convert('L')
    #
    # image2grsc = image1.convert('L')
    # image2grsc = image2grsc.convert('RGB')
    #
    #
    # image1 = transform(image1)
    # image2grsc = transform(image2grsc)
    #
    # image1 = torch.unsqueeze(image1, 0)
    # image2grsc = torch.unsqueeze(image2grsc, 0)
    #
    # images = image1

    # image3 = Image.open(filename2).convert('RGB')
    # image3 = transform(image3)
    # image3 = torch.unsqueeze(image3, 0)
    #
    # img = transforms.ToPILImage()((image2grsc[0]))

    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 255, 1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(255),
                nn.Conv2d(255, 50, 1, padding='same'),

                nn.Conv2d(50, 25, 1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(25, 3, 1, padding='same'))

        def forward(self, input):
            return self.model(input)


    learning_rate = 0.0001
    gen = Generator().to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(gen.parameters(), lr=learning_rate, momentum=0.5)
    optimizer = torch.optim.Adam(gen.parameters(),lr =learning_rate)


    # pre_epochs = 20000
    #
    # for epoch in range(pre_epochs):
    #
    #     optimizer.zero_grad()
    #
    #     output = gen(image2grsc.to(device))
    #
    #     loss = criterion(output, image2.to(device))
    #
    #     loss.backward()
    #
    #     optimizer.step()
    #
    #     if epoch % 500 == 0:
    #         print(f'Epoch = {epoch},  Loss: {loss.item()}')

    # fig2 = plt.figure(figsize=(10, 7))
    # for index in range(batch_size):
    #     disppreepoch_images = transforms.ToPILImage()(pre_trained[index])
    #     fig2.add_subplot(1, batch_size, 1 + index)
    #     # plt.title(f'Generated{index+1}')
    #     plt.axis('off')
    #     plt.imshow(disppreepoch_images)
    # plt.show()



    epochs = 10

    for epoch in range(epochs):

        for i, data in enumerate(trainloader):

            inp_img, labels = data
            fuckmig = inp_img
            haderprogrammering = transforms.Grayscale(3)(inp_img)

            optimizer.zero_grad()

            output = gen(haderprogrammering.to(device))

            loss = criterion(output, fuckmig.to(device))

            loss.backward()

            optimizer.step()

            if i % 500 == 0:
                print(f'Epoch = {epoch}, I = {i},  Loss: {loss.item()}')

        epoch_images = gen(input_images.to(device))

        fig3 = plt.figure(figsize=(10, 7))

        for index in range(batch_size):
            dispepoch_images = transforms.ToPILImage()(epoch_images[index])
            fig3.add_subplot(1, batch_size,1+index)
            # plt.title(f'Generated{index+1}')
            plt.axis('off')
            plt.imshow(dispepoch_images)
        plt.show()


    # new_images = gen.forward(input_images.to(device))
    #
    # for index in range(batch_size):
    #     disp_images = transforms.ToPILImage()(new_images[index])
    #     fig.add_subplot(3, batch_size, batch_size * 2 + index + 1)
    #     # plt.title(f'Generated{index+1}')
    #     plt.axis('off')
    #     plt.imshow(disp_images)
    #
    # plt.show()





    # imshow(torchvision.utils.make_grid(new_images))

    # new = gen.forward(images.to(device))
    #
    # newdiff = gen.forward(image3.to(device))
    #
    # img2 = transforms.ToPILImage()((new[0]))
    # img3 = transforms.ToPILImage()((newdiff[0]))
    #
    #
    # rows = 1
    # columns = 3
    # fig = plt.figure(figsize=(10, 7))
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(img)
    # plt.title('Original')
    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(img2)
    # plt.title('Generated')
    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(img3)
    # plt.title('new')
    #
    #
    # plt.show()
