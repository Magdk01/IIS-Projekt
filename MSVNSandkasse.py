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

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    Size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((Size))])

    grsctrans = transforms.Compose(
        [transforms.Grayscale()]
    )

    batch_size = 1

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)
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
    # dataiter = (iter(trainloader))
    # images, labels = dataiter.next()

    # print(images)

    filename = ('landscape2.jpg')
    filename2 = ('CarWorld-16.jpg')

    image2 = Image.open(filename).convert('RGB')
    image2grsc = image2.convert('L')
    image2grsc = image2grsc.convert('RGB')


    image2 = transform(image2)
    image2grsc = transform(image2grsc)

    image2 = torch.unsqueeze(image2, 0)
    image2grsc = torch.unsqueeze(image2grsc, 0)

    images = image2




    image3 = Image.open(filename2).convert('RGB')
    image3 = transform(image3)
    image3 = torch.unsqueeze(image3, 0)

    img = transforms.ToPILImage()((image2grsc[0]))


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


    gen = Generator().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(gen.parameters(), lr=0.0001, momentum=0.5)

    epochs = 100000

    for epoch in range(epochs):

        optimizer.zero_grad()

        data = image2grsc.to(device)
        output = gen(data)

        loss = criterion(output, image2.to(device))

        loss.backward()

        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch = {epoch},  Loss: {loss.item()}')


    new = gen.forward(images.to(device))

    newdiff = gen.forward(image3.to(device))

    img2 = transforms.ToPILImage()((new[0]))
    img3 = transforms.ToPILImage()((newdiff[0]))


    rows = 1
    columns = 3
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.title('Original')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img2)
    plt.title('Generated')
    fig.add_subplot(rows, columns, 3)
    plt.imshow(img3)
    plt.title('new')


    plt.show()
