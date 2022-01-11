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

    batch_size = 1

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

    original_set_tensor = images
    original_set_numpy = original_set_tensor.numpy()
    original_set_numpy = np.transpose(original_set_numpy, (2, 3, 0, 1))
    original_set_lab = rgb2lab(original_set_numpy)
    copy_of_lab = np.copy(original_set_lab).astype('float32')
    copy_of_lab = np.transpose(copy_of_lab, (2, 3, 0, 1))
    copy_of_lab = copy_of_lab[:,0]
    copy_of_lab = torch.from_numpy(copy_of_lab.astype('float32'))
    copy_of_lab = torch.unsqueeze(copy_of_lab,1)

    fig = plt.figure(figsize=(10, 7))

    for index in range(batch_size):
        fig.add_subplot(3, batch_size, index + 1)
        # plt.title(f'Original{index + 1}')
        plt.axis('off')
        plt.imshow(original_set_numpy[:,:,index])

    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size + index + 1)
        # plt.title(f'GRSC{index+1}')
        plt.axis('off')
        plt.imshow(original_set_numpy[:,:,index,0],cmap='gray')


    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(1, 255, 1, padding='same'),
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
    optimizer = torch.optim.Adam(gen.parameters(),lr = learning_rate)


    epochs = 1

    for epoch in range(epochs):

        for i, data in enumerate(trainloader):

            inp_img, labels = data

            original_set_tensor = inp_img
            original_set_numpy = original_set_tensor.numpy()
            original_set_numpy = np.transpose(original_set_numpy, (2, 3, 0, 1))
            original_set_lab = rgb2lab(original_set_numpy)
            new_set_lab = np.transpose(original_set_lab, (2, 3, 0, 1))
            lab_grayscale = np.copy(new_set_lab)
            lab_grayscale = lab_grayscale[:,0]

            lab_grayscale = torch.from_numpy(lab_grayscale.astype('float32'))
            lab_grayscale = torch.unsqueeze(lab_grayscale, 1)

            lab_target = torch.from_numpy(new_set_lab.astype('float32'))


            optimizer.zero_grad()

            output = gen(lab_grayscale.to(device))

            loss = criterion(output, lab_target.to(device))

            loss.backward()

            optimizer.step()

            if i % 500 == 0:
                print(f'Epoch = {epoch}, I = {i},  Loss: {loss.item()}')



    epoch_images = gen(copy_of_lab.to(device))

    epoch_images = epoch_images.cpu()
    epoch_images = epoch_images.detach().numpy()


    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size*2 + index + 1)
        # plt.title(f'GRSC{index+1}')
        plt.axis('off')
        img_slice = epoch_images[index]
        print_img = np.transpose(img_slice,(1,2,0))
        plt.imshow(lab2rgb(print_img))
    plt.show()









