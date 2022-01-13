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
import os, time

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    start_time = time.time()

    Size = 64
    transform = transforms.Compose(
        [
         transforms.Resize((Size)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),


        ])

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

    original_set_tensor = images
    original_set_numpy = original_set_tensor.numpy()
    original_set_numpy = np.transpose(original_set_numpy, (2, 3, 0, 1))
    original_set_lab = rgb2lab(original_set_numpy)
    copy_of_lab = np.copy(original_set_lab).astype('float32')
    copy_of_lab = np.transpose(copy_of_lab, (2, 3, 0, 1))
    grayscale_copy_of_lab = copy_of_lab[:,0]
    grayscale_copy_of_lab = torch.from_numpy(grayscale_copy_of_lab.astype('float32'))
    grayscale_copy_of_lab = torch.unsqueeze(grayscale_copy_of_lab,1)



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


                #Downscale
                nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                #Upscale
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2)
                )

        def forward(self, input):
            return self.model(input)


    learning_rate = 0.001
    gen = Generator().to(device)

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(gen.parameters(), lr=learning_rate, momentum=0.5)
    optimizer = torch.optim.Adam(gen.parameters(),lr = learning_rate)


    epochs = 20

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
            lab_AB = np.copy(new_set_lab)
            lab_AB = lab_AB[:,[1,2]]

            lab_grayscale = torch.from_numpy(lab_grayscale.astype('float32'))
            lab_grayscale = torch.unsqueeze(lab_grayscale, 1)

            lab_target = torch.from_numpy(lab_AB.astype('float32'))


            optimizer.zero_grad()

            output = gen(lab_grayscale.to(device))

            loss = criterion(output, lab_target.to(device))

            loss.backward()

            optimizer.step()

            if i % 500 == 0:
                print(f'Epoch = {epoch}, I = {i},  Loss: {loss.item()}, Time: {time.time()-start_time}')

    PATH = './Main_Gen1.1_Model_20Epochs.pth'
    torch.save(gen.state_dict(), PATH)

    GeneratedAB_img = gen(grayscale_copy_of_lab.to(device))

    GeneratedAB_img = GeneratedAB_img.cpu()
    GeneratedAB_img = GeneratedAB_img.detach().numpy()

    merged_img = np.concatenate((grayscale_copy_of_lab,GeneratedAB_img), axis=1)


    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size*2 + index + 1)
        # plt.title(f'GRSC{index+1}')
        plt.axis('off')
        img_slice = merged_img[index]
        print_img = np.transpose(img_slice,(1,2,0))
        plt.imshow(lab2rgb(print_img))
    plt.show()









