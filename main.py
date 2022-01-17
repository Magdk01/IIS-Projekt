import cv2
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
from torch.utils.data import Dataset

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])

# trainset = torchvision.datasets.Places365(root='./data', split='val', small=True, download=False,
#                                           transform=transform)

image_path = './data/val_256_new'
train_image_paths = []
for pictures in os.listdir(image_path):
    train_image_paths.append(os.path.join(image_path, pictures))


class CustomDataset123(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert('RGB')

        label = 1
        if self.transform is not None:
            image = self.transform(image)

        return image, label


trainset = CustomDataset123(image_paths=train_image_paths, transform=transform)

batch_size = 4

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    start_time = time.time()

    dataiter = (iter(trainloader))
    images, labels = dataiter.next()

    original_set_tensor = images
    original_set_numpy = original_set_tensor.numpy()
    original_set_numpy = np.transpose(original_set_numpy, (2, 3, 0, 1))
    original_set_lab = rgb2lab(original_set_numpy)
    copy_of_lab = np.copy(original_set_lab).astype('float32')
    copy_of_lab = np.transpose(copy_of_lab, (2, 3, 0, 1))
    grayscale_copy_of_lab = copy_of_lab[:, 0]
    grayscale_copy_of_lab = torch.from_numpy(grayscale_copy_of_lab.astype('float32'))
    grayscale_copy_of_lab = torch.unsqueeze(grayscale_copy_of_lab, 1)


    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(

                # Downsample
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                # Upsample
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
    # optimizer = torch.optim.SGD(gen.parameters(), lr=learning_rate, momentum=0.5)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)
    epochs = 1
    total_loss = 0
    epoch_loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0

        for i, data in enumerate(trainloader):

            inp_img_inside, labels = data

            original_set_tensor_inside = inp_img_inside
            original_set_numpy_inside = original_set_tensor_inside.numpy()
            original_set_numpy_inside = np.transpose(original_set_numpy_inside, (2, 3, 0, 1))
            original_set_lab_inside = rgb2lab(original_set_numpy_inside)
            new_set_lab_inside = np.transpose(original_set_lab_inside, (2, 3, 0, 1))
            lab_grayscale_inside = np.copy(new_set_lab_inside)
            lab_grayscale_inside = lab_grayscale_inside[:, 0]
            lab_AB_inside = np.copy(new_set_lab_inside)
            lab_AB_inside = lab_AB_inside[:, [1, 2]]

            lab_grayscale_inside = torch.from_numpy(lab_grayscale_inside.astype('float32'))
            lab_grayscale_inside = torch.unsqueeze(lab_grayscale_inside, 1)

            lab_target_inside = torch.from_numpy(lab_AB_inside.astype('float32'))

            optimizer.zero_grad()

            output = gen(lab_grayscale_inside.to(device))

            loss = criterion(output, lab_target_inside.to(device))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            epoch_loss += loss.item()

            if i % 250 == 0:
                print(f'Epoch = {epoch}, I = {i},  Loss: {total_loss / 500}, Time: {time.time() - start_time}')
                total_loss = 0
        epoch_loss_list.append(epoch_loss)
        plt.plot(epoch_loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.show()


    PATH = './Main_Gen1.1_Model_1Epoch_hackeddataset_MSVN.pth'
    torch.save(gen.state_dict(), PATH)

    GeneratedAB_img = gen(grayscale_copy_of_lab.to(device))

    GeneratedAB_img = GeneratedAB_img.cpu()
    GeneratedAB_img = GeneratedAB_img.detach().numpy()

    merged_img = np.concatenate((grayscale_copy_of_lab, GeneratedAB_img), axis=1)




    fig = plt.figure(figsize=(10, 7))

    for index in range(batch_size):
        fig.add_subplot(3, batch_size, index + 1)
        plt.axis('off')
        plt.imshow(original_set_numpy[:, :, index])

    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size + index + 1)
        plt.axis('off')
        plt.imshow(original_set_numpy[:, :, index, 0], cmap='gray')

    for index in range(batch_size):
        fig.add_subplot(3, batch_size, batch_size * 2 + index + 1)
        plt.axis('off')
        img_slice = merged_img[index]
        print_img = np.transpose(img_slice, (1, 2, 0))
        plt.imshow(lab2rgb(print_img))
    plt.show()
