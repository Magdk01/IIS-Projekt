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




if __name__ == '__main__':




    #input GRSC image
    filename = 'valheim.JPG'

    #Saved weights for the CNN model
    PATH = './Main_Gen1.3_Model_5Epochs_places365_valset_JB.pth'

    #Color saturation multiplier
    ColorSat_Multiplier = 1.5





    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    Size = 256
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((Size, Size))])

    class Generator(nn.Module):

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(

                # Downscale
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                # Upscale
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
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

    net = Generator().to(device)
    net.load_state_dict(torch.load(PATH))



    def load_image(RGB_image_filename):
        RGB_array = imread(RGB_image_filename)
        LAB_array = rgb2lab(RGB_array).astype('float32')

        return RGB_array, LAB_array

    def process_image(LAB_array):
        L_array = LAB_array[..., 0]
        L_tensor = transform(L_array)
        L_tensor = torch.unsqueeze(L_tensor, 0)
        L_array = L_tensor.numpy()
        Generated_AB_tensor = net.forward(L_tensor.to(device= device))


        Generated_AB_array = Generated_AB_tensor.cpu().detach().numpy()*ColorSat_Multiplier
        Merged_LAB_array = np.concatenate((L_array, Generated_AB_array), axis=1)
        Merged_LAB_array = Merged_LAB_array[0]
        Transposed_LAB_array = np.transpose(Merged_LAB_array,(1,2,0))
        Merged_RGB_array = lab2rgb(Transposed_LAB_array)


        return Merged_RGB_array

    def display_image_pair(RGB_array,new_RGB_array):
        fig = plt.figure(figsize=(10,7))
        fig.add_subplot(1,2,1)
        plt.imshow(RGB_array)
        fig.add_subplot(1, 2, 2)
        plt.imshow(new_RGB_array)

        plt.show()




    RGB_array, LAB_array = load_image(filename)

    new_RGB_array = process_image(LAB_array)

    display_image_pair(RGB_array, new_RGB_array)