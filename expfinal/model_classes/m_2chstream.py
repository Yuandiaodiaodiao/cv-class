import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class STREAM_BLOCK(nn.Module):
    def __init__(self):
        super(STREAM_BLOCK, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=96, kernel_size=5, stride=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpooling0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.maxpooling0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.flatten(x, 1)
        return x


class MODEL_2CH2STREAM(nn.Module):

    def __init__(self):
        super(MODEL_2CH2STREAM, self).__init__()
        self.fovea_input = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fovea = STREAM_BLOCK()
        self.retina_input = F.pad
        self.retina = STREAM_BLOCK()

        self.classifier = nn.Sequential(
            nn.Linear(192*2*2*2, 192*2*2),
            nn.ReLU(inplace=True),
            nn.Linear(192*2*2, 1)
        )

    def forward(self, x):
        # arrayImg = x.cpu().numpy()  # transfer tensor to array
        # plt.subplot(321)
        # plt.imshow(arrayImg[0][0],cmap='gray')  # show image
        # plt.subplot(322)
        # plt.imshow(arrayImg[0][1],cmap='gray')  # show image

        fovea = self.fovea_input(x)

        # arrayImg = fovea.cpu().numpy()  # transfer tensor to array
        # plt.subplot(323)
        # plt.imshow(arrayImg[0][0], cmap='gray')  # show image
        # plt.subplot(324)
        # plt.imshow(arrayImg[0][1], cmap='gray')  # show image

        fovea = self.fovea(fovea)

        retina = self.retina_input(x, (-16,) * 4)

        # arrayImg = retina.cpu().numpy()  # transfer tensor to array
        # plt.subplot(325)
        # plt.imshow(arrayImg[0][0], cmap='gray')  # show image
        # plt.subplot(326)
        # plt.imshow(arrayImg[0][1], cmap='gray')  # show image
        # plt.show()

        retina = self.retina(retina)

        x = torch.cat([fovea, retina], dim=1)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
