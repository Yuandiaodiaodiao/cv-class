import torch
import torch.nn as nn


class MODEL_2CH(nn.Module):

    def __init__(self):
        super(MODEL_2CH, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=96, kernel_size=7, stride=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
