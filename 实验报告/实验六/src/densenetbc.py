import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Sequential):
    def __init__(self, num_inputs, growth_rate, bn_size, drop_rate=.15):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_inputs)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_inputs, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, features):
        cat_features = torch.cat(features, 1)
        for name, layer in self.named_children():
            cat_features = layer(cat_features)
        cat_features = F.dropout(cat_features, self.drop_rate, self.training)
        return cat_features


class DenseBlock(nn.Module):
    def __init__(self, growth_rate, depth, input_channels, drop_rate=.15, bn_size=4):
        super(DenseBlock, self).__init__()
        for i in range(depth):
            layer = DenseLayer(
                input_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,

            )
            self.add_module('layer' + str(i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Sequential):
    def __init__(self, growth_rate, depth_list, num_classes, n_out_features):
        # depth_list is a list of 4 numbers which are the number of convolutional layers in each dense block
        # growth_rate must be divisible by 2
        # n_out_features is the size of the feature vector before the final FC layer (hard to calculate)
        super(DenseNet, self).__init__()
        self.n_out_features = n_out_features
        self.mylist = []
        self.add_module('conv1', nn.Conv2d(3, 2 * growth_rate, 7, padding=3, stride=2))
        self.add_module('maxPool1', nn.MaxPool2d(3, 2, 1))
        self.add_module('dense1', DenseBlock(growth_rate, depth_list[0], 2 * growth_rate))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2',
                        nn.Conv2d(int(growth_rate * (depth_list[0] + 2)), int(growth_rate / 2 * (depth_list[0] + 2)),
                                  1))
        num_channels = int(growth_rate / 2 * (depth_list[0] + 2))
        self.add_module('avgPool1', nn.AvgPool2d(2, 2, 1))
        self.add_module('dense2', DenseBlock(growth_rate, depth_list[1], num_channels))
        num_channels = int(num_channels + growth_rate * (depth_list[1]))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', nn.Conv2d(num_channels, num_channels // 2, 1))
        num_channels = num_channels // 2
        self.add_module('avgPool2', nn.AvgPool2d(2, 2))
        self.add_module('dense3', DenseBlock(growth_rate, depth_list[2], num_channels))
        num_channels = int(num_channels + growth_rate * (depth_list[2]))
        self.add_module('relu3', nn.ReLU())
        self.add_module('conv4', nn.Conv2d(num_channels, num_channels // 2, 1))
        num_channels = num_channels // 2
        self.add_module('avgPool3', nn.AvgPool2d(2, 2, 1))
        self.add_module('dense4', DenseBlock(growth_rate, depth_list[3], num_channels))
        num_channels = int(num_channels + growth_rate * (depth_list[3]))
        self.add_module('relu4', nn.ReLU())
        self.add_module('GAP', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('linear', nn.Linear(self.n_out_features, num_classes))

    def forward(self, x):
        for name, layer in self.named_children():
            if name == 'linear':
                continue
            x = layer(x)
        x = x.view(-1, self.n_out_features)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x
