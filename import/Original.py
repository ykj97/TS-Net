import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size
                               =kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        """Identity Mapping"""
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, padding=0)

        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        s = self.identity(inputs)
        skip = x + s
        out = self.relu2(skip)

        return out


class Original(nn.Module):
    def __init__(self):
        super(Original, self).__init__()

        """Encoder"""

        self.enc1 = Residual_Block(in_channels=1, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = Residual_Block(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = Residual_Block(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = Residual_Block(in_channels=256, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        """Bridge"""
        self.bridge = Residual_Block(in_channels=512, out_channels=1024)

        """Decoder"""
        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4 = Residual_Block(in_channels=2 * 512, out_channels=512)

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3 = Residual_Block(in_channels=2 * 256, out_channels=256)

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2 = Residual_Block(in_channels=2 * 128, out_channels=128)

        self.unpool1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1 = Residual_Block(in_channels=2 * 64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        enc1 = self.enc1(inputs)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        bridge = self.bridge(pool4)

        unpool4 = self.unpool4(bridge)

        cat4 = torch.cat((unpool4, enc4), dim=1)  # concatnate / dim = [0:batch, 1:channel, 2:height, 3:width]
        dec4 = self.dec4(cat4)

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat((unpool3, enc3), dim=1)
        dec3 = self.dec3(cat3)

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat((unpool2, enc2), dim=1)
        dec2 = self.dec2(cat2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat((unpool1, enc1), dim=1)
        dec1 = self.dec1(cat1)

        x = self.fc(dec1)
        x = self.sigmoid(x)
        return x

    
