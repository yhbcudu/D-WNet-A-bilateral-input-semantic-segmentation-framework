import torch
from torch import nn
from torch.nn import functional as F
import random, os
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score as sklearn_f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import warnings

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 2, 0, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1, bias=False)

    def forward(self, x, feature_map):
        up = F.interpolate(x, size=(feature_map.shape[2], feature_map.shape[3]), mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self, num):
        super(UNet, self).__init__()
        self.c1 = Conv_Block(num, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.d5 = DownSample(1024)
        self.c6 = Conv_Block(1024, 2048)
        self.u1 = UpSample(2048)
        self.c7 = Conv_Block(2048, 1024)
        self.u2 = UpSample(1024)
        self.c8 = Conv_Block(1024, 512)
        self.u3 = UpSample(512)
        self.c9 = Conv_Block(512, 256)
        self.u4 = UpSample(256)
        self.c10 = Conv_Block(256, 128)
        self.u5 = UpSample(128)
        self.c11 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, 23, 3, padding=1)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R6 = self.c6(self.d5(R5))
        O1 = self.c7(self.u1(R6, R5))
        O2 = self.c8(self.u2(O1, R4))
        O3 = self.c9(self.u3(O2, R3))
        O4 = self.c10(self.u4(O3, R2))
        O5 = self.c11(self.u5(O4, R1))

        return self.out(O5)
