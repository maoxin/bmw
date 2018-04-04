# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        print(f'x1: {x1.shape}')
        x2 = self.down1(x1)
        print(f'x2: {x2.shape}')
        x3 = self.down2(x2)
        print(f'x3: {x3.shape}')
        x4 = self.down3(x3)
        print(f'x4: {x4.shape}')
        x5 = self.down4(x4)
        print(f'x5: {x5.shape}')
        x = self.up1(x5, x4)
        print(f'up1: {x.shape}')
        x = self.up2(x, x3)
        print(f'up2: {x.shape}')
        x = self.up3(x, x2)
        print(f'up3: {x.shape}')
        x = self.up4(x, x1)
        print(f'up4: {x.shape}')
        x = self.outc(x)
        print(f'out: {x.shape}')
        
        return x
