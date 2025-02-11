import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES

class Residual_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):

        return self.conv_block(x)+self.conv_skip(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

@BACKBONES.register_module()
class Resunet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Resunet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Residual_block(n_channels, 64)
        self.down1 = Residual_block(64, 128)
        self.down2 = Residual_block(128, 256)
        self.down3 = Residual_block(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Residual_block(512, 1024 // factor)
        self.up1 = Residual_block(1024, 512 // factor)
        self.up2 = Residual_block(512, 256 // factor)
        self.up3 = Residual_block(256, 128 // factor)
        self.up4 = Residual_block(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dec_outs = [x5]
        x = self.up1(x5)
        dec_outs.append(x)
        x = self.up2(x)
        dec_outs.append(x)
        x = self.up3(x)
        dec_outs.append(x)
        x = self.up4(x)
        dec_outs.append(x)

        return dec_outs







