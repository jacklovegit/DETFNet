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
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1
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
class AttU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.att1 = Attention_block(F_g=512 // factor, F_l=512, F_int=256 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.att2 = Attention_block(F_g=256 // factor, F_l=256, F_int=128 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.att3 = Attention_block(F_g=128 // factor, F_l=128, F_int=64 // factor)
        self.up4 = Up(128, 64, bilinear)
        self.att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dec_outs = [x5]
        x = self.up1(x5, x4)
        x = self.att1(x, x4)  # 更新后的特征图传递给 x
        dec_outs.append(x)
        x = self.up2(x, x3)
        x = self.att2(x, x3)  # 更新后的特征图传递给 x
        dec_outs.append(x)
        x = self.up3(x, x2)
        x = self.att3(x, x2)  # 更新后的特征图传递给 x
        dec_outs.append(x)
        x = self.up4(x, x1)
        x = self.att4(x, x1)  # 更新后的特征图传递给 x
        dec_outs.append(x)

        return dec_outs

class R2U_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(R2U_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.rrcnn1 = RRCNN_block(512, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.rrcnn2 = RRCNN_block(256, 256 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.rrcnn3 = RRCNN_block(128, 128 // factor)
        self.up4 = Up(128, 64, bilinear)
        self.rrcnn4 = RRCNN_block(64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        dec_outs = [x5]
        x = self.up1(x5, x4)
        x = self.rrcnn1(x)
        dec_outs.append(x)
        x = self.up2(x, x3)
        x = self.rrcnn2(x)
        dec_outs.append(x)
        x = self.up3(x, x2)
        x = self.rrcnn3(x)
        dec_outs.append(x)
        x = self.up4(x, x1)
        x = self.rrcnn4(x)
        dec_outs.append(x)

        return dec_outs







